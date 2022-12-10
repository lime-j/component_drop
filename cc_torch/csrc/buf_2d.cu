#include "buf.h"

// 2d
#define BLOCK_ROWS 32
#define BLOCK_COLS 32
#define BLOCK_BATCHES 1 // You may change this for better performance, but for me, this is what i can have for my GPU
#define DROP_THRESH 0.2

struct comp_int32{
    __host__ __device__ bool operator() (int32_t x, int32_t y) {return x < y;}
};


struct identity_functor{
    __device__ int32_t operator()(int idx){
        return idx;
    }
};

struct sort_functor{

    thrust::device_ptr<int32_t> data;
    int dsize;
    __host__ __device__ void operator()(int start_idx, int end_idx){
        thrust::sort(thrust::device, data + (dsize) * start_idx, data + (dsize) * end_idx + 1);
    }
};

namespace label_collecting{
/*
    This namespace contains impl collecting labels into a final set;
 */
    __global__ void finalize_mask(const int32_t *label, const int32_t *thresh_idx, const int32_t* sorted_idx, 
                                  uint8_t* visit_map, const int32_t H, const int32_t W, const int32_t N, const int32_t max_count){
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t batch = (blockIdx.z * blockDim.z + threadIdx.z); 
        
        const uint32_t visit_pos = batch * H * W + row * W + col;

        if (row >= H || col >= W || batch >= N) return ;
        
        const int32_t r_limit = thresh_idx[batch];
        
        for (int i = 0; i <= r_limit; ++ i){
            visit_map[visit_pos] &=  label[visit_pos] && (label[visit_pos] != 1 + sorted_idx[batch * max_count + i]);
            visit_map[visit_pos + 1] &= label[visit_pos + 1] && (label[visit_pos + 1] != 1 + sorted_idx[batch * max_count + i]);
            visit_map[visit_pos + W] &= label[visit_pos + W] && (label[visit_pos + W] != 1 + sorted_idx[batch * max_count + i]);
            visit_map[visit_pos + W + 1] &= label[visit_pos + 1 + W]&& (label[visit_pos + W + 1] != 1 + sorted_idx[batch * max_count + i]);
        }
        
    }
    __global__ void index(int32_t* dist, const int64_t* idx, const int32_t* src, const int32_t M, const int32_t N){
        const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t m = blockIdx.y * blockDim.y + threadIdx.y;
        if (n >= N || m >= M) return ;
        const uint32_t to_id = n * M + m;
        dist[to_id] = src[idx[to_id] + n * M];
        //atomicAdd(dist + M - 1, 1);
    }

    __global__ void calc_thresh(const int32_t* size, int32_t *result, const int32_t N, const int max_size, const float lam){
        const uint32_t start_point = blockIdx.x * blockDim.x + threadIdx.x;
        if (start_point >= N) return;
        result[start_point] = int(float(size[(start_point + 1)* max_size - 1]) * (DROP_THRESH - lam) / DROP_THRESH);
    } 


    __global__ void collect(const int32_t* label, const int32_t* size, 
                            int32_t* result_idx, int32_t* result_size, int32_t *result_count, 
                            const uint32_t max_count, const uint32_t W, const uint32_t H, const uint32_t N){
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t batch = (blockIdx.z * blockDim.z + threadIdx.z);
        
        if (row >= H || col >= W || batch >= N) return;

        const uint32_t curr_idx = row * W + col;
        const uint32_t batch_delta = batch * H * W; 
        
        //const int32_t label_lu = label[batch_delta + curr_idx];
        int32_t cur_flg = -1;
        if (label[curr_idx + batch_delta] == curr_idx + 1) {
            cur_flg = atomicAdd(result_count + batch , 1);
            result_size[batch * max_count + cur_flg] = size[curr_idx + batch_delta];
        }else if (label[curr_idx + 1 + batch_delta] == curr_idx + 1){
            cur_flg = atomicAdd(result_count + batch, 1);
            result_size[batch * max_count + cur_flg] = size[curr_idx + 1 + batch_delta];
        }else if (label[curr_idx + 1 + W + batch_delta] == curr_idx + 1){
            cur_flg = atomicAdd(result_count + batch, 1);
            result_size[batch * max_count + cur_flg] = size[curr_idx + 1 + W + batch_delta];
        }else if (label[curr_idx + W + batch_delta] == curr_idx + 1){
            cur_flg = atomicAdd(result_count + batch, 1);
            result_size[batch * max_count + cur_flg] = size[curr_idx + W + batch_delta];
        }else return ;
        result_idx[batch * max_count + cur_flg] = curr_idx;
    }
    
}

namespace cc2d {
    __global__ void
    init_labeling(int32_t *label, const uint32_t W, const uint32_t H, const uint32_t N) {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t batch = (blockIdx.z * blockDim.z + threadIdx.z);
        if (batch >= N) return;

        const uint32_t idx = batch * H * W + row * W + col;

        if (row < H && col < W) {
            label[idx] = idx;
        }

    }

    __global__ void init_sizing(const uint8_t *img, int32_t *size, const uint32_t W, const uint32_t H,
                                const uint32_t N) {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t batch = (blockIdx.z * blockDim.z + threadIdx.z);
        if (batch >= N) return;

        const uint32_t idx = batch * H * W + row * W + col;

        if (row < H && col < W) {
            size[idx] = (img[idx] > 0 ? 1 : 0) + (img[idx + 1] > 0 ? 1 : 0) + (img[idx + W] > 0 ? 1 : 0) +
                               (img[idx + 1 + W] > 0 ? 1 : 0);
            size[idx + 1] = size[idx + W] = size[idx + 1 + W] = size[idx] = size[idx];
        }

    }

    __global__ void
    merge(uint8_t *img, int32_t *label, const uint32_t W, const uint32_t H, const uint32_t N) {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t batch = (blockIdx.z * blockDim.z + threadIdx.z);
        if (batch >= N) return;
        const uint32_t idx = batch * H * W + row * W + col;

        if (row >= H || col >= W) return;

        uint32_t P = 0;

        // NOTE : Original Codes, but occurs silent error
        // NOTE : Programs keep runnig, but now showing printf logs, and the result is weird
        // uint8_t buffer[4] = {0};
        // if (col + 1 < W) {
        //     *(reinterpret_cast<uint16_t*>(buffer)) = *(reinterpret_cast<uint16_t*>(img + idx));
        //     if (row + 1 < H) {
        //         *(reinterpret_cast<uint16_t*>(buffer + 2)) = *(reinterpret_cast<uint16_t*>(img + idx + W));
        //     }
        // }
        // else {
        //     buffer[0] = img[idx];
        //     if (row + 1 < H)
        //         buffer[2] = img[idx + W];
        // }
        // if (buffer[0])              P |= 0x777;
        // if (buffer[1])              P |= (0x777 << 1);
        // if (buffer[2])              P |= (0x777 << 4);

        if (img[idx]) P |= 0x777;      // 0000 0111 0111 0111
        if (row + 1 < H && img[idx + W]) P |= 0x777 << 4; // 0111 0111 0111 0000
        if (col + 1 < W && img[idx + 1]) P |= 0x777 << 1; // 0000 1110 1110 1110

        if (col == 0) P &= 0xEEEE;            // 1110 1110 1110 1110
        if (col + 1 >= W) P &= 0x3333;            // 0011 0011 0011 0011
        else if (col + 2 >= W) P &= 0x7777;            // 0111 0111 0111 0111

        if (row == 0) P &= 0xFFF0;            // 1111 1111 1111 0000
        if (row + 1 >= H) P &= 0xFF;              // 0000 0000 1111 1111

        if (P > 0) {
            // If need check about top-left pixel(if flag the first bit) and hit the top-left pixel
            if (hasBit(P, 0) && img[idx - W - 1]) {
                union_(label, idx, idx - 2 * W - 2); // top left block
            }

            if ((hasBit(P, 1) && img[idx - W]) || (hasBit(P, 2) && img[idx - W + 1]))
                union_(label, idx, idx - 2 * W); // top bottom block

            if (hasBit(P, 3) && img[idx + 2 - W])
                union_(label, idx, idx - 2 * W + 2); // top right block

            if ((hasBit(P, 4) && img[idx - 1]) || (hasBit(P, 8) && img[idx + W - 1]))
                union_(label, idx, idx - 2); // just left block
        }
    }

    __global__ void compression(int32_t *label, int32_t *size, const int32_t W, const int32_t H, const uint32_t N) {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t batch = (blockIdx.z * blockDim.z + threadIdx.z);
        if (batch >= N) return;

        const uint32_t idx = batch * H * W + row * W + col;


        if (row < H && col < W)
            find_n_compress(label, size, idx);
    }

    __global__ void
    final_labeling(const uint8_t *img, int32_t *label, int32_t *size, int32_t *count, const int32_t W,
                   const int32_t H, const uint32_t N) {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t batch = (blockIdx.z * blockDim.z + threadIdx.z);
        if (batch >= N) return;
        const uint32_t delta = batch * H * W;
        const uint32_t idx = batch * H * W + row * W + col;
        if (row >= H || col >= W)
            return;
        
        //bool is_father = false;
        //int32_t father_id = -1;
        const int32_t y = label[idx] + 1;
        int32_t block_size = size[idx];
        if (idx == label[idx])  {
            if (size[idx]) atomicAdd(count + batch, 1); 
            block_size -= size[idx + 1];
        }
        if (img[idx]) {
            label[idx] = y - delta;
            size[idx] = block_size;
        }
        else {
            label[idx] = 0;
            size[idx] = 0;
        }
        if (col + 1 < W) {
            if (img[idx + 1]) {
                label[idx + 1] = y - delta;
                size[idx + 1] = block_size;
            }else {
                label[idx + 1] = 0;
                size[idx + 1] = 0;
            }

            if (row + 1 < H) {
                if (img[idx + W + 1]) {
                    label[idx + W + 1] = y - delta;
                    size[idx + W + 1] = block_size;
                }else {
                    label[idx + W + 1] = 0;
                    size[idx + W + 1] = 0;
                }
            }
        }

        if (row + 1 < H) {
            if (img[idx + W]) {
                label[idx + W] = y - delta;
                size[idx + W] = block_size;
            }else {
                label[idx + W] = 0;
                size[idx + W] = 0;
            }
        }
    }

} // namespace cc2d

torch::Tensor connected_componnets_labeling_2d(const torch::Tensor &input, const torch::Tensor &lam) {
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(input.ndimension() == 3, "input must be a [N, H, W] shape, where N is the batch dim");
    AT_ASSERTM(input.scalar_type() == torch::kUInt8, "input must be a uint8 type tensor");
    //AT_ASSERTM(lam <= DROP_THRESH + 1e-10, "lam must smaller than drop thresh!");
    //std::cerr << "Input" << input << std::endl;
    const uint32_t N = input.size(0);
    const uint32_t H = input.size(-2);
    const uint32_t W = input.size(-1);

    AT_ASSERTM(!(H % 2), "H must be an even number");
    AT_ASSERTM(!(W % 2), "W must be an even number");
    
    // label must be uint32_t
    auto on_device_i32_config = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    auto on_host_i32_config = torch::TensorOptions().dtype(torch::kInt32);
    auto on_device_f32_config = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    
    auto on_device_u8_config = torch::TensorOptions().dtype(torch::kU8).device(input.device());
    
    torch::Tensor count = torch::zeros({N, 1}, on_device_i32_config);
    torch::Tensor label = torch::zeros({N, H, W}, on_device_i32_config);
    torch::Tensor size = torch::zeros({N, H, W}, on_device_i32_config);
    dim3 grid = dim3(((W + 1) / 2 + BLOCK_COLS - 1) / BLOCK_COLS, ((H + 1) / 2 + BLOCK_ROWS - 1) / BLOCK_ROWS,
                     (N + BLOCK_BATCHES - 1) / BLOCK_BATCHES);
    dim3 block = dim3(BLOCK_COLS, BLOCK_ROWS, BLOCK_BATCHES);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cc2d::init_sizing<<<grid, block, 0, stream>>>(
                    input.data_ptr<uint8_t>(),
                    size.data_ptr<int32_t>(),
                    W, H, N
    );
    
    cc2d::init_labeling<<<grid, block, 0, stream>>>(
            label.data_ptr<int32_t>(),
                    W, H, N
    );
    cc2d::init_labeling<<<grid, block, 0, stream>>>(
                    label.data_ptr<int32_t>(),
                    W, H, N
    );

    cc2d::merge<<<grid, block, 0, stream>>>(
                    input.data_ptr<uint8_t>(),
                    label.data_ptr<int32_t>(),
                    W, H, N
    );
    cc2d::compression<<<grid, block, 0, stream>>>(
            label.data_ptr<int32_t>(),
                    size.data_ptr<int32_t>(),
                    W, H, N
    );
    cc2d::final_labeling<<<grid, block, 0, stream>>>(
                    input.data_ptr<uint8_t>(),
                    label.data_ptr<int32_t>(),
                    size.data_ptr<int32_t>(),
                    count.data_ptr<int32_t>(),
                    W, H, N
    );
    //std::cerr << label << std::endl;
    auto max_count_tup = torch::max(count, 0);
    auto max_count = std::get<0>(max_count_tup)[0].item<int>();
    torch::Tensor count_tmp = torch::zeros({N}, on_device_i32_config);
    torch::Tensor count_size = torch::zeros({N, max_count}, on_device_i32_config) + 192608170; 
    // NOTE: >= 1920 * 1080 * 64, safe enough (for me)
    torch::Tensor count_idx = torch::zeros({N, max_count}, on_device_i32_config);
    label_collecting::collect<<<grid, block, 0, stream>>>(
                    label.data_ptr<int32_t>(), 
                    size.data_ptr<int32_t>(),
                    count_idx.data_ptr<int32_t>(),
                    count_size.data_ptr<int32_t>(),
                    count_tmp.data_ptr<int32_t>(),
                    max_count, W, H, N
    );
    auto sort_ret_pir = count_size.sort(1);
    
    torch::Tensor sorted_count = std::get<0>(sort_ret_pir), sort_idx = std::get<1>(sort_ret_pir);
    sorted_count.index_put_({sorted_count == 192608170}, 0);
    //std::cerr << "count_size" << count_size << std::endl;
    //std::cerr << "count_idx" << count_idx << std::endl;
    //std::cerr << "sorted_count" << sorted_count << std::endl;
    //std::cerr << "sort_idx" << sort_idx << std::endl;
    torch::Tensor sorted_idx = torch::zeros({N, max_count}, on_device_i32_config);
    //std::cerr << N << " " << max_count << std::endl;
    dim3 grid_n = dim3(N, (max_count + 1023) / 1024);
    dim3 block_max_count = dim3(1, 1024);

    label_collecting::index<<<grid_n, block_max_count, 0, stream>>>(
            sorted_idx.data_ptr<int32_t>(),
            sort_idx.data_ptr<int64_t>(),
            count_idx.data_ptr<int32_t>(), max_count, N
    );

    //count_idx.index({sort_idx});
    torch::Tensor curr_sorted_count = sorted_count.cumsum(1).to(torch::kInt32); 
    
    dim3 grid_1 = dim3(1);
    dim3 block_thread = dim3(((1024 + N) / 1024) * 1024);
    const float lamb = lam.item<float>();
    //std::cerr << lamb << std::endl;
    torch::Tensor thresh = torch::zeros({N}, on_device_i32_config); 
    
    label_collecting::calc_thresh<<<grid_1, block_thread, 0, stream>>> (
                    curr_sorted_count.data_ptr<int32_t>(),
                    thresh.data_ptr<int32_t>(),
                    N, max_count, lamb
    );
    
    std::vector<int> thresh_idx(N);
    for (int i = 0; i < N; ++ i){
        const int32_t start_idx = i * max_count, end_idx = (i + 1) * max_count;
        auto ptr = thrust::lower_bound(thrust::device, curr_sorted_count.data_ptr<int32_t>() + start_idx, curr_sorted_count.data_ptr<int32_t>() + end_idx, thresh[i].item<int>());
        thresh_idx[i] = thrust::distance(curr_sorted_count.data_ptr<int32_t>() + start_idx, ptr) - 1;
        thresh_idx[i] = thresh_idx[i] > 0 ? thresh_idx[i] : 0;
        //std::cerr << thresh_idx[i] << std::endl;
        
    }
    torch::Tensor thresh_idx_device = torch::from_blob(thresh_idx.data(), {N}, on_host_i32_config).to(torch::kCUDA);
   
    torch::Tensor visit_map = torch::ones({N, H, W}, on_device_u8_config);
    label_collecting::finalize_mask<<<grid, block, 0, stream>>>(
            label.data_ptr<int32_t>(),
            thresh_idx_device.data_ptr<int32_t>(),
            sorted_idx.data_ptr<int32_t>(),
            visit_map.data_ptr<uint8_t>(),
            H, W, N, max_count
    );
    return visit_map;
}
