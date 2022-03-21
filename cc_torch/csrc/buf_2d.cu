#include "buf.h"

// 2d
#define BLOCK_ROWS 16
#define BLOCK_COLS 16
#define BLOCK_BATCHES 4 // You may change this for better performance, but for me, this is what i can have for my GPU


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
    final_labeling(const uint8_t *img, int32_t *label, int32_t *size, const int32_t W,
                   const int32_t H, const uint32_t N) {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t batch = (blockIdx.z * blockDim.z + threadIdx.z);
        if (batch >= N) return;

        const uint32_t idx = batch * H * W + row * W + col;
        if (row >= H || col >= W)
            return;

        const int32_t y = label[idx] + 1;
        int32_t block_size = size[idx];
        if (idx == label[idx])  block_size -= size[idx + 1];
        
        if (img[idx]) {
            label[idx] = y;
            size[idx] = block_size;
        }
        else {
            label[idx] = 0;
            size[idx] = 0;
        }
        if (col + 1 < W) {
            if (img[idx + 1]) {
                label[idx + 1] = y;
                size[idx + 1] = block_size;
            }else {
                label[idx + 1] = 0;
                size[idx + 1] = 0;
            }

            if (row + 1 < H) {
                if (img[idx + W + 1]) {
                    label[idx + W + 1] = y;
                    size[idx + W + 1] = block_size;
                }else {
                    label[idx + W + 1] = 0;
                    size[idx + W + 1] = 0;
                }
            }
        }

        if (row + 1 < H) {
            if (img[idx + W]) {
                label[idx + W] = y;
                size[idx + W] = block_size;
            }else {
                label[idx + W] = 0;
                size[idx + W] = 0;
            }
        }
    }

} // namespace cc2d

std::vector <torch::Tensor> connected_componnets_labeling_2d(const torch::Tensor &input) {
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(input.ndimension() == 3, "input must be a [N, H, W] shape, where N is the batch dim");
    AT_ASSERTM(input.scalar_type() == torch::kUInt8, "input must be a uint8 type");

    const uint32_t N = input.size(0);
    const uint32_t H = input.size(-2);
    const uint32_t W = input.size(-1);

    AT_ASSERTM((H % 2) == 0, "shape must be a even number");
    AT_ASSERTM((W % 2) == 0, "shape must be a even number");

    // label must be uint32_t
    auto on_device_i32_config = torch::TensorOptions().dtype(torch::kInt32).device(input.device());

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
                    W, H, N
    );


    return std::vector < torch::Tensor > {label, size};
}
