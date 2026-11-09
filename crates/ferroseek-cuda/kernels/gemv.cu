// Small-projection GEMVs: fp32 and bf16 weights (gate, weights_proj,
// compressor wkv/wgate, indexer wk). One block per output row.

#include "common.cuh"

extern "C" __global__ void f32_gemv(
    const float* __restrict__ w,   // [rows, cols]
    const float* __restrict__ x,   // [cols]
    float* __restrict__ y,         // [rows]
    int rows, int cols)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* wr = w + (size_t)r * cols;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) acc += wr[c] * x[c];
    block_reduce_sum(acc, &y[r]);
}

__device__ __forceinline__ float bf16_to_f32d(uint16_t v) {
    return __uint_as_float((uint32_t)v << 16);
}

extern "C" __global__ void bf16_gemv(
    const uint16_t* __restrict__ w,   // [rows, cols] bf16
    const float* __restrict__ x,      // [cols]
    float* __restrict__ y,            // [rows]
    int rows, int cols)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const uint16_t* wr = w + (size_t)r * cols;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) acc += bf16_to_f32d(wr[c]) * x[c];
    block_reduce_sum(acc, &y[r]);
}

// Up to five rows share one weight read. Each output keeps bf16_gemv's
// per-thread accumulation and block reduction order, so batching does not
// change the FP32 result. x/y strides cover WO-A groups and vocab shards.
extern "C" __global__ void bf16_gemm_m5(
    const uint16_t* __restrict__ w,   // [rows, cols] bf16
    const float* __restrict__ x,      // [m, x_stride]
    float* __restrict__ y,            // [m, y_stride]
    int m, int rows, int cols, int x_stride, int y_stride)
{
    int r = blockIdx.x;
    if (r >= rows || m < 1 || m > 5) return;
    const uint16_t* wr = w + (size_t)r * cols;
    float acc[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float wv = bf16_to_f32d(wr[c]);
#pragma unroll
        for (int i = 0; i < 5; i++) {
            if (i < m) acc[i] += wv * x[i * x_stride + c];
        }
    }
    __shared__ float reduce[5][256];
#pragma unroll
    for (int i = 0; i < 5; i++) {
        if (i < m) {
            reduce[i][threadIdx.x] = acc[i];
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) reduce[i][threadIdx.x] += reduce[i][threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) y[i * y_stride + r] = reduce[i][0];
            __syncthreads();
        }
    }
}

// y[i] += x[i] — the fp32 MoE accumulator gathers expert outputs
// (model-spec §6: experts added in ascending expert-id order).
extern "C" __global__ void acc_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += x[i];
}
