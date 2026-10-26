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
