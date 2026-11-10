// Small-projection GEMVs: fp32 and bf16 weights (gate, weights_proj,
// compressor wkv/wgate, indexer wk). One block per output row.

#include "common.cuh"
#include <cuda_bf16.h>

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

// FP8 WO-A with f32 activations. Each weight is dequantized and rounded to
// BF16 exactly as the host-side loader did, then multiplied in f32. This
// preserves the BF16 GEMV's numerics while halving weight traffic.
extern "C" __global__ void fp8_wo_a_gemv(
    const uint8_t* __restrict__ w,  // [rows, cols] e4m3
    const uint8_t* __restrict__ s,  // ue8m0 block scales
    const float* __restrict__ x,    // [cols]
    float* __restrict__ y,          // [rows]
    int rows, int cols, int block, int scale_cols)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const uint8_t* wr = w + (size_t)r * cols;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float scale = e8m0_to_f32(s[(r / block) * scale_cols + c / block]);
        __nv_bfloat16 rounded = __float2bfloat16(e4m3_to_f32(wr[c]) * scale);
        acc += __bfloat162float(rounded) * x[c];
    }
    block_reduce_sum(acc, &y[r]);
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
