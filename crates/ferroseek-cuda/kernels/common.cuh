// Shared device helpers for ferroseek kernels.
#pragma once
#include <cstdint>
#include <cuda_fp8.h>

__device__ __forceinline__ float e8m0_to_f32(uint8_t e) {
    return exp2f((float)e - 127.0f);
}

__device__ __forceinline__ float e4m3_to_f32(uint8_t v) {
    return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3));
}

__device__ __constant__ float FP4_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Warp+block reduction of a per-thread partial into out (thread 0 writes).
__device__ __forceinline__ void block_reduce_sum(float partial, float* out) {
    __shared__ float buf[256];
    buf[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = buf[0];
}
