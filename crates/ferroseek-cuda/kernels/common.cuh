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

// f32 → e4m3fn bits: round-to-nearest-even on the 3-bit mantissa,
// saturate to 448 (finite-only variant, no inf).
__device__ __forceinline__ uint8_t f32_to_e4m3_bits(float v) {
    uint8_t sign = (__float_as_uint(v) >> 24) & 0x80;
    float a = fminf(fabsf(v), 448.0f);
    if (a == 0.0f) return sign;
    int exp;
    float m = frexpf(a, &exp); // a = m * 2^exp, m in [0.5, 1)
    int e = exp - 1 + 7;
    if (e <= 0) {
        // subnormal quantum 2^-9; 7.5+ rounds up to the smallest normal
        int mant = (int)rintf(a * 512.0f);
        if (mant >= 8) return sign | 0x08; // e=1, m=0 → 2^-6
        return sign | (uint8_t)mant;
    }
    int mant = (int)rintf((m * 2.0f - 1.0f) * 8.0f);
    if (mant == 8) { mant = 0; e += 1; }
    // E=15 with mant 0..6 is the valid 256..448 range (448 = S.1111.110);
    // only the carry past E=15 saturates (unreachable after the 448 clamp,
    // kept for safety). Saturating at e>=15 instead would snap every value
    // in (256, 448) to 448, diverging from kernel.py and the CPU reference.
    if (e >= 16) return sign | 0x7E; // 448
    return sign | (uint8_t)((e << 3) | mant);
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
