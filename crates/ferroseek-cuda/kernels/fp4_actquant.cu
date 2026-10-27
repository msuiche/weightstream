// fp4_act_quant snap (model-spec §7, kernel.py:127-204), two variants:
//   scale_e4m3 = 0 (indexer q/k): block 32, amax floor 6*2^-126,
//                s = 2^ceil(log2(amax/6))          (power-of-2)
//   scale_e4m3 = 1 (compressed KV): block 16, amax floor 6*2^-9,
//                s = e4m3(amax/6)                  (NOT power-of-2)
// Snap = quantize to fp4 then dequantize back to f32, in place.

#include "common.cuh"
#include <cuda_fp4.h>

__device__ __forceinline__ float f32_to_fp4(float v) {
    __nv_fp4_storage_t q = __nv_cvt_float_to_fp4(v, __NV_E2M1, cudaRoundNearest);
    return __half2float(__nv_cvt_fp4_to_halfraw(q, __NV_E2M1));
}

extern "C" __global__ void fp4_act_quant_snap(
    float* __restrict__ x,
    int n,            // multiple of block_size
    int block_size,   // 32 (indexer) or 16 (compressed KV)
    int scale_e4m3)
{
    int group = blockIdx.x;
    float* g = x + (size_t)group * block_size;

    __shared__ float partial[32];
    float amax = 0.0f;
    if (threadIdx.x < block_size) amax = fabsf(g[threadIdx.x]);
    partial[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + s]);
        __syncthreads();
    }
    amax = partial[0];

    float s;
    if (scale_e4m3) {
        amax = fmaxf(amax, 6.0f * 0x1p-9f);
        s = e4m3_to_f32(f32_to_e4m3_bits(amax / 6.0f));
    } else {
        amax = fmaxf(amax, 6.0f * 0x1p-126f);
        // bit-form ceil(log2): matches the CPU log2_ceil exactly
        // (log2f+ceilf can round across an integer boundary)
        int bits = __float_as_int(amax / 6.0f);
        int e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
        s = exp2f((float)e);
    }

    if (threadIdx.x < block_size) {
        float y = g[threadIdx.x] / s;
        y = fminf(fmaxf(y, -6.0f), 6.0f);
        g[threadIdx.x] = f32_to_fp4(y) * s;
    }
}
