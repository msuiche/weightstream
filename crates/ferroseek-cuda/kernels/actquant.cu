// act_quant ue8m0 snap (model-spec §7): per 32-element group,
// amax floored at 1e-4, scale = 2^ceil(log2(amax/448)), clamp ±448,
// fp8 e4m3 quant then dequant back to f32 ("inplace" snap).
// The gate expects bit-exact match with the CPU reference (all ops are
// exact in fp32 except the fp8 round, which is deterministic).

#include "common.cuh"

extern "C" __global__ void act_quant_ue8m0_snap(
    float* __restrict__ x,
    int n) // n must be a multiple of 32
{
    int group = blockIdx.x;
    const float* g = x + (size_t)group * 32;

    __shared__ float partial[32];
    float amax = 0.0f;
    if (threadIdx.x < 32) amax = fabsf(g[threadIdx.x]);
    partial[threadIdx.x] = amax;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + s]);
        __syncthreads();
    }
    amax = fmaxf(partial[0], 1e-4f);
    // s = 2^ceil(log2(amax/448))
    float s = exp2f(ceilf(log2f(amax / 448.0f)));

    if (threadIdx.x < 32) {
        float y = g[threadIdx.x] / s;
        y = fminf(fmaxf(y, -448.0f), 448.0f);
        x[(size_t)group * 32 + threadIdx.x] = e4m3_to_f32(f32_to_e4m3_bits(y)) * s;
    }
}
