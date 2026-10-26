// act_quant ue8m0 snap (model-spec §7): per 32-element group,
// amax floored at 1e-4, scale = 2^ceil(log2(amax/448)), clamp ±448,
// fp8 e4m3 quant then dequant back to f32 ("inplace" snap).
// The gate expects bit-exact match with the CPU reference (all ops are
// exact in fp32 except the fp8 round, which is deterministic).

#include <cstdint>

__device__ __forceinline__ float f32_to_e4m3(float v) {
    // Round-to-nearest-even to e4m3fn (bias 7, 3 mantissa bits, max 448).
    uint32_t bits = __float_as_uint(v);
    uint32_t sign = bits & 0x80000000u;
    float a = fabsf(v);
    if (a > 448.0f) a = 448.0f;
    int exp;
    float m = frexpf(a, &exp); // a = m * 2^exp, m in [0.5, 1)
    // e4m3 exponent field = exp + 6 (bias 7, and m in [0.5,1) -> 1.m with exp-1)
    int e = exp - 1 + 7;
    float val;
    if (e <= 0) {
        // subnormal: value = mant/8 * 2^-6
        val = rintf(a * 512.0f) / 512.0f;
    } else {
        int mant = (int)rintf((m * 2.0f - 1.0f) * 8.0f);
        if (mant == 8) { mant = 0; e += 1; }
        if (e >= 15) {
            val = 448.0f; // saturate (finite-only variant has no inf)
        } else {
            val = ldexpf(1.0f + (float)mant / 8.0f, e - 7);
        }
    }
    return sign ? -val : val;
}

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
        x[(size_t)group * 32 + threadIdx.x] = f32_to_e4m3(y) * s;
    }
}
