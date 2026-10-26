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

// act_quant emitting quantized storage (the "non-inplace" half of spec §7):
// per 32-element group, f32 -> e4m3 bytes + ue8m0 scale bytes, exactly the
// activation format fp8_gemm / fp4_expert_gemm consume. Bit-consistent
// with act_quant_ue8m0_snap's scale rule; gated bit-exact against the CPU
// reference's act quantization.
extern "C" __global__ void act_quant_fp8_emit(
    const float* __restrict__ x,   // [n]
    uint8_t* __restrict__ q,       // [n] e4m3 bytes
    uint8_t* __restrict__ s,       // [n/32] ue8m0 scale bytes
    int n) // multiple of 32
{
    int group = blockIdx.x;
    const float* g = x + (size_t)group * 32;

    __shared__ float partial[32];
    float amax = 0.0f;
    if (threadIdx.x < 32) amax = fabsf(g[threadIdx.x]);
    partial[threadIdx.x] = amax;
    __syncthreads();
    for (int h = 16; h > 0; h >>= 1) {
        if (threadIdx.x < h) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + h]);
        __syncthreads();
    }
    amax = fmaxf(partial[0], 1e-4f);
    // ceil(log2(amax/448)) by exponent/mantissa bits — bit-identical to the
    // CPU reference's log2_ceil (log2f+ceilf can round across an integer
    // boundary and pick a different scale).
    int bits = __float_as_int(amax / 448.0f);
    int e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
    if (threadIdx.x == 0) s[group] = (uint8_t)(e + 127);
    float scale = exp2f((float)e);
    if (threadIdx.x < 32) {
        float y = g[threadIdx.x] / scale;
        y = fminf(fmaxf(y, -448.0f), 448.0f);
        q[(size_t)group * 32 + threadIdx.x] = f32_to_e4m3_bits(y);
    }
}
