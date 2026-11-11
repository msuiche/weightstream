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
    // ceil(log2(amax/448)) by exponent/mantissa bits — bit-identical to
    // the CPU reference's log2_ceil (log2f+ceilf can round across an
    // integer boundary and pick a different scale; act_quant_fp8_emit
    // already used this form — the snap kernel was left behind, which
    // flipped a boundary amax at step 1 and diverged the engines).
    int bits = __float_as_int(amax / 448.0f);
    int e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
    float s = exp2f((float)e);

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
    uint8_t* __restrict__ s,       // [n/32] ue8m0 scale bytes + optional packed tail
    int n,                         // multiple of 32
    int packed_offset)             // 0 = raw only, else packed tail byte offset
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
    if (threadIdx.x == 0) {
        const uint8_t scale_byte = (uint8_t)(e + 127);
        s[group] = scale_byte;
        if (packed_offset > 0) s[packed_offset + group] = scale_byte;
    }
    float scale = exp2f((float)e);
    if (threadIdx.x < 32) {
        float y = g[threadIdx.x] / scale;
        y = fminf(fmaxf(y, -448.0f), 448.0f);
        q[(size_t)group * 32 + threadIdx.x] = f32_to_e4m3_bits(y);
    }
}

// Expand the emit format's UE8M0 bytes into TileLang's packed words:
// four consecutive K32 exponents per uint32, little-endian byte order.
extern "C" __global__ void pack_scale_words(
    const uint8_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n / 128) return;
    dst[i] = (uint32_t)src[i * 4]
        | ((uint32_t)src[i * 4 + 1] << 8)
        | ((uint32_t)src[i * 4 + 2] << 16)
        | ((uint32_t)src[i * 4 + 3] << 24);
}
