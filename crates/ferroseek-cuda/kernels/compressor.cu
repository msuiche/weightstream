// CSA compressor finish (model-spec §5): on pair completion, the latent is
// the softmax-weighted sum of the two parked fp32 slots, then an RMS norm.
// The caller runs f32_gemv (wkv/wgate) into the state buffers per token.

#include "common.cuh"

// kv_state [2, dim], score_state [2, dim] -> latent [dim]
// (softmax over the 2 slot-scores per dim — note: per-dim softmax across
// slots, matching the reference's softmax(score_state, dim=slots))
extern "C" __global__ void compressor_finish(
    const float* __restrict__ kv_state,
    const float* __restrict__ score_state,
    float* __restrict__ latent,
    int dim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float s0 = score_state[d];
    float s1 = score_state[dim + d];
    float mx = fmaxf(s0, s1);
    float e0 = expf(s0 - mx);
    float e1 = expf(s1 - mx);
    float sum = e0 + e1;
    latent[d] = (kv_state[d] * e0 + kv_state[dim + d] * e1) / sum;
}

// bf16 snap (round-to-nearest-even): the reference casts the latent to
// bf16 before the RMS norm (model.py:480).
extern "C" __global__ void bf16_snap(
    float* __restrict__ x,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t bits = __float_as_uint(x[i]);
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    x[i] = __uint_as_float(bits & 0xFFFF0000);
}
