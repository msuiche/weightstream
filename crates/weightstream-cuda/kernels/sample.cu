// argmax over logits (greedy sampling, model-spec §10). Gumbel-max at
// temperature > 0 explicitly does not replicate torch's Philox stream
// (spec §12.10) — greedy is the gate-relevant path.

#include "common.cuh"

extern "C" __global__ void argmax_rows(
    const float* __restrict__ logits,  // [n]
    int* __restrict__ out_idx,         // [1]
    float* __restrict__ out_val,       // [1]
    int n)
{
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    float best = -1e30f;
    int best_i = -1;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = logits[i];
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    s_val[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = s_val[threadIdx.x + s];
            int oi = s_idx[threadIdx.x + s];
            if (ov > s_val[threadIdx.x] || (ov == s_val[threadIdx.x] && oi < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = ov;
                s_idx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_val[0] = s_val[0];
        out_idx[0] = s_idx[0];
    }
}
