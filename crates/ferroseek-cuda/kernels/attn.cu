// sparse_attn (model-spec §3, kernel.py:311-403): fp32 online softmax over
// gathered KV rows, one block per head.
//   logit_i = (q_h · kv[idx_i]) * scale     idx_i < 0 → skip (logit -inf)
//   online max/sum/rescale per row, then sum_exp += exp(attn_sink[h] - max)
//   out = acc / sum_exp                      (value row of the sink is zero)
// kv rows are 512-wide f32 latents (window + compressed, concatenated).

#include "common.cuh"

extern "C" __global__ void sparse_attn(
    const float* __restrict__ q,        // [n_heads, dim]
    const float* __restrict__ kv,       // [n_rows, dim]
    const int*   __restrict__ idxs,     // [n_sel]
    const float* __restrict__ sink,     // [n_heads]
    float* __restrict__ out,            // [n_heads, dim]
    int n_heads, int dim, int n_sel, float scale)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    const float* qh = q + (size_t)h * dim;
    float* oh = out + (size_t)h * dim;

    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_dot;

    float m = -1e30f;
    float s = 0.0f;
    // accumulator lives in registers: each thread owns a strided slice
    // of the 512-wide output (dim multiple of blockDim for our shapes)
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = 0.0f;

    for (int i = 0; i < n_sel; i++) {
        int idx = idxs[i];
        // dot(q_h, kv[idx])
        float partial = 0.0f;
        if (idx >= 0) {
            const float* row = kv + (size_t)idx * dim;
            for (int d = threadIdx.x; d < dim; d += blockDim.x) {
                partial += qh[d] * row[d];
            }
        }
        if (threadIdx.x == 0) s_dot = 0.0f;
        __syncthreads();
        block_reduce_sum(partial, &s_dot);
        __syncthreads();

        float logit = (idx >= 0) ? s_dot * scale : -1e30f;
        float mnew = fmaxf(m, logit);
        float corr = expf(m - mnew);
        float w = expf(logit - mnew);
        s = s * corr + w;
        if (idx >= 0) {
            const float* row = kv + (size_t)idx * dim;
            int k = 0;
            for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
                acc[k] = acc[k] * corr + w * row[d];
            }
        } else {
            int k = 0;
            for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
                acc[k] *= corr;
            }
        }
        m = mnew;
    }

    // attention sink: one extra key with zero value vector
    s += expf(sink[h] - m);
    int k = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
        oh[d] = acc[k] / s;
    }
}
