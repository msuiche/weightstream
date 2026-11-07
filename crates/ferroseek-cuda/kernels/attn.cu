// sparse_attn (model-spec §3, kernel.py:311-403): fp32 online softmax over
// gathered KV rows, one block per head.
//   logit_i = (q_h · kv[idx_i]) * scale     idx_i < 0 → skip (logit -inf)
//   online max/sum/rescale per row, then sum_exp += exp(attn_sink[h] - max)
//   out = acc / sum_exp                      (value row of the sink is zero)
// kv rows are 512-wide f32 latents (window + compressed, concatenated).

#include "common.cuh"
#include <cuda_bf16.h>

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
    float qv[4];
    {
        int k = 0;
        for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) qv[k] = qh[d];
    }

    for (int i = 0; i < n_sel; i++) {
        int idx = idxs[i];
        // dot(q_h, kv[idx])
        float partial = 0.0f;
        float v[4];
        if (idx >= 0) {
            const float* row = kv + (size_t)idx * dim;
            int k = 0;
            for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
                v[k] = row[d];
                partial += qv[k] * v[k];
            }
        }
        if (threadIdx.x == 0) s_dot = 0.0f;
        __syncthreads();
        block_reduce_sum(partial, &s_dot);
        __syncthreads();

        // invalid indices are -inf, matching kernel.py's acc_s convention:
        // they contribute exactly zero to the softmax. (-1e30 would
        // contribute exp(0)=1 each while the running max is still at its
        // floor, inflating the denominator at early decode positions.)
        float logit = (idx >= 0) ? s_dot * scale : __int_as_float(0xff800000);
        float mnew = fmaxf(m, logit);
        float corr = expf(m - mnew);
        float w = expf(logit - mnew);
        s = __fmaf_rn(s, corr, w);
        if (idx >= 0) {
            int k = 0;
            for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
                acc[k] = __fmaf_rn(acc[k], corr, w * v[k]);
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

// sparse_attn_m: M-batch variant — grid (n_heads, m). Each block runs the
// exact sparse_attn arithmetic for one (head, position) pair: same dot
// order, same online softmax, so outputs are bit-identical to m sequential
// sparse_attn calls. idxs is [m, n_sel] with each position's own (causal)
// selection list into the shared gathered kv table.
extern "C" __global__ void sparse_attn_m(
    const float* __restrict__ q,        // [m, n_heads, dim]
    const float* __restrict__ kv,       // [n_rows, dim]
    const int*   __restrict__ idxs,     // [m, n_sel]
    const float* __restrict__ sink,     // [n_heads]
    float* __restrict__ out,            // [m, n_heads, dim]
    int n_heads, int dim, int n_sel, float scale)
{
    int h = blockIdx.x;
    int pos = blockIdx.y;
    if (h >= n_heads) return;
    const float* qh = q + ((size_t)pos * n_heads + h) * dim;
    const int* pidx = idxs + (size_t)pos * n_sel;
    float* oh = out + ((size_t)pos * n_heads + h) * dim;

    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_dot;

    float m = -1e30f;
    float s = 0.0f;
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = 0.0f;
    float qv[4];
    {
        int k = 0;
        for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) qv[k] = qh[d];
    }

    for (int i = 0; i < n_sel; i++) {
        int idx = pidx[i];
        float partial = 0.0f;
        float v[4];
        if (idx >= 0) {
            const float* row = kv + (size_t)idx * dim;
            int k = 0;
            for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
                v[k] = row[d];
                partial += qv[k] * v[k];
            }
        }
        if (threadIdx.x == 0) s_dot = 0.0f;
        __syncthreads();
        block_reduce_sum(partial, &s_dot);
        __syncthreads();

        float logit = (idx >= 0) ? s_dot * scale : __int_as_float(0xff800000);
        float mnew = fmaxf(m, logit);
        float corr = expf(m - mnew);
        float w = expf(logit - mnew);
        s = __fmaf_rn(s, corr, w);
        if (idx >= 0) {
            int k = 0;
            for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
                acc[k] = __fmaf_rn(acc[k], corr, w * v[k]);
            }
        } else {
            int k = 0;
            for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
                acc[k] *= corr;
            }
        }
        m = mnew;
    }

    s += expf(sink[h] - m);
    int k = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
        oh[d] = acc[k] / s;
    }
}

// dsv41f split-KV decode: 128-entry segments compute independent online
// states, then merge folds them in the original order. Partial output is
// unnormalized; stats are (m, s) per (segment, head).
extern "C" __global__ void sparse_attn_split(
    const float* __restrict__ q,
    const float* __restrict__ kv,
    const int* __restrict__ idxs,
    float* __restrict__ partial,
    float* __restrict__ stats,
    int n_heads, int dim, int n_sel, float scale)
{
    int h = blockIdx.x;
    int seg = blockIdx.y;
    if (h >= n_heads) return;
    int first = seg * 128;
    int last = min(first + 128, n_sel);
    const float* qh = q + (size_t)h * dim;
    float* oh = partial + ((size_t)seg * n_heads + h) * dim;
    __shared__ float s_dot;
    float m = -1e30f;
    float s = 0.0f;
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = 0.0f;
    float qv[4];
    {
        int k = 0;
        for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) qv[k] = qh[d];
    }
    for (int i = first; i < last; i++) {
        int idx = idxs[i];
        float partial_dot = 0.0f;
        float v[4];
        if (idx >= 0) {
            const float* row = kv + (size_t)idx * dim;
            int k = 0;
            for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
                v[k] = row[d];
                partial_dot += qv[k] * v[k];
            }
        }
        if (threadIdx.x == 0) s_dot = 0.0f;
        __syncthreads();
        block_reduce_sum(partial_dot, &s_dot);
        __syncthreads();
        float logit = (idx >= 0) ? s_dot * scale : __int_as_float(0xff800000);
        float mnew = fmaxf(m, logit);
        float corr = expf(m - mnew);
        float w = expf(logit - mnew);
        s = __fmaf_rn(s, corr, w);
        if (idx >= 0) {
            int k = 0;
            for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
                acc[k] = __fmaf_rn(acc[k], corr, w * v[k]);
            }
        } else {
            int k = 0;
            for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) acc[k] *= corr;
        }
        m = mnew;
    }
    int k = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) oh[d] = acc[k];
    if (threadIdx.x == 0) {
        stats[((size_t)seg * n_heads + h) * 2] = m;
        stats[((size_t)seg * n_heads + h) * 2 + 1] = s;
    }
}

extern "C" __global__ void sparse_attn_merge(
    const float* __restrict__ partial,
    const float* __restrict__ stats,
    const float* __restrict__ sink,
    float* __restrict__ out,
    int n_heads, int dim, int segments)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    float m = -1e30f;
    float s = 0.0f;
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = 0.0f;
    for (int seg = 0; seg < segments; seg++) {
        float sm = stats[((size_t)seg * n_heads + h) * 2];
        float ss = stats[((size_t)seg * n_heads + h) * 2 + 1];
        float mnew = fmaxf(m, sm);
        float corr = expf(m - mnew);
        float corr_seg = expf(sm - mnew);
        const float* part = partial + ((size_t)seg * n_heads + h) * dim;
        int k = 0;
        for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) {
            acc[k] = __fmaf_rn(acc[k], corr, corr_seg * part[d]);
        }
        s = __fmaf_rn(s, corr, ss * corr_seg);
        m = mnew;
    }
    s += expf(sink[h] - m);
    float* oh = out + (size_t)h * dim;
    int k = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x, k++) oh[d] = acc[k] / s;
}

extern "C" __global__ void f32_to_bf16(
    const float* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

extern "C" __global__ void copy_u8(
    const uint8_t* __restrict__ x,
    uint8_t* __restrict__ y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
