// Engram (model-spec §8): fp8 row gather + gated merge into the hc stream.
//
// gather_rows_fp8: out[r] = dequant(table[idx[r]]) with per-32 ue8m0 scales
// engram_merge: per hc copy j: dot = sum_d h[j][d]*w[d]*key[j][d], where
//   w = q_weight * k_weight (precomputed product); normalized by the two
//   rstds and dim^-0.5; gate = sigmoid(copysign(sqrt(max(|dot|,1e-6)),dot));
//   h[j] += gate[j] * value

#include "common.cuh"

extern "C" __global__ void gather_rows_fp8(
    const uint8_t* __restrict__ table,   // [n_rows, cols]
    const uint8_t* __restrict__ scales,  // [n_rows, cols/32]
    const long*   __restrict__ idxs,     // [n_sel]
    float* __restrict__ out,             // [n_sel, cols]
    int n_sel, int cols)
{
    int r = blockIdx.x;
    if (r >= n_sel) return;
    long idx = idxs[r];
    const uint8_t* w = table + (size_t)idx * cols;
    const uint8_t* s = scales + (size_t)idx * (cols / 32);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        out[(size_t)r * cols + c] = e4m3_to_f32(w[c]) * e8m0_to_f32(s[c / 32]);
    }
}

extern "C" __global__ void engram_merge(
    float* __restrict__ h,          // [4, dim] hc stream (in/out)
    const float* __restrict__ key,  // [4, dim]
    const float* __restrict__ value,// [dim]
    const float* __restrict__ w,    // [dim] = q_weight * k_weight
    int dim, float inv_sqrt_dim)
{
    int j = blockIdx.x;
    if (j >= 4) return;
    float* hj = h + (size_t)j * dim;
    const float* kj = key + (size_t)j * dim;

    __shared__ float s_dot;
    __shared__ float s_hsq;
    __shared__ float s_ksq;
    if (threadIdx.x == 0) { s_dot = 0.0f; s_hsq = 0.0f; s_ksq = 0.0f; }
    __syncthreads();

    float dot = 0.0f, hsq = 0.0f, ksq = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        dot += hj[d] * w[d] * kj[d];
        hsq += hj[d] * hj[d];
        ksq += kj[d] * kj[d];
    }
    block_reduce_sum(dot, &s_dot);
    __syncthreads();
    block_reduce_sum(hsq, &s_hsq);
    __syncthreads();
    block_reduce_sum(ksq, &s_ksq);
    __syncthreads();

    float rstd_h = rsqrtf(s_hsq / (float)dim + 1e-20f);
    float rstd_k = rsqrtf(s_ksq / (float)dim + 1e-20f);
    float d01 = s_dot * rstd_h * rstd_k * inv_sqrt_dim;
    float gate = 1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(d01), 1e-6f)), d01)));

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        hj[d] += gate * value[d];
    }
}

// f32 row gather — the embedding lookup (spec §1): one token row out of
// the [vocab, dim] f32 table. Same shape as gather_rows_fp8 minus scales.
extern "C" __global__ void gather_rows_f32(
    const float* __restrict__ table,  // [n_rows_total, cols]
    const long*   __restrict__ idxs,   // [n_sel]
    float* __restrict__ out,           // [n_sel, cols]
    int n_sel, int cols)
{
    int r = blockIdx.x;
    if (r >= n_sel) return;
    long idx = idxs[r];
    const float* src = table + (size_t)idx * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        out[(size_t)r * cols + c] = src[c];
    }
}
