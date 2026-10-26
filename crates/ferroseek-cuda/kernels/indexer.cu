// DSA indexer scoring (model-spec §4):
//   score[h,t] = relu( sum_d q[h,d] * index_k[t,d] )
//   index_score[t] = sum_h score[h,t] * weights[h]
// q [n_heads, head_dim] and index_k [n_rows, head_dim] are fp4-snapped
// values stored f32; weights [n_heads].

#include "common.cuh"

extern "C" __global__ void indexer_score(
    const float* __restrict__ q,        // [n_heads, head_dim]
    const float* __restrict__ index_k,  // [n_rows, head_dim]
    const float* __restrict__ weights,  // [n_heads]
    float* __restrict__ out,            // [n_rows]
    int n_heads, int head_dim, int n_rows)
{
    int t = blockIdx.x;
    if (t >= n_rows) return;
    const float* k = index_k + (size_t)t * head_dim;

    float acc = 0.0f;
    for (int h = threadIdx.x; h < n_heads; h += blockDim.x) {
        const float* qh = q + (size_t)h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += qh[d] * k[d];
        acc += fmaxf(dot, 0.0f) * weights[h];
    }
    block_reduce_sum(acc, &out[t]);
}
