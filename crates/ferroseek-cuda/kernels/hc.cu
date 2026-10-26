// Hyper-connection mixing + 20-iteration Sinkhorn (model-spec §2,
// model.py:950-968, kernel.py:406-474).
//
// One block per token. mixes = (xf @ hc_fn^T) * rstd over the flattened
// 4x-dim stream; then pre/post/comb with the exact eps placement:
//   pre[j]  = sigmoid(mixes[j]        * scale[0] + base[j]) + eps
//   post[j] = 2 * sigmoid(mixes[4+j]  * scale[1] + base[4+j])
//   comb_raw[j][k] = mixes[8 + j*4 + k] * scale[2] + base[8 + j*4 + k]
//   comb = softmax_rows(comb_raw) + eps; /(colsum+eps);
//   19 x { /(rowsum+eps); /(colsum+eps) }

#include "common.cuh"

extern "C" __global__ void hc_block(
    const float* __restrict__ xf,      // [n_tokens, hc_dim]
    const float* __restrict__ hc_fn,   // [mix_hc, hc_dim]
    const float* __restrict__ hc_base, // [mix_hc]
    const float* __restrict__ hc_scale,// [3]
    float* __restrict__ out_pre,       // [n_tokens, 4]
    float* __restrict__ out_post,      // [n_tokens, 4]
    float* __restrict__ out_comb,      // [n_tokens, 16]
    int n_tokens, int hc_dim, int mix_hc, float eps)
{
    int tok = blockIdx.x;
    if (tok >= n_tokens) return;
    const float* x = xf + (size_t)tok * hc_dim;

    __shared__ float s_sumsq;
    __shared__ float s_mix[24];
    if (mix_hc > 24) return; // spec constant: (2+4)*4

    // rstd over the whole flattened stream (one statistic per token)
    float acc = 0.0f;
    for (int i = threadIdx.x; i < hc_dim; i += blockDim.x) {
        float v = x[i];
        acc += v * v;
    }
    block_reduce_sum(acc, &s_sumsq);
    __syncthreads();
    float rstd = rsqrtf(s_sumsq / (float)hc_dim + 1e-20f);

    // mixes[row] = dot(x, hc_fn[row]) * rstd
    for (int row = threadIdx.x; row < mix_hc; row += blockDim.x) {
        const float* w = hc_fn + (size_t)row * hc_dim;
        float dot = 0.0f;
        for (int i = 0; i < hc_dim; i++) dot += x[i] * w[i];
        s_mix[row] = dot * rstd;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float pre[4], comb[16];
        for (int j = 0; j < 4; j++) {
            pre[j] = 1.0f / (1.0f + expf(-(s_mix[j] * hc_scale[0] + hc_base[j]))) + eps;
            out_post[(size_t)tok * 4 + j] =
                2.0f / (1.0f + expf(-(s_mix[4 + j] * hc_scale[1] + hc_base[4 + j])));
            out_pre[(size_t)tok * 4 + j] = pre[j];
            for (int k = 0; k < 4; k++) {
                comb[j * 4 + k] = s_mix[8 + j * 4 + k] * hc_scale[2] + hc_base[8 + j * 4 + k];
            }
        }
        // row softmax, THEN entrywise +eps (spec order)
        for (int j = 0; j < 4; j++) {
            float mx = fmaxf(fmaxf(comb[j*4], comb[j*4+1]), fmaxf(comb[j*4+2], comb[j*4+3]));
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) { comb[j*4+k] = expf(comb[j*4+k] - mx); sum += comb[j*4+k]; }
            for (int k = 0; k < 4; k++) comb[j*4+k] = comb[j*4+k] / sum + eps;
        }
        // column normalize
        for (int k = 0; k < 4; k++) {
            float cs = eps;
            for (int j = 0; j < 4; j++) cs += comb[j * 4 + k];
            for (int j = 0; j < 4; j++) comb[j * 4 + k] /= cs;
        }
        // 19 row/col pairs
        for (int it = 0; it < 19; it++) {
            for (int j = 0; j < 4; j++) {
                float rs = eps;
                for (int k = 0; k < 4; k++) rs += comb[j * 4 + k];
                for (int k = 0; k < 4; k++) comb[j * 4 + k] /= rs;
            }
            for (int k = 0; k < 4; k++) {
                float cs = eps;
                for (int j = 0; j < 4; j++) cs += comb[j * 4 + k];
                for (int j = 0; j < 4; j++) comb[j * 4 + k] /= cs;
            }
        }
        for (int k = 0; k < 16; k++) out_comb[(size_t)tok * 16 + k] = comb[k];
    }
}

// hc_pre: y[d] = sum_j pre[j] * x[j][d]   (collapse 4 copies -> one)
extern "C" __global__ void hc_pre(
    const float* __restrict__ x,    // [n_tokens, 4, dim]
    const float* __restrict__ pre,  // [n_tokens, 4]
    float* __restrict__ y,          // [n_tokens, dim]
    int n_tokens, int dim)
{
    int tok = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= n_tokens || d >= dim) return;
    const float* xt = x + (size_t)tok * 4 * dim;
    const float* p = pre + (size_t)tok * 4;
    float acc = 0.0f;
    for (int j = 0; j < 4; j++) acc += p[j] * xt[(size_t)j * dim + d];
    y[(size_t)tok * dim + d] = acc;
}

// hc_post: y[j][d] = post[j] * x[d] + sum_a C[a][j] * res[a][d]
extern "C" __global__ void hc_post(
    const float* __restrict__ x,     // [n_tokens, dim]
    const float* __restrict__ res,   // [n_tokens, 4, dim]
    const float* __restrict__ post,  // [n_tokens, 4]
    const float* __restrict__ comb,  // [n_tokens, 16]
    float* __restrict__ y,           // [n_tokens, 4, dim]
    int n_tokens, int dim)
{
    int tok = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= n_tokens || d >= dim) return;
    const float* xt = x + (size_t)tok * dim;
    const float* rt = res + (size_t)tok * 4 * dim;
    const float* p = post + (size_t)tok * 4;
    const float* c = comb + (size_t)tok * 16;
    for (int j = 0; j < 4; j++) {
        float acc = p[j] * xt[d];
        for (int a = 0; a < 4; a++) acc += c[a * 4 + j] * rt[(size_t)a * dim + d];
        y[(size_t)tok * 4 * dim + (size_t)j * dim + d] = acc;
    }
}
