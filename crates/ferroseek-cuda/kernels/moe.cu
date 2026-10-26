// MoE helpers (model-spec §6):
// - moe_route: sqrtsoftplus gate scoring, bias-steered top-k, unbiased
//   weights, 1e-20 norm, route_scale
// - swiglu_limit10: silu(clamp(gate, max=10)) * clamp(up, -10, 10),
//   optional routing-weight multiply

#include "common.cuh"
#include <cuda_bf16.h>

__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    return __uint_as_float(bits);
}

// One block: n_experts threads score, thread 0 selects + normalizes.
extern "C" __global__ void moe_route(
    const float* __restrict__ x,          // [dim]
    const uint16_t* __restrict__ gate_w,  // [n_experts, dim] bf16
    const float* __restrict__ bias,       // [n_experts]
    float* __restrict__ out_weights,      // [top_k]
    int* __restrict__ out_indices,        // [top_k]
    int dim, int n_experts, int top_k, float route_scale)
{
    extern __shared__ float s_scores[];
    // scores[e] = sqrt(softplus(dot(x, gate_w[e])))
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        const uint16_t* w = gate_w + (size_t)e * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; d++) dot += x[d] * bf16_to_f32(w[d]);
        float sp = (dot > 20.0f) ? dot : log1pf(expf(dot));
        s_scores[e] = sqrtf(sp);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        bool taken[1024];
        for (int i = 0; i < n_experts && i < 1024; i++) taken[i] = false;
        float wsum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            int best = -1;
            float best_v = -1e30f;
            for (int e = 0; e < n_experts; e++) {
                if (taken[e]) continue;
                float v = s_scores[e] + bias[e];
                if (v > best_v) { best_v = v; best = e; }
            }
            taken[best] = true;
            out_indices[k] = best;
            float w = s_scores[best]; // unbiased weight (bias steers only)
            out_weights[k] = w;
            wsum += w;
        }
        for (int k = 0; k < top_k; k++) {
            out_weights[k] = out_weights[k] / (wsum + 1e-20f) * route_scale;
        }
    }
}

extern "C" __global__ void swiglu_limit10(
    const float* __restrict__ gate,   // [n]
    const float* __restrict__ up,     // [n]
    float* __restrict__ out,          // [n]
    int n, float routing_weight)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = fminf(gate[i], 10.0f);
    float u = fminf(fmaxf(up[i], -10.0f), 10.0f);
    float silu = g / (1.0f + expf(-g));
    out[i] = silu * u * routing_weight;
}
