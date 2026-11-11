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

// One block per expert: the dot is split across 256 threads (the old
// one-block variant did 384 serial 5120-length dots — 1.1 ms per call,
// 20% of all GPU time in the nsys profile).
extern "C" __global__ void moe_gate(
    const float* __restrict__ x,          // [dim]
    const uint16_t* __restrict__ gate_w,  // [n_experts, dim] bf16
    const float* __restrict__ bias,       // [n_experts]
    float* __restrict__ scores,           // [n_experts] (scratch, reused as input by moe_select)
    int dim, int n_experts)
{
    int e = blockIdx.x;
    if (e >= n_experts) return;
    const uint16_t* w = gate_w + (size_t)e * dim;
    __shared__ float red[256];
    float part = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        part += x[d] * bf16_to_f32(w[d]);
    }
    red[threadIdx.x] = part;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float dot = red[0];
        float sp = (dot > 20.0f) ? dot : logf(1.0f + expf(dot));
        scores[e] = sqrtf(sp);
    }
}

// Thread 0 over the gate's scores: bias-steered top-k, unbiased weights,
// 1e-20 norm, route_scale — identical arithmetic to the old inline loop.
extern "C" __global__ void moe_select(
    float* __restrict__ scores,           // [n_experts] scratch: taken entries neutralized
    const float* __restrict__ bias,       // [n_experts]
    float* __restrict__ out_weights,      // [top_k]
    int* __restrict__ out_indices,        // [top_k]
    int n_experts, int top_k, float route_scale)
{
    // 32 threads scan-stride the experts; warp-reduce max with the serial
    // loop's tie rule (strictly greater wins -> lowest index on equal
    // values), lane 0 emits. Deterministic, output-identical to the
    // one-thread form (65 us, 2% of a step). A taken expert is
    // neutralized in the scores buffer (-1e31 sits below every real
    // score+bias); moe_gate rewrites the whole buffer next layer.
    const int lane = threadIdx.x;
    float wsum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        float bv = -1e30f;
        int be = -1;
        for (int e = lane; e < n_experts; e += 32) {
            float v = scores[e] + bias[e];
            if (v > bv) {
                bv = v;
                be = e;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
            int oe = __shfl_down_sync(0xFFFFFFFFu, be, off);
            if (ov > bv) {
                bv = ov;
                be = oe;
            }
        }
        if (lane == 0) {
            out_indices[k] = be;
            float w = scores[be];
            out_weights[k] = w;
            wsum += w;
            scores[be] = -1e31f; // taken
        }
        __syncwarp();
    }
    if (lane == 0) {
        for (int k = 0; k < top_k; k++) {
            out_weights[k] = out_weights[k] / (wsum + 1e-20f) * route_scale;
        }
    }
}

// Kept for the synthetic gate (TP1 path, one token): one block, the
// original fused form.
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
        float sp = (dot > 20.0f) ? dot : logf(1.0f + expf(dot));
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
    // silu exactly as the CPU reference computes it: g * (1 / (1 + exp(-g)))
    float silu = g * (1.0f / (1.0f + expf(-g)));
    out[i] = silu * u * routing_weight;
}

// moe_gate_m: M-batch gate scoring — grid (n_experts, m). Identical
// per-block arithmetic to moe_gate (bit-exact per row).
extern "C" __global__ void moe_gate_m(
    const float* __restrict__ x,          // [m, dim]
    const uint16_t* __restrict__ gate_w,  // [n_experts, dim] bf16
    const float* __restrict__ bias,       // [n_experts]
    float* __restrict__ scores,           // [m, n_experts]
    int dim, int n_experts)
{
    int e = blockIdx.x;
    int pos = blockIdx.y;
    if (e >= n_experts) return;
    const float* xr = x + (size_t)pos * dim;
    const uint16_t* w = gate_w + (size_t)e * dim;
    __shared__ float red[256];
    float part = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        part += xr[d] * bf16_to_f32(w[d]);
    }
    red[threadIdx.x] = part;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float dot = red[0];
        float sp = (dot > 20.0f) ? dot : logf(1.0f + expf(dot));
        scores[(size_t)pos * n_experts + e] = sqrtf(sp);
    }
}
