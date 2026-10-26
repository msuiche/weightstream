// Fused grouped-expert MoE (model-spec §6, TP4 note): the gate's top-6 is
// consumed on the DEVICE — no per-layer D2H readback, no host expert loop
// (the readback + loop cost ~107 ms of a 300 ms token: 40 pageable-copy
// syncs + 18 sequential launches per layer).
//
// moe_local_slots: gate_i_out[6] (global expert ids) -> the local ones,
// sorted ascending by global id (the reference accumulation order,
// model.py:895), as slot indices into the gate outputs.
//
// expert_glu: for one (slot, n): h = silu(clamp(w1x,10)) * clamp(w3x,-10,10)
// * routing_weight(slot) — the exact fp4_expert_gemm + swiglu_limit10
// arithmetic, one block per output element like the unfused path (same
// block_reduce shape -> bit-identical dots).
//
// expert_quant: act_quant_fp8_emit's math over one slot's 32-group.
//
// expert_reduce: for one output dim: y = sum over local slots ASCENDING of
// w2_slot · act_h_slot — the same per-expert block_reduce and the same
// add order as the host loop's acc_f32 chain -> bit-identical partial.

#include "common.cuh"

extern "C" __global__ void moe_local_slots(
    const int* __restrict__ gate_idx,     // [top_k] global expert ids
    int* __restrict__ local_slots,        // [top_k] slot ids, ascending expert id
    int* __restrict__ local_experts,      // [top_k] local rows (global - e_start)
    int* __restrict__ local_count,        // [1]
    int top_k, int e_start, int e_len)
{
    if (threadIdx.x != 0) return;
    __shared__ int ids[16];
    __shared__ int loc[16];
    __shared__ float key[16];
    int n = 0;
    for (int s = 0; s < top_k && s < 16; s++) {
        int e = gate_idx[s];
        if (e >= e_start && e < e_start + e_len) {
            ids[n] = s;           // slot index into the gate outputs
            loc[n] = e - e_start; // local expert row in the slabs
            key[n] = (float)e;    // sort key: global expert id
            n++;
        }
    }
    // insertion sort ascending (top_k <= 6)
    for (int i = 1; i < n; i++) {
        int vi = ids[i]; int vl = loc[i]; float ki = key[i];
        int j = i - 1;
        while (j >= 0 && key[j] > ki) {
            ids[j + 1] = ids[j]; loc[j + 1] = loc[j]; key[j + 1] = key[j]; j--;
        }
        ids[j + 1] = vi; loc[j + 1] = vl; key[j + 1] = ki;
    }
    for (int i = 0; i < n; i++) {
        local_slots[i] = ids[i];
        local_experts[i] = loc[i];
    }
    *local_count = n;
}

// One block per (slot, n): both dots with the fp4_expert_gemm inner loop.
extern "C" __global__ void expert_glu(
    const uint8_t* __restrict__ act_x,    // [K] e4m3
    const uint8_t* __restrict__ act_x_s, // [K/32] ue8m0
    const uint8_t* __restrict__ w1,      // packed fp4 [N, K/2]
    const uint8_t* __restrict__ w1_s,
    const uint8_t* __restrict__ w3,
    const uint8_t* __restrict__ w3_s,
    const float* __restrict__ route_w,   // [top_k] routing weights (scaled)
    const int* __restrict__ local_slots, // [top_k]
    const int* __restrict__ local_experts, // [top_k] local expert rows
    const int* __restrict__ local_count, // [1]
    float* __restrict__ h_slots,         // [top_k][N]
    int N, int K, int top_k)
{
    int slot = blockIdx.y;
    if (slot >= *local_count) return;
    int e_loc = local_experts[slot];
    slot = local_slots[slot];
    int n = blockIdx.x;
    if (n >= N) return;
    int kb = K / 32;
    // one expert's matrix inside the contiguous slab
    const uint8_t* w1e = w1 + (size_t)e_loc * N * (K / 2);
    const uint8_t* w3e = w3 + (size_t)e_loc * N * (K / 2);
    const uint8_t* w1se = w1_s + (size_t)e_loc * N * kb;
    const uint8_t* w3se = w3_s + (size_t)e_loc * N * kb;
    float acc1 = 0.0f, acc3 = 0.0f;
    for (int b = threadIdx.x; b < kb; b += blockDim.x) {
        const uint8_t* a = act_x + b * 32;
        const uint8_t* b1 = w1e + (size_t)n * (K / 2) + b * 16;
        const uint8_t* b3 = w3e + (size_t)n * (K / 2) + b * 16;
        float d1 = 0.0f, d3 = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; k++) {
            uint8_t byte1 = b1[k], byte3 = b3[k];
            d1 += e4m3_to_f32(a[2 * k]) * FP4_TABLE[byte1 & 0x0F];
            d1 += e4m3_to_f32(a[2 * k + 1]) * FP4_TABLE[byte1 >> 4];
            d3 += e4m3_to_f32(a[2 * k]) * FP4_TABLE[byte3 & 0x0F];
            d3 += e4m3_to_f32(a[2 * k + 1]) * FP4_TABLE[byte3 >> 4];
        }
        float sa = e8m0_to_f32(act_x_s[b]);
        acc1 += d1 * sa * e8m0_to_f32(w1se[(size_t)n * kb + b]);
        acc3 += d3 * sa * e8m0_to_f32(w3se[(size_t)n * kb + b]);
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc1;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float g = red[0];
    red[threadIdx.x] = acc3;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float gg = fminf(g, 10.0f);
        float u = fminf(fmaxf(red[0], -10.0f), 10.0f);
        float silu = gg * (1.0f / (1.0f + expf(-gg)));
        h_slots[(size_t)slot * N + n] = silu * u * route_w[slot];
    }
}

// act_quant_fp8_emit over one slot's [N] h values, grid (slot, N/32).
extern "C" __global__ void expert_quant(
    const float* __restrict__ h_slots,     // [top_k][N]
    const int* __restrict__ local_slots,
    const int* __restrict__ local_count,
    uint8_t* __restrict__ q_slots,         // [top_k][N]
    uint8_t* __restrict__ s_slots,         // [top_k][N/32]
    int N, int top_k)
{
    int slot = blockIdx.y;
    if (slot >= *local_count) return;
    slot = local_slots[slot];
    int group = blockIdx.x;
    const float* g = h_slots + (size_t)slot * N + (size_t)group * 32;

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
    int bits = __float_as_int(amax / 448.0f);
    int e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
    if (threadIdx.x == 0) s_slots[(size_t)slot * (N / 32) + group] = (uint8_t)(e + 127);
    float scale = exp2f((float)e);
    if (threadIdx.x < 32) {
        float y = g[threadIdx.x] / scale;
        y = fminf(fmaxf(y, -448.0f), 448.0f);
        q_slots[(size_t)slot * N + (size_t)group * 32 + threadIdx.x] = f32_to_e4m3_bits(y);
    }
}

// y[n] = sum over local slots ASCENDING of w2_e · act_h_e — the host loop's
// fp4_gemm + acc_f32 chain, order and reduction shape identical.
extern "C" __global__ void expert_reduce(
    const uint8_t* __restrict__ act_slots,  // [top_k][inter]
    const uint8_t* __restrict__ act_s_slots, // [top_k][inter/32]
    const uint8_t* __restrict__ w2,          // packed fp4 [dim, inter/2] (one expert)
    const uint8_t* __restrict__ w2_s,       // [dim, inter/32]
    const int* __restrict__ gate_idx,        // [top_k] global expert ids
    const int* __restrict__ local_slots,
    const int* __restrict__ local_count,
    float* __restrict__ y,                   // [dim]
    int dim, int inter, int top_k, int e_start)
{
    int n = blockIdx.x;
    if (n >= dim) return;
    int kb = inter / 32;
    __shared__ int w2_of[16];   // per local slot: local expert row (0..e_len)
    __shared__ int slots[16];
    if (threadIdx.x == 0) {
        int cnt = *local_count;
        for (int i = 0; i < cnt && i < 16; i++) {
            int s = local_slots[i];
            slots[i] = s;
            w2_of[i] = gate_idx[s] - e_start; // global id -> local table row
        }
    }
    __syncthreads();
    int cnt = *local_count;
    float acc = 0.0f;
    __shared__ float red[256];
    for (int i = 0; i < cnt; i++) {
        int slot = slots[i];
        int e_loc = w2_of[i];
        const uint8_t* b = w2 + (size_t)e_loc * ((size_t)dim * (inter / 2));
        const uint8_t* bs = w2_s + (size_t)e_loc * ((size_t)dim * kb);
        const uint8_t* a = act_slots + (size_t)slot * inter;
        const uint8_t* as = act_s_slots + (size_t)slot * kb;

        float part = 0.0f;
        for (int blk = threadIdx.x; blk < kb; blk += blockDim.x) {
            const uint8_t* bp = b + (size_t)n * (inter / 2) + blk * 16;
            const uint8_t* ap = a + blk * 32;
            float dot = 0.0f;
#pragma unroll
            for (int k = 0; k < 16; k++) {
                uint8_t byte = bp[k];
                dot += e4m3_to_f32(ap[2 * k]) * FP4_TABLE[byte & 0x0F];
                dot += e4m3_to_f32(ap[2 * k + 1]) * FP4_TABLE[byte >> 4];
            }
            part += dot * e8m0_to_f32(as[blk]) * e8m0_to_f32(bs[(size_t)n * kb + blk]);
        }
        red[threadIdx.x] = part;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        acc += red[0];  // ascending expert order, one rounding per add
        __syncthreads();
    }
    if (threadIdx.x == 0) y[n] = acc;
}
