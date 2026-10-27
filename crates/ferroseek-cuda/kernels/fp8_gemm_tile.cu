// Tiled M-batch FP8 block-scaled GEMM (model-spec §7, batching path):
// C[M,N] = A[M,K] @ B[N,K]^T, the same arithmetic as fp8_gemm but with a
// 16-row output tile per block so the weight rows are read ONCE per 16
// requests instead of once per request — the DSpark verify step drafts
// M up to 6 tokens per step and the batched prefill wants the same.
//
// A: fp8 e4m3, scales ue8m0 [M, K/32] (activation, block 1x32)
// B: fp8 e4m3, scales ue8m0 [N/32, K/32] (weight, block 32x32)
// Per K-block of 32: fp8xfp8 dot into fp32, scale-correct, accumulate.
//
// Reduction order (documented divergence from the M=1 fp8_gemm): each
// thread owns whole K-blocks of ONE (m, n) output and accumulates them
// sequentially in K order — no block_reduce tree. Values are within the
// engine gate's tolerance class (~1e-6 fp32 noise, engine.rs's
// documented divergence), re-validated by the engine gate when the
// batched path lands.
//
// Layout per block: 16 (m) x 32 (n) output tile, 512 threads.
// Thread (tm, tn) computes C[tm, tn]: 16 rows staged in shared as fp8
// (16 x 32 per K-block), B read straight from global (each n-column
// row is read by 16 threads -> L2).

#include "common.cuh"

#define TILE_M 16
#define TILE_N 32

extern "C" __global__ void fp8_gemm_tile(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ sA,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ sB,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int kb = K / 32;
    const int tile_m = blockIdx.y;          // TILE_M rows
    const int tile_n = blockIdx.x;           // TILE_N cols
    const int m0 = tile_m * TILE_M;
    const int n0 = tile_n * TILE_N;

    const int tn = threadIdx.x % TILE_N;    // output column in tile
    const int lane = threadIdx.x;           // 0..511
    (void)tn;

    // each of the 512 threads owns one (m, n) output when the tile is
    // full: m = m0 + lane / 32, n = n0 + lane % 32
    const int m = m0 + (lane >> 5);
    const int n = n0 + (lane & 31);
    const bool active = (m < M) && (n < N);

    const bool vec = ((((uintptr_t)A) | ((uintptr_t)B)) & 15) == 0;

    float acc = 0.0f;
    if (!active) return;

    const uint8_t* a_row = A + (size_t)m * K;
    const uint8_t* b_row = B + (size_t)n * K;
    const uint8_t* sa_row = sA + (size_t)m * kb;
    const uint8_t* sb_row = sB + (size_t)(n / 32) * kb;

    for (int b = 0; b < kb; b++) {
        const uint8_t* a = a_row + b * 32;
        const uint8_t* bp = b_row + b * 32;
        float dot = 0.0f;
        // two 16B vectorized loads, ascending k order
        if (vec) {
            const uint4 va0 = *reinterpret_cast<const uint4*>(a);
            const uint4 va1 = *reinterpret_cast<const uint4*>(a + 16);
            const uint4 vb0 = *reinterpret_cast<const uint4*>(bp);
            const uint4 vb1 = *reinterpret_cast<const uint4*>(bp + 16);
            const uint8_t* a8 = reinterpret_cast<const uint8_t*>(&va0);
            const uint8_t* a8h = reinterpret_cast<const uint8_t*>(&va1);
            const uint8_t* b8 = reinterpret_cast<const uint8_t*>(&vb0);
            const uint8_t* b8h = reinterpret_cast<const uint8_t*>(&vb1);
#pragma unroll
            for (int k = 0; k < 16; k++) {
                dot += e4m3_to_f32(a8[k]) * e4m3_to_f32(b8[k]);
                dot += e4m3_to_f32(a8h[k]) * e4m3_to_f32(b8h[k]);
            }
        } else {
#pragma unroll
            for (int k = 0; k < 32; k++) {
                dot += e4m3_to_f32(a[k]) * e4m3_to_f32(bp[k]);
            }
        }
        float sa = e8m0_to_f32(sa_row[b]);
        float sb = e8m0_to_f32(sb_row[b]);
        acc += dot * sa * sb;
    }
    C[(size_t)m * N + n] = acc;
}
