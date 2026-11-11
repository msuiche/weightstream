// Tiled M-batch FP4 expert GEMM (model-spec §7, batching path):
// C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T — the batched sibling of
// fp8_gemm_tile for the routed experts (packed fp4 weights).
//
// One 16x32 output tile per block, 512 threads, each thread owning one
// (m, n) output accumulated over whole K-blocks in K order — the same
// reduction-order class as fp8_gemm_tile (documented divergence from
// the M=1 fp4_expert_gemm's block_reduce tree; the engine gate
// re-validates when the batched path lands).
//
// A: fp8 e4m3, scales ue8m0 [M, K/32] (activation, block 1x32)
// B: fp4 e2m1 packed [N, K/2] (low nibble = even K), scales ue8m0 [N, K/32]
// Per K-block of 32: fp4 -> f32 (lossless via the table), fp8xfp8-equivalent
// dot in fp32, scale-correct, fp32 accumulate.

#include "common.cuh"

#define T4_M 16
#define T4_N 32

extern "C" __global__ void fp4_gemm_tile(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ sA,
    const uint8_t* __restrict__ B,   // packed, K/2 bytes per row
    const uint8_t* __restrict__ sB,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int kb = K / 32;
    const int m = blockIdx.y * T4_M + (threadIdx.x >> 5);
    const int n = blockIdx.x * T4_N + (threadIdx.x & 31);
    if (m >= M || n >= N) return;

    const uint8_t* a_row = A + (size_t)m * K;
    const uint8_t* b_row = B + (size_t)n * (K / 2);
    const uint8_t* sa_row = sA + (size_t)m * kb;
    const uint8_t* sb_row = sB + (size_t)n * kb;

    const bool vec = ((((uintptr_t)A) | ((uintptr_t)B)) & 15) == 0;

    float acc = 0.0f;
    for (int b = 0; b < kb; b++) {
        const uint8_t* a = a_row + b * 32;
        const uint8_t* bp = b_row + b * 16;
        float dot = 0.0f;
        if (vec) {
            const uint4 act0 = *reinterpret_cast<const uint4*>(a);
            const uint4 act1 = *reinterpret_cast<const uint4*>(a + 16);
            const uint4 pack = *reinterpret_cast<const uint4*>(bp);
            const uint8_t* a0 = reinterpret_cast<const uint8_t*>(&act0);
            const uint8_t* a1 = reinterpret_cast<const uint8_t*>(&act1);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&pack);
#pragma unroll
            for (int k = 0; k < 8; k++) {
                dot += e4m3_to_f32(a0[2 * k]) * FP4_TABLE[bytes[k] & 0x0F];
                dot += e4m3_to_f32(a0[2 * k + 1]) * FP4_TABLE[bytes[k] >> 4];
            }
#pragma unroll
            for (int k = 8; k < 16; k++) {
                dot += e4m3_to_f32(a1[2 * k - 16]) * FP4_TABLE[bytes[k] & 0x0F];
                dot += e4m3_to_f32(a1[2 * k - 15]) * FP4_TABLE[bytes[k] >> 4];
            }
        } else {
#pragma unroll
            for (int k = 0; k < 16; k++) {
                uint8_t byte = bp[k];
                dot += e4m3_to_f32(a[2 * k]) * FP4_TABLE[byte & 0x0F];
                dot += e4m3_to_f32(a[2 * k + 1]) * FP4_TABLE[byte >> 4];
            }
        }
        float sa = e8m0_to_f32(sa_row[b]);
        float sb = e8m0_to_f32(sb_row[b]);
        acc += dot * sa * sb;
    }
    C[(size_t)m * N + n] = acc;
}
