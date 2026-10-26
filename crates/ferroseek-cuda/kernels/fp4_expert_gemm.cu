// FP4 expert GEMM (model-spec §7): C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T
// A: fp8 e4m3, scales ue8m0 [M, K/32] (activation, block 1x32)
// B: fp4 e2m1 packed [N, K/2] (low nibble = even K), scales ue8m0 [N, K/32]
// Per K-block of 32: fp4 → f32 → fp8 (lossless: every e2m1 value is
// representable in e4m3), fp8×fp8 dot in fp32, scale-correct, fp32 acc.

#include "common.cuh"

extern "C" __global__ void fp4_expert_gemm(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ sA,
    const uint8_t* __restrict__ B,   // packed, K/2 bytes per row
    const uint8_t* __restrict__ sB,
    float* __restrict__ C,
    int M, int N, int K)
{
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (m >= M || n >= N) return;
    int kb = K / 32;

    float acc = 0.0f;
    for (int b = threadIdx.x; b < kb; b += blockDim.x) {
        const uint8_t* a = A + (size_t)m * K + b * 32;
        const uint8_t* bp = B + (size_t)n * (K / 2) + b * 16;
        float dot = 0.0f;
        // 16-byte vectorized loads, same element order (identical adds):
        // one uint4 covers the 16 packed-fp4 bytes (32 K values); the 32
        // fp8 activation bytes need two.
        const uint4 pack = *reinterpret_cast<const uint4*>(bp);
        const uint4 act0 = *reinterpret_cast<const uint4*>(a);
        const uint4 act1 = *reinterpret_cast<const uint4*>(a + 16);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&pack);
        const uint8_t* a0 = reinterpret_cast<const uint8_t*>(&act0);
        const uint8_t* a1 = reinterpret_cast<const uint8_t*>(&act1);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            uint8_t byte = bytes[k];
            dot += e4m3_to_f32(a0[2 * k]) * FP4_TABLE[byte & 0x0F];
            dot += e4m3_to_f32(a0[2 * k + 1]) * FP4_TABLE[byte >> 4];
        }
#pragma unroll
        for (int k = 8; k < 16; k++) {
            uint8_t byte = bytes[k];
            dot += e4m3_to_f32(a1[2 * k - 16]) * FP4_TABLE[byte & 0x0F];
            dot += e4m3_to_f32(a1[2 * k - 15]) * FP4_TABLE[byte >> 4];
        }
        float sa = e8m0_to_f32(sA[(size_t)m * kb + b]);
        float sb = e8m0_to_f32(sB[(size_t)n * kb + b]);
        acc += dot * sa * sb;
    }
    block_reduce_sum(acc, &C[(size_t)m * N + n]);
}
