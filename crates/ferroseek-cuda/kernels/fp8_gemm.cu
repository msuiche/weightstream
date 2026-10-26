// FP8 block-scaled GEMM (model-spec §7): C[M,N] = A[M,K] @ B[N,K]^T
// A: fp8 e4m3, scales ue8m0 [M, K/32] (activation, block 1x32)
// B: fp8 e4m3, scales ue8m0 [N/32, K/32] (weight, block 32x32)
// Per K-block of 32: fp8xfp8 dot into fp32, scale-correct, accumulate fp32.
// Output f32 (callers cast as needed).

#include "common.cuh"

extern "C" __global__ void fp8_gemm(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ sA,
    const uint8_t* __restrict__ B,
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
        const uint8_t* bp = B + (size_t)n * K + b * 32;
        float dot = 0.0f;
#pragma unroll
        for (int k = 0; k < 32; k++) {
            dot += e4m3_to_f32(a[k]) * e4m3_to_f32(bp[k]);
        }
        float sa = e8m0_to_f32(sA[(size_t)m * kb + b]);
        float sb = e8m0_to_f32(sB[(size_t)(n / 32) * kb + b]);
        acc += dot * sa * sb;
    }
    block_reduce_sum(acc, &C[(size_t)m * N + n]);
}
