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

    // streaming-mode pointers sit inside registered safetensors shards,
    // which align tensors at 8 bytes — the uint4 path demands 16. Fall
    // back to the byte loop (the pre-vectorization order, bit-identical)
    // when either base is misaligned. Xid 13 root cause: misaligned 16B
    // reads faulted the SM on GB10 and corrupted the channel for every
    // later test in the process (memcheck: 622 invalid reads).
    const bool vec = ((((uintptr_t)A) | ((uintptr_t)B)) & 15) == 0;

    float acc = 0.0f;
    for (int b = threadIdx.x; b < kb; b += blockDim.x) {
        const uint8_t* a = A + (size_t)m * K + b * 32;
        const uint8_t* bp = B + (size_t)n * K + b * 32;
        float dot = 0.0f;
        if (vec) {
            // two 16B vectorized loads, same element order (identical adds)
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
        float sa = e8m0_to_f32(sA[(size_t)m * kb + b]);
        float sb = e8m0_to_f32(sB[(size_t)(n / 32) * kb + b]);
        acc += dot * sa * sb;
    }
    block_reduce_sum(acc, &C[(size_t)m * N + n]);
}
