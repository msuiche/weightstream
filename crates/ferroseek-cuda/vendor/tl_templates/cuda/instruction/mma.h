#pragma once

#include "../common.h"
namespace tl {

// Minimal SM80+ atoms used by the generated kernels. A is four packed
// registers, B is two, C/D are four FP32 accumulators.
template <DataType A, DataType B, DataType C, int M, int N, int K,
          bool row_a, bool row_b>
__device__ __forceinline__ void mma_sync(float* d, const unsigned* a,
                                         const unsigned* b) {
    if constexpr (A == DataType::kTensorFloat32 && B == DataType::kTensorFloat32 &&
                  C == DataType::kFloat32 && M == 16 && N == 8 && K == 8) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%10, %11, %12, %13};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
    } else {
        static_assert(A == DataType::kBFloat16 && B == DataType::kBFloat16 &&
                      C == DataType::kFloat32 && M == 16 && N == 8 && K == 16,
                      "only BF16 16x8x16 and TF32 16x8x8 are supported");
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%10, %11, %12, %13};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
    }
}

} // namespace tl
