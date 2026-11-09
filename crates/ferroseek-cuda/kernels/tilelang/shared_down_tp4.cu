#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma_block_scale.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void gemm_mxfp8_kernel(const fp8_e4_t* __restrict__ A, const fp8_e4_t* __restrict__ B, float* __restrict__ C, const uint* __restrict__ SFA, const uint* __restrict__ SFB, int M, int Mp);
extern "C" __global__ void __launch_bounds__(64, 1) gemm_mxfp8_kernel(const fp8_e4_t* __restrict__ A, const fp8_e4_t* __restrict__ B, float* __restrict__ C, const uint* __restrict__ SFA, const uint* __restrict__ SFB, int M, int Mp) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 2048));
  void* SFA_shared = ((void*)((char*)buf_dyn_shmem + 18432));
  void* SFB_shared = ((void*)((char*)buf_dyn_shmem + 18560));
  float C_local[32];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_shared)[((((((int)threadIdx.x) >> 2) * 64) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A[(((((int64_t)((int)blockIdx.y)) * (int64_t)9216) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)576)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)16))])), ((((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 2)) < Mp) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 2)) < Mp)));
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)B_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B[((((((int)blockIdx.x) * 73728) + (i_1 * 9216)) + ((((int)threadIdx.x) >> 2) * 576)) + ((((int)threadIdx.x) & 3) * 16))])));
  }
  if (((int)threadIdx.x) < 16) {
    tl::cp_async_gs_conditional<4>((&(((uint*)SFA_shared)[((int)threadIdx.x)])), (&(SFA[(min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + ((int64_t)((int)threadIdx.x))), (((int64_t)Mp) - (int64_t)1)) * (int64_t)5)])), (1 <= Mp));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::cp_async_gs<4>((&(((uint*)SFB_shared)[((i_2 * 64) + ((int)threadIdx.x))])), (&(SFB[(((((int)blockIdx.x) * 640) + (i_2 * 320)) + (((int)threadIdx.x) * 5))])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 8; ++ko) {
    __syncthreads();
    tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_shared)[((((((ko + 1) & 1) * 1024) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A[(((((((int64_t)((int)blockIdx.y)) * (int64_t)9216) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)576)) + (((int64_t)ko) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)16)) + (int64_t)64)])), ((((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 2)) < Mp) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 2)) < Mp)));
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      tl::cp_async_gs<16>((&(((fp8_e4_t*)B_shared)[(((((((ko + 1) & 1) * 8192) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B[((((((((int)blockIdx.x) * 73728) + (i_3 * 9216)) + ((((int)threadIdx.x) >> 2) * 576)) + (ko * 64)) + ((((int)threadIdx.x) & 3) * 16)) + 64)])));
    }
    if (((int)threadIdx.x) < 16) {
      tl::cp_async_gs_conditional<4>((&(((uint*)SFA_shared)[((((ko + 1) & 1) * 16) + ((int)threadIdx.x))])), (&(SFA[((min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + ((int64_t)((int)threadIdx.x))), (((int64_t)Mp) - (int64_t)1)) * (int64_t)5) + ((((int64_t)ko) + (int64_t)1) >> (int64_t)1))])), (1 <= Mp));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      tl::cp_async_gs<4>((&(((uint*)SFB_shared)[(((((ko + 1) & 1) * 128) + (i_4 * 64)) + ((int)threadIdx.x))])), (&(SFB[((((((int)blockIdx.x) * 640) + (i_4 * 320)) + (((int)threadIdx.x) * 5)) + ((ko + 1) >> 1))])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      fp8_e4_t A_local[16];
      fp8_e4_t B_local[64];
      for (int ki = 0; ki < 2; ++ki) {
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_shared)[(((((ko & 1) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])), (&(A_local[0])));
        #pragma unroll
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)B_shared)[((((((((ko & 1) * 8192) + ((((int)threadIdx.x) >> 5) * 4096)) + (i_5 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])), (&(B_local[(i_5 * 16)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + (j * 8)), reinterpret_cast<const uint32_t*>(A_local + 0), reinterpret_cast<const uint32_t*>(B_local + (j * 16)), reinterpret_cast<const float*>(C_local + (j * 8)), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((ko & 1) * 16) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[(((((ko & 1) * 128) + ((((int)threadIdx.x) >> 5) * 64)) + (j * 16)) + ((((int)threadIdx.x) & 31) >> 2))])))), static_cast<uint16_t>((((ko & 1) * 2) + ki)), static_cast<uint16_t>(0), static_cast<uint16_t>((((ko & 1) * 2) + ki)), static_cast<uint16_t>(0));
          tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + ((j * 8) + 4)), reinterpret_cast<const uint32_t*>(A_local + 0), reinterpret_cast<const uint32_t*>(B_local + ((j * 16) + 8)), reinterpret_cast<const float*>(C_local + ((j * 8) + 4)), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((ko & 1) * 16) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[((((((ko & 1) * 128) + ((((int)threadIdx.x) >> 5) * 64)) + (j * 16)) + ((((int)threadIdx.x) & 31) >> 2)) + 8)])))), static_cast<uint16_t>((((ko & 1) * 2) + ki)), static_cast<uint16_t>(0), static_cast<uint16_t>((((ko & 1) * 2) + ki)), static_cast<uint16_t>(0));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    fp8_e4_t A_local_1[16];
    fp8_e4_t B_local_1[64];
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_shared)[((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])), (&(A_local_1[0])));
      #pragma unroll
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)B_shared)[(((((((((int)threadIdx.x) >> 5) * 4096) + (i_6 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16))])), (&(B_local_1[(i_6 * 16)])));
      }
      for (int j_1 = 0; j_1 < 4; ++j_1) {
        tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + (j_1 * 8)), reinterpret_cast<const uint32_t*>(A_local_1 + 0), reinterpret_cast<const uint32_t*>(B_local_1 + (j_1 * 16)), reinterpret_cast<const float*>(C_local + (j_1 * 8)), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[(((((int)threadIdx.x) & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2))])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[((((((int)threadIdx.x) >> 5) * 64) + (j_1 * 16)) + ((((int)threadIdx.x) & 31) >> 2))])))), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0));
        tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + ((j_1 * 8) + 4)), reinterpret_cast<const uint32_t*>(A_local_1 + 0), reinterpret_cast<const uint32_t*>(B_local_1 + ((j_1 * 16) + 8)), reinterpret_cast<const float*>(C_local + ((j_1 * 8) + 4)), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[(((((int)threadIdx.x) & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2))])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[(((((((int)threadIdx.x) >> 5) * 64) + (j_1 * 16)) + ((((int)threadIdx.x) & 31) >> 2)) + 8)])))), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0));
      }
    }
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 16; ++i_7) {
    if ((((((int)blockIdx.y) * 16) + ((i_7 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < M) {
      *(float2*)(C + (((((((((int64_t)((int)blockIdx.y)) * (int64_t)81920) + ((((int64_t)i_7) & (int64_t)1) * (int64_t)40960)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)5120)) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)64)) + ((((int64_t)i_7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(C_local + (i_7 * 2));
    }
  }
}
