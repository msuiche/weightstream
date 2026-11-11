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
extern "C" __global__ void __launch_bounds__(128, 1) gemm_mxfp8_kernel(const fp8_e4_t* __restrict__ A, const fp8_e4_t* __restrict__ B, float* __restrict__ C, const uint* __restrict__ SFA, const uint* __restrict__ SFB, int M, int Mp) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 6144));
  void* SFA_shared = ((void*)((char*)buf_dyn_shmem + 30720));
  void* SFB_shared = ((void*)((char*)buf_dyn_shmem + 30912));
  float C_local[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_shared)[(((((((int)threadIdx.x) >> 3) * 128) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A[(((((int64_t)((int)blockIdx.y)) * (int64_t)36864) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)2304)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16))])), ((((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < Mp) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < Mp)));
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)B_shared)[(((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B[((((((int)blockIdx.x) * 147456) + (i_1 * 36864)) + ((((int)threadIdx.x) >> 3) * 2304)) + ((((int)threadIdx.x) & 7) * 16))])));
  }
  if (((int)threadIdx.x) < 16) {
    tl::cp_async_gs_conditional<4>((&(((uint*)SFA_shared)[((int)threadIdx.x)])), (&(SFA[(min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + ((int64_t)((int)threadIdx.x))), (((int64_t)Mp) - (int64_t)1)) * (int64_t)18)])), (1 <= Mp));
  }
  if (((int)threadIdx.x) < 64) {
    tl::cp_async_gs<4>((&(((uint*)SFB_shared)[((int)threadIdx.x)])), (&(SFB[((((int)blockIdx.x) * 1152) + (((int)threadIdx.x) * 18))])));
  }
  tl::cp_async_commit();
  tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_shared)[((((((((int)threadIdx.x) >> 3) * 128) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 2048)])), (&(A[((((((int64_t)((int)blockIdx.y)) * (int64_t)36864) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)2304)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16)) + (int64_t)128)])), ((((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < Mp) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < Mp)));
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)B_shared)[((((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192)])), (&(B[(((((((int)blockIdx.x) * 147456) + (i_2 * 36864)) + ((((int)threadIdx.x) >> 3) * 2304)) + ((((int)threadIdx.x) & 7) * 16)) + 128)])));
  }
  if (((int)threadIdx.x) < 16) {
    tl::cp_async_gs_conditional<4>((&(((uint*)SFA_shared)[(((int)threadIdx.x) + 16)])), (&(SFA[((min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + ((int64_t)((int)threadIdx.x))), (((int64_t)Mp) - (int64_t)1)) * (int64_t)18) + (int64_t)1)])), (1 <= Mp));
  }
  if (((int)threadIdx.x) < 64) {
    tl::cp_async_gs<4>((&(((uint*)SFB_shared)[(((int)threadIdx.x) + 64)])), (&(SFB[(((((int)blockIdx.x) * 1152) + (((int)threadIdx.x) * 18)) + 1)])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 16; ++ko) {
    __syncthreads();
    tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_shared)[(((((((ko + 2) % 3) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A[(((((((int64_t)((int)blockIdx.y)) * (int64_t)36864) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)2304)) + (((int64_t)ko) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16)) + (int64_t)256)])), ((((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < Mp) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < Mp)));
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs<16>((&(((fp8_e4_t*)B_shared)[((((((((ko + 2) % 3) * 8192) + (i_3 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B[((((((((int)blockIdx.x) * 147456) + (i_3 * 36864)) + ((((int)threadIdx.x) >> 3) * 2304)) + (ko * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 256)])));
    }
    if (((int)threadIdx.x) < 16) {
      tl::cp_async_gs_conditional<4>((&(((uint*)SFA_shared)[((((ko + 2) % 3) * 16) + ((int)threadIdx.x))])), (&(SFA[(((min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + ((int64_t)((int)threadIdx.x))), (((int64_t)Mp) - (int64_t)1)) * (int64_t)18) + ((int64_t)ko)) + (int64_t)2)])), (1 <= Mp));
    }
    if (((int)threadIdx.x) < 64) {
      tl::cp_async_gs<4>((&(((uint*)SFB_shared)[((((ko + 2) % 3) * 64) + ((int)threadIdx.x))])), (&(SFB[((((((int)blockIdx.x) * 1152) + (((int)threadIdx.x) * 18)) + ko) + 2)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      fp8_e4_t A_local[16];
      fp8_e4_t B_local[16];
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_shared)[((((((ko % 3) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)B_shared)[(((((((ko % 3) * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)threadIdx.x) & 7) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B_local[0])));
        tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const uint32_t*>(A_local + 0), reinterpret_cast<const uint32_t*>(B_local + 0), reinterpret_cast<const float*>(C_local + 0), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((ko % 3) * 16) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[((((ko % 3) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((((int)threadIdx.x) & 31) >> 2))])))), static_cast<uint16_t>(ki), static_cast<uint16_t>(0), static_cast<uint16_t>(ki), static_cast<uint16_t>(0));
        tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + 4), reinterpret_cast<const uint32_t*>(A_local + 0), reinterpret_cast<const uint32_t*>(B_local + 8), reinterpret_cast<const float*>(C_local + 4), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((ko % 3) * 16) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[(((((ko % 3) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((((int)threadIdx.x) & 31) >> 2)) + 8)])))), static_cast<uint16_t>(ki), static_cast<uint16_t>(0), static_cast<uint16_t>(ki), static_cast<uint16_t>(0));
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    fp8_e4_t A_local_1[16];
    fp8_e4_t B_local_1[16];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_shared)[((((((((int)threadIdx.x) & 15) * 128) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 2048)])), (&(A_local_1[0])));
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)B_shared)[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 7) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192)])), (&(B_local_1[0])));
      tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const uint32_t*>(A_local_1 + 0), reinterpret_cast<const uint32_t*>(B_local_1 + 0), reinterpret_cast<const float*>(C_local + 0), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((((int)threadIdx.x) & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 16)])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((((int)threadIdx.x) & 31) >> 2)) + 64)])))), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0));
      tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + 4), reinterpret_cast<const uint32_t*>(A_local_1 + 0), reinterpret_cast<const uint32_t*>(B_local_1 + 8), reinterpret_cast<const float*>(C_local + 4), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((((int)threadIdx.x) & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 16)])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((((int)threadIdx.x) & 31) >> 2)) + 72)])))), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0));
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    fp8_e4_t A_local_2[16];
    fp8_e4_t B_local_2[16];
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_shared)[((((((((int)threadIdx.x) & 15) * 128) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 4096)])), (&(A_local_2[0])));
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)B_shared)[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 7) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384)])), (&(B_local_2[0])));
      tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const uint32_t*>(A_local_2 + 0), reinterpret_cast<const uint32_t*>(B_local_2 + 0), reinterpret_cast<const float*>(C_local + 0), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((((int)threadIdx.x) & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((((int)threadIdx.x) & 31) >> 2)) + 128)])))), static_cast<uint16_t>(ki_2), static_cast<uint16_t>(0), static_cast<uint16_t>(ki_2), static_cast<uint16_t>(0));
      tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + 4), reinterpret_cast<const uint32_t*>(A_local_2 + 0), reinterpret_cast<const uint32_t*>(B_local_2 + 8), reinterpret_cast<const float*>(C_local + 4), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((((int)threadIdx.x) & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 32)])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((((int)threadIdx.x) & 31) >> 2)) + 136)])))), static_cast<uint16_t>(ki_2), static_cast<uint16_t>(0), static_cast<uint16_t>(ki_2), static_cast<uint16_t>(0));
    }
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    if ((((((int)blockIdx.y) * 16) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < M) {
      *(float2*)(C + (((((((((int64_t)((int)blockIdx.y)) * (int64_t)81920) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)40960)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)5120)) + (((int64_t)((int)blockIdx.x)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + ((((int64_t)i_4) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(C_local + (i_4 * 2));
    }
  }
}

