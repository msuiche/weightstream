#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma_block_scale.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void moe_grouped_fp8_fp4_gemm_kernel(const fp8_e4_t* __restrict__ A, __grid_constant__ const CUtensorMap B_desc, float* __restrict__ C, const int* __restrict__ Order, const uint* __restrict__ SFA, const uint* __restrict__ SFB, const int* __restrict__ TileExpert, int tiles, int tokens);
extern "C" __global__ void __launch_bounds__(128, 1) moe_grouped_fp8_fp4_gemm_kernel(const fp8_e4_t* __restrict__ A, __grid_constant__ const CUtensorMap B_desc, float* __restrict__ C, const int* __restrict__ Order, const uint* __restrict__ SFA, const uint* __restrict__ SFB, const int* __restrict__ TileExpert, int tiles, int tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* rows_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* SFA_shared = ((void*)((char*)buf_dyn_shmem + 128));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 1024));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 50176));
  void* sf_dest = ((void*)((char*)buf_dyn_shmem + 62464));
  __shared__ __align__(16) uint64_t loaded_mem[3];
  auto loaded = reinterpret_cast<Barrier*>(loaded_mem);
  int a_row[2];
  int sfa_row[1];
  float C_local[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    loaded[0].init(128);
    loaded[1].init(128);
    loaded[2].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  int expert = TileExpert[((int64_t)((int)blockIdx.y))];
  if (expert < 0) {
    return;
  }
  if (((int)threadIdx.x) < 32) {
    int route = Order[((((int64_t)((int)blockIdx.y)) * (int64_t)32) + ((int64_t)((int)threadIdx.x)))];
    int condval;
    if ((0 <= route)) {
      condval = (route / 6);
    } else {
      condval = -1;
    }
    ((int*)rows_shared)[((int)threadIdx.x)] = condval;
  }
  __syncthreads();
  #pragma unroll
  for (int c = 0; c < 2; ++c) {
    a_row[c] = ((int*)rows_shared)[((c * 16) + (((int)threadIdx.x) >> 3))];
  }
  sfa_row[0] = ((int*)rows_shared)[min(((int)threadIdx.x), 31)];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  __syncthreads();
  #pragma unroll
  for (int s = 0; s < 2; ++s) {
    if (tl::tl_shuffle_elect<128>()) {
      loaded[s].expect_transaction(8192);
      tl::fence_proxy_async();
      tl::tma_load(B_desc, loaded[s], (&(((uint8_t*)B_shared)[(s * 16384)])), 0, 0, (((expert * 720) + (((int)blockIdx.x) * 40)) + s));
    }
    #pragma unroll
    for (int c_1 = 0; c_1 < 2; ++c_1) {
      tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_shared)[((((((s * 4096) + (c_1 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A[(((max(((int64_t)a_row[((int64_t)c_1)]), (int64_t)0) * (int64_t)5120) + (((int64_t)s) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16))])), (((0 <= a_row[c_1]) && (a_row[c_1] < tokens)) && (0 < tokens)));
    }
    if (((int)threadIdx.x) < 32) {
      tl::cp_async_gs_conditional<4>((&(((uint*)SFA_shared)[((s * 32) + ((int)threadIdx.x))])), (&(SFA[((max(((int64_t)sfa_row[(int64_t)0]), (int64_t)0) * (int64_t)40) + ((int64_t)s))])), (((0 <= sfa_row[0]) && (sfa_row[0] < tokens)) && (0 < tokens)));
    }
    tl::cp_async_gs_conditional<4>((&(((uint*)sf_dest)[((s * 128) + ((int)threadIdx.x))])), (&(SFB[((((((int64_t)expert) * (int64_t)92160) + (((int64_t)((int)blockIdx.x)) * (int64_t)5120)) + (((int64_t)((int)threadIdx.x)) * (int64_t)40)) + ((int64_t)s))])), ((expert < 96) && (0 <= expert)));
    tl::cp_async_commit();
    loaded[s].arrive();
  }
  for (int ko = 0; ko < 40; ++ko) {
    __syncthreads();
    if (ko < 38) {
      if (tl::tl_shuffle_elect<128>()) {
        loaded[((ko + 2) % 3)].expect_transaction(8192);
        tl::fence_proxy_async();
        tl::tma_load(B_desc, loaded[((ko + 2) % 3)], (&(((uint8_t*)B_shared)[(((ko + 2) % 3) * 16384)])), 0, 0, ((((expert * 720) + (((int)blockIdx.x) * 40)) + ko) + 2));
      }
      #pragma unroll
      for (int c_2 = 0; c_2 < 2; ++c_2) {
        tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_shared)[((((((((ko + 2) % 3) * 4096) + (c_2 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A[((((max(((int64_t)a_row[((int64_t)c_2)]), (int64_t)0) * (int64_t)5120) + (((int64_t)ko) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16)) + (int64_t)256)])), (((0 <= a_row[c_2]) && (a_row[c_2] < tokens)) && (0 < tokens)));
      }
      if (((int)threadIdx.x) < 32) {
        tl::cp_async_gs_conditional<4>((&(((uint*)SFA_shared)[((((ko + 2) % 3) * 32) + ((int)threadIdx.x))])), (&(SFA[(((max(((int64_t)sfa_row[(int64_t)0]), (int64_t)0) * (int64_t)40) + ((int64_t)ko)) + (int64_t)2)])), (((0 <= sfa_row[0]) && (sfa_row[0] < tokens)) && (0 < tokens)));
      }
      tl::cp_async_gs_conditional<4>((&(((uint*)sf_dest)[((((ko + 2) % 3) * 128) + ((int)threadIdx.x))])), (&(SFB[(((((((int64_t)expert) * (int64_t)92160) + (((int64_t)((int)blockIdx.x)) * (int64_t)5120)) + (((int64_t)((int)threadIdx.x)) * (int64_t)40)) + ((int64_t)ko)) + (int64_t)2)])), ((expert < 96) && (0 <= expert)));
      tl::cp_async_commit();
      loaded[((ko + 2) % 3)].arrive();
    } else {
      tl::cp_async_commit();
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    loaded[(ko % 3)].wait(((ko % 6) / 3));
    __syncthreads();
    {
      fp8_e4_t A_local[16];
      uchar B_local[64];
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_shared)[(((((((ko % 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A_local[0])));
        #pragma unroll
        for (int i_1 = 0; i_1 < 4; ++i_1) {
          tl::ptx_ldmatrix_b4x16_p64_x4((&(((uint8_t*)B_shared)[(((((((((ko % 3) * 16384) + ((((int)threadIdx.x) >> 6) * 8192)) + (i_1 * 2048)) + (((((int)threadIdx.x) & 31) >> 4) * 1024)) + ((((int)threadIdx.x) & 7) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B_local[(i_1 * 16)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE2M1>(reinterpret_cast<float*>(C_local + (j * 8)), reinterpret_cast<const uint32_t*>(A_local + 0), reinterpret_cast<const uint32_t*>(B_local + (j * 16)), reinterpret_cast<const float*>(C_local + (j * 8)), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[(((((ko % 3) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)sf_dest)[(((((ko % 3) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (j * 16)) + ((((int)threadIdx.x) & 31) >> 2))])))), static_cast<uint16_t>(ki), static_cast<uint16_t>(0), static_cast<uint16_t>(ki), static_cast<uint16_t>(0));
          tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE2M1>(reinterpret_cast<float*>(C_local + ((j * 8) + 4)), reinterpret_cast<const uint32_t*>(A_local + 0), reinterpret_cast<const uint32_t*>(B_local + ((j * 16) + 8)), reinterpret_cast<const float*>(C_local + ((j * 8) + 4)), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[(((((ko % 3) * 32) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)sf_dest)[((((((ko % 3) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (j * 16)) + ((((int)threadIdx.x) & 31) >> 2)) + 8)])))), static_cast<uint16_t>(ki), static_cast<uint16_t>(0), static_cast<uint16_t>(ki), static_cast<uint16_t>(0));
        }
      }
    }
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    if (0 <= ((int*)rows_shared)[(((((((int)threadIdx.x) & 63) >> 5) * 16) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]) {
      *(float2*)(C + ((((((((((int64_t)((int)blockIdx.y)) * (int64_t)73728) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)36864)) + ((((int64_t)i_2) & (int64_t)1) * (int64_t)18432)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)2304)) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + ((((int64_t)i_2) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(C_local + (i_2 * 2));
    }
  }
}

