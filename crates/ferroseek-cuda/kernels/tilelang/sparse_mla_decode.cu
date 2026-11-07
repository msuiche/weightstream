#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/cuda_fp4.h>
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void sparse_mla_kernel(const int* __restrict__ IdxIndices, const int* __restrict__ IdxLengths, const int* __restrict__ IdxPageTable, const uchar* __restrict__ IdxRecords, float* __restrict__ Partial, float* __restrict__ PartialStats, const bfloat16_t* __restrict__ Q, const int* __restrict__ SwaIndices, const int* __restrict__ SwaLengths, const uchar* __restrict__ SwaRecords, int idx_pages, int idx_pool, int idx_stride, int rows, int swa_pool, int swa_stride);
extern "C" __global__ void sparse_mla_kernel_1(const float* __restrict__ Cos, const float* __restrict__ Sin, float* __restrict__ Output, const float* __restrict__ Partial, const float* __restrict__ PartialStats, const int64_t* __restrict__ Positions, const float* __restrict__ Sinks, int positions_max, int rows, int segments);
extern "C" __global__ void __launch_bounds__(256, 1) sparse_mla_kernel(const int* __restrict__ IdxIndices, const int* __restrict__ IdxLengths, const int* __restrict__ IdxPageTable, const uchar* __restrict__ IdxRecords, float* __restrict__ Partial, float* __restrict__ PartialStats, const bfloat16_t* __restrict__ Q, const int* __restrict__ SwaIndices, const int* __restrict__ SwaLengths, const uchar* __restrict__ SwaRecords, int idx_pages, int idx_pool, int idx_stride, int rows, int swa_pool, int swa_stride) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* slots = ((void*)((char*)buf_dyn_shmem + 0));
  void* staged_fp4 = ((void*)((char*)buf_dyn_shmem + 512));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 17408));
  void* KV_shared = ((void*)((char*)buf_dyn_shmem + 50176));
  void* P_shared = ((void*)((char*)buf_dyn_shmem + 82944));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 84992));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 86016));
  int copy_slot[5];
  float acc_g[64];
  float l_g[2];
  float m_g[2];
  float acc_s[4];
  float m_prev[2];
  float2 m_g_pack[2];
  float alpha[2];
  float2 row_sum_pack[2];
  float row_sum[2];
  for (int c = 0; c < 4; ++c) {
    if (((int)blockIdx.x) < 1) {
      if (((int)threadIdx.x) < 32) {
        int slot = SwaIndices[((((((int64_t)((int)blockIdx.z)) * (int64_t)128) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + (((int64_t)c) * (int64_t)32)) + ((int64_t)((int)threadIdx.x)))];
        int condval;
        if ((((bool)1 & ((((((int)blockIdx.x) * 128) + (c * 32)) + ((int)threadIdx.x)) < SwaLengths[((int64_t)((int)blockIdx.z))])) & (0 <= slot))) {
          condval = slot;
        } else {
          condval = -1;
        }
        ((int*)slots)[((c * 32) + ((int)threadIdx.x))] = condval;
      }
    } else {
      if (((int)threadIdx.x) < 32) {
        int logical = IdxIndices[(((((((int64_t)((int)blockIdx.z)) * (int64_t)2048) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + (((int64_t)c) * (int64_t)32)) + ((int64_t)((int)threadIdx.x))) - (int64_t)128)];
        int condval_1;
        if ((1 <= idx_pages)) {
          condval_1 = IdxPageTable[(min((max(((int64_t)logical), (int64_t)0) >> (int64_t)6), (((int64_t)idx_pages) - (int64_t)1)) + (((int64_t)((int)blockIdx.z)) * ((int64_t)idx_pages)))];
        } else {
          condval_1 = 0;
        }
        int page = condval_1;
        int condval_2;
        if ((((bool)1 & ((((((int)blockIdx.x) * 128) + (c * 32)) + ((int)threadIdx.x)) < (IdxLengths[((int64_t)((int)blockIdx.z))] + 128))) & (0 <= logical))) {
          condval_2 = ((page * 64) + (logical & 63));
        } else {
          condval_2 = -1;
        }
        ((int*)slots)[((c * 32) + ((int)threadIdx.x))] = condval_2;
      }
    }
  }
  __syncthreads();
  if (((int)blockIdx.x) < 1) {
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
      copy_slot[k] = ((int*)slots)[min((((k * 256) + ((int)threadIdx.x)) / 33), 31)];
    }
    __syncthreads();
    #pragma unroll
    for (int k_1 = 0; k_1 < 5; ++k_1) {
      if (((k_1 * 8) + (((int)threadIdx.x) >> 5)) < 33) {
        tl::cp_async_gs_conditional<16>((&(((uchar*)staged_fp4)[((k_1 * 4096) + (((int)threadIdx.x) * 16))])), (&(SwaRecords[(((((((int64_t)k_1) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) % (int64_t)33) * (int64_t)16) + (max(((int64_t)copy_slot[((int64_t)k_1)]), (int64_t)0) * ((int64_t)swa_stride)))])), (((0 <= copy_slot[k_1]) && (copy_slot[k_1] < swa_pool)) && (0 < swa_pool)));
      }
    }
    tl::cp_async_commit();
  } else {
    #pragma unroll
    for (int k_2 = 0; k_2 < 3; ++k_2) {
      copy_slot[k_2] = ((int*)slots)[min((((k_2 * 128) + (((int)threadIdx.x) >> 1)) / 9), 31)];
    }
    __syncthreads();
    #pragma unroll
    for (int k_3 = 0; k_3 < 3; ++k_3) {
      if (((k_3 * 4) + (((int)threadIdx.x) >> 6)) < 9) {
        tl::cp_async_gs_conditional<16>((&(((uchar*)staged_fp4)[(((((k_3 * 128) + (((int)threadIdx.x) >> 1)) / 9) * 528) + ((((k_3 * 256) + ((int)threadIdx.x)) % 18) * 16))])), (&(IdxRecords[((((max(((int64_t)copy_slot[((int64_t)k_3)]), (int64_t)0) & (int64_t)63) * (int64_t)288) + ((((((int64_t)k_3) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) % (int64_t)18) * (int64_t)16)) + ((max(((int64_t)copy_slot[((int64_t)k_3)]), (int64_t)0) >> (int64_t)6) * ((int64_t)idx_stride)))])), ((0 <= copy_slot[k_3]) && ((max(copy_slot[k_3], 0) >> 6) < idx_pool)));
      }
    }
    tl::cp_async_commit();
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_3;
    if ((i < 4)) {
      condval_3 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.z)) * (int64_t)8192) + (((int64_t)i) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
    } else {
      condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(((bfloat16_t*)Q_shared) + ((((((((((int)threadIdx.x) & 63) >> 3) * 2048) + (i * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_g + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(l_g + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = -CUDART_INF_F;
  *(float2*)(m_g + 0) = make_float2(broadcast_var_3, broadcast_var_3);
  for (int c_1 = 0; c_1 < 4; ++c_1) {
    tl::cp_async_wait<0>();
    __syncthreads();
    if (((int)blockIdx.x) < 1) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        fp8_e4_t staged_fp8_local_cast_1[8];
        uchar staged_local_cast_2[8];
        bfloat16_t KV_shared_local_cast[8];
        *(fp8_e4_8_t*)(staged_fp8_local_cast_1 + 0) = *(fp8_e4_8_t*)(((fp8_e4_t*)staged_fp4) + (((i_2 * 2112) + ((((int)threadIdx.x) >> 6) * 528)) + ((((int)threadIdx.x) & 63) * 8)));
        uchar v_ = ((uchar*)staged_fp4)[((((i_2 * 2112) + ((((int)threadIdx.x) >> 6) * 528)) + ((((int)threadIdx.x) & 63) >> 2)) + 512)];
        *(uint2*)(staged_local_cast_2 + 0) = tl::broadcast<uint2>(v_);
        for (int vec = 0; vec < 8; ++vec) {
          uint v__1 = ((uint)staged_local_cast_2[vec]) << (uint)23;
          KV_shared_local_cast[vec] = ((bfloat16_t)(((float)staged_fp8_local_cast_1[vec]) * (*(float *)(&(v__1)))));
        }
        *(uint4*)(((bfloat16_t*)KV_shared) + ((((((((((int)threadIdx.x) & 63) >> 3) * 2048) + (i_2 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(KV_shared_local_cast + 0);
      }
    } else {
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        fp4_e2_2_t staged_fp4_local_cast_4_packed[4];
        fp8_e4_t staged_fp8_local_cast_5[8];
        bfloat16_t KV_shared_local_cast_3[8];
        *(fp4_e2_8_t*)(staged_fp4_local_cast_4_packed + 0) = *(fp4_e2_8_t*)(((fp4_e2_t*)staged_fp4) + (((i_3 * 2112) + ((((int)threadIdx.x) >> 6) * 528)) + ((((int)threadIdx.x) & 63) * 4)));
        fp8_e4_t v__2 = ((fp8_e4_t*)staged_fp4)[((((i_3 * 2112) + ((((int)threadIdx.x) >> 6) * 528)) + ((((int)threadIdx.x) & 63) >> 1)) + 256)];
        *(fp8_e4_8_t*)(staged_fp8_local_cast_5 + 0) = tl::broadcast<fp8_e4_8_t>(v__2);
        for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
          uint2 __1;
          float4 __2;
            float4 __3;
            fp4_e2_4_t v__3 = *(fp4_e2_4_t*)(staged_fp4_local_cast_4_packed + (vec_1 * 2));
            ((float2*)(&__3))[0] = __tl_cvt_fp4x2_to_float2((reinterpret_cast<uint8_t*>(&v__3))[0]);
            ((float2*)(&__3))[1] = __tl_cvt_fp4x2_to_float2((reinterpret_cast<uint8_t*>(&v__3))[1]);
            float4 __4;
            fp8_e4_4_t v__4 = *(fp8_e4_4_t*)(staged_fp8_local_cast_5 + (vec_1 * 4));
            ((float2*)(&__4))[0] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__4))[0], __NV_E4M3);
            ((float2*)(&__4))[1] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__4))[1], __NV_E4M3);
            *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(__3.x)), *(float2*)(&(__4.x)));
            *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(__3.z)), *(float2*)(&(__4.z)));
          (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
          (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&__2))[1]);
          *(uint2*)(KV_shared_local_cast_3 + (vec_1 * 4)) = __1;
        }
        *(uint4*)(((bfloat16_t*)KV_shared) + ((((((((((int)threadIdx.x) & 63) >> 3) * 2048) + (i_3 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(KV_shared_local_cast_3 + 0);
      }
    }
    __syncthreads();
    if ((c_1 < 3) & (((((int)blockIdx.x) * 4) + c_1) < 67)) {
      if (((((int)blockIdx.x) * 4) + c_1) < 3) {
        #pragma unroll
        for (int k_4 = 0; k_4 < 5; ++k_4) {
          copy_slot[k_4] = ((int*)slots)[(((c_1 * 32) + min((((k_4 * 256) + ((int)threadIdx.x)) / 33), 31)) + 32)];
        }
        __syncthreads();
        #pragma unroll
        for (int k_5 = 0; k_5 < 5; ++k_5) {
          if (((k_5 * 8) + (((int)threadIdx.x) >> 5)) < 33) {
            tl::cp_async_gs_conditional<16>((&(((uchar*)staged_fp4)[((k_5 * 4096) + (((int)threadIdx.x) * 16))])), (&(SwaRecords[(((((((int64_t)k_5) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) % (int64_t)33) * (int64_t)16) + (max(((int64_t)copy_slot[((int64_t)k_5)]), (int64_t)0) * ((int64_t)swa_stride)))])), (((0 <= copy_slot[k_5]) && (copy_slot[k_5] < swa_pool)) && (0 < swa_pool)));
          }
        }
        tl::cp_async_commit();
      } else {
        #pragma unroll
        for (int k_6 = 0; k_6 < 3; ++k_6) {
          copy_slot[k_6] = ((int*)slots)[(((c_1 * 32) + min((((k_6 * 128) + (((int)threadIdx.x) >> 1)) / 9), 31)) + 32)];
        }
        __syncthreads();
        #pragma unroll
        for (int k_7 = 0; k_7 < 3; ++k_7) {
          if (((k_7 * 4) + (((int)threadIdx.x) >> 6)) < 9) {
            tl::cp_async_gs_conditional<16>((&(((uchar*)staged_fp4)[(((((k_7 * 128) + (((int)threadIdx.x) >> 1)) / 9) * 528) + ((((k_7 * 256) + ((int)threadIdx.x)) % 18) * 16))])), (&(IdxRecords[((((max(((int64_t)copy_slot[((int64_t)k_7)]), (int64_t)0) & (int64_t)63) * (int64_t)288) + ((((((int64_t)k_7) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) % (int64_t)18) * (int64_t)16)) + ((max(((int64_t)copy_slot[((int64_t)k_7)]), (int64_t)0) >> (int64_t)6) * ((int64_t)idx_stride)))])), ((0 <= copy_slot[k_7]) && ((max(copy_slot[k_7], 0) >> 6) < idx_pool)));
          }
        }
        tl::cp_async_commit();
      }
    }
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[4];
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + 0) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
      __syncthreads();
      for (int ki = 0; ki < 32; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)Q_shared)[(((((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x2((&(((bfloat16_t*)KV_shared)[(((((((ki >> 2) * 2048) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      float condval_4;
      if ((0 <= ((int*)slots)[((((c_1 * 32) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1))])) {
        condval_4 = __fmul_rn(acc_s[i_4], 0x1.0527dbd5cafffp-4f/*6.375872e-02*/);
      } else {
        condval_4 = -CUDART_INF_F;
      }
      acc_s[i_4] = condval_4;
    }
    *(float2*)(m_prev + 0) = *(float2*)(m_g + 0);
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      float broadcast_var_5 = -CUDART_INF_F;
      m_g_pack[i_5] = make_float2(broadcast_var_5, broadcast_var_5);
      m_g_pack[i_5] = tl::max2(m_g_pack[i_5], *(float2*)(acc_s + (i_5 * 2)));
      m_g[i_5] = max((m_g_pack[i_5]).x, (m_g_pack[i_5]).y);
      __syncthreads();
      m_g[i_5] = tl::AllReduce<tl::MaxOp, 256, 64, 0, tl::NamedBarrier<256>>::run(m_g[i_5], (&(((float*)workspace)[0])));
      m_g[i_5] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(m_g[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      m_g[i_6] = max(m_g[i_6], m_prev[i_6]);
      float condval_5;
      if ((m_prev[i_6] == -CUDART_INF_F)) {
        condval_5 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_5 = exp2f((m_prev[i_6] - m_g[i_6]));
      }
      alpha[i_6] = condval_5;
    }
    __syncthreads();
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      float condval_6;
      if ((0 <= ((int*)slots)[((((c_1 * 32) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))])) {
        condval_6 = exp2f((acc_s[i_7] - m_g[(i_7 >> 1)]));
      } else {
        condval_6 = 0x0p+0f/*0.000000e+00*/;
      }
      acc_s[i_7] = condval_6;
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      row_sum_pack[i_8] = make_float2(broadcast_var_6, broadcast_var_6);
      row_sum_pack[i_8] = tl::add2(row_sum_pack[i_8], *(float2*)(acc_s + (i_8 * 2)));
      row_sum[i_8] = ((row_sum_pack[i_8]).x + (row_sum_pack[i_8]).y);
      __syncthreads();
      row_sum[i_8] = tl::AllReduce<tl::SumOp, 256, 64, 0, tl::NamedBarrier<256>>::run(row_sum[i_8], (&(((float*)workspace_1)[0])));
      row_sum[i_8] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(row_sum[i_8]);
    }
    float2 __5;
      float2 v__5 = *(float2*)(l_g + 0);
      float2 v__6 = *(float2*)(alpha + 0);
      float2 v__7 = *(float2*)(row_sum + 0);
      *(float2*)(&(__5.x)) = tl::fma2(*(float2*)(&(v__5.x)), *(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
    *(float2*)(l_g + 0) = __5;
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      float2 __6;
        float2 v__8 = *(float2*)(acc_g + (i_9 * 2));
        float2 v__9 = make_float2(alpha[(i_9 & 1)], alpha[(i_9 & 1)]);
        *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(v__8.x)), *(float2*)(&(v__9.x)));
      *(float2*)(acc_g + (i_9 * 2)) = __6;
    }
    tl::ptx_stmatrix_m8n8_x2((&(((bfloat16_t*)P_shared)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)acc_s[0]), ((bfloat16_t)acc_s[1])), __pack_half2(((bfloat16_t)acc_s[2]), ((bfloat16_t)acc_s[3])));
    {
      bfloat16_t A_local_1[8];
      bfloat16_t B_local_1[64];
      __syncthreads();
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)P_shared)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[0])));
        #pragma unroll
        for (int i_10 = 0; i_10 < 8; ++i_10) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)KV_shared)[((((((((((int)threadIdx.x) >> 6) * 4096) + ((i_10 >> 2) * 2048)) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_10 * 8)])));
        }
        for (int j = 0; j < 8; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_g + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_g + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
        }
      }
    }
  }
  if ((((int)threadIdx.x) & 63) < 32) {
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      *(float2*)(Partial + (((((((((int64_t)((int)blockIdx.z)) * (int64_t)139264) + (((int64_t)((int)blockIdx.x)) * (int64_t)8192)) + ((((int64_t)i_11) & (int64_t)1) * (int64_t)4096)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)128)) + ((((int64_t)i_11) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(acc_g + (i_11 * 2));
    }
  }
  if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 6)) == 0) {
    if ((((int)threadIdx.x) & 63) < 32) {
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        PartialStats[((((((int64_t)((int)blockIdx.z)) * (int64_t)544) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + (((int64_t)i_12) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)2))] = m_g[i_12];
        PartialStats[(((((((int64_t)((int)blockIdx.z)) * (int64_t)544) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + (((int64_t)i_12) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)2)) + (int64_t)1)] = l_g[i_12];
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(128, 1) sparse_mla_kernel_1(const float* __restrict__ Cos, const float* __restrict__ Sin, float* __restrict__ Output, const float* __restrict__ Partial, const float* __restrict__ PartialStats, const int64_t* __restrict__ Positions, const float* __restrict__ Sinks, int positions_max, int rows, int segments) {
  float M = 0x0p+0f/*0.000000e+00*/;
  float L = 0x0p+0f/*0.000000e+00*/;
  float o[4];
  float m_new = 0x0p+0f/*0.000000e+00*/;
  float a = 0x0p+0f/*0.000000e+00*/;
  float b = 0x0p+0f/*0.000000e+00*/;
  extern __shared__ __align__(1024) bfloat16_t O_shared[];
  float X[4];
  float2 amax_pack[1];
  float amax[1];
  float inv[1];
  float scale[1];
  M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);
  L = 0x1p+0f/*1.000000e+00*/;
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(o + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  #pragma unroll
  for (int c = 0; c < segments; ++c) {
    m_new = max(M, PartialStats[(((((int64_t)((int)blockIdx.y)) * (int64_t)544) + (((int64_t)c) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) * (int64_t)2))]);
    a = exp2f((M - m_new));
    b = exp2f((PartialStats[(((((int64_t)((int)blockIdx.y)) * (int64_t)544) + (((int64_t)c) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) * (int64_t)2))] - m_new));
    L = __fmaf_rn(PartialStats[((((((int64_t)((int)blockIdx.y)) * (int64_t)544) + (((int64_t)c) * (int64_t)32)) + (((int64_t)((int)blockIdx.x)) * (int64_t)2)) + (int64_t)1)], b, __fmul_rn(L, a));
    M = m_new;
    float4 __1;
      float4 v_ = *(float4*)(Partial + ((((((int64_t)((int)blockIdx.y)) * (int64_t)139264) + (((int64_t)c) * (int64_t)8192)) + (((int64_t)((int)blockIdx.x)) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
      float4 v__1 = make_float4(b, b, b, b);
      float4 __2;
        float4 v__2 = *(float4*)(o + 0);
        float4 v__3 = make_float4(a, a, a, a);
        *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
        *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(v__2.z)), *(float2*)(&(v__3.z)));
      *(float2*)(&(__1.x)) = tl::fma2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)), *(float2*)(&(__2.x)));
      *(float2*)(&(__1.z)) = tl::fma2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)), *(float2*)(&(__2.z)));
    *(float4*)(o + 0) = __1;
  }
  L = (0x1p+0f/*1.000000e+00*/ / L);
  {
    bfloat16_t O_shared_local_cast_6[4];
    uint2 __3;
    float4 __4;
      float4 v__4 = *(float4*)(o + 0);
      float4 v__5 = make_float4(L, L, L, L);
      *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
      *(float2*)(&(__4.z)) = tl::mul2(*(float2*)(&(v__4.z)), *(float2*)(&(v__5.z)));
    (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&__4))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&__4))[1]);
    *(uint2*)(O_shared_local_cast_6 + 0) = __3;
    *(uint2*)(O_shared + (((int)threadIdx.x) * 4)) = *(uint2*)(O_shared_local_cast_6 + 0);
  }
  int64_t position = Positions[((int64_t)((int)blockIdx.y))];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    float first = ((float)O_shared[((((int)threadIdx.x) * 2) + 448)]);
    float second = ((float)O_shared[((((int)threadIdx.x) * 2) + 449)]);
    float condval;
    if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position))) {
      condval = Cos[((position * (int64_t)64) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    float cosine = condval;
    float condval_1;
    if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position))) {
      condval_1 = Sin[(((position * (int64_t)64) + ((int64_t)((int)threadIdx.x))) + (int64_t)32)];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    float sine = condval_1;
    O_shared[((((int)threadIdx.x) * 2) + 448)] = ((bfloat16_t)__fmaf_rn(first, cosine, (second * sine)));
    O_shared[((((int)threadIdx.x) * 2) + 449)] = ((bfloat16_t)__fmaf_rn(first, (sine * -0x1p+0f/*-1.000000e+00*/), (second * cosine)));
  }
  {
    bfloat16_t O_shared_local_cast_7[4];
    __syncthreads();
    *(uint2*)(O_shared_local_cast_7 + 0) = *(uint2*)(O_shared + (((int)threadIdx.x) * 4));
    float4 __5;
    uint2 v__6 = *(uint2*)(O_shared_local_cast_7 + 0);
    ((float2*)(&__5))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[0]);
    ((float2*)(&__5))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[1]);
    *(float4*)(X + 0) = __5;
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  amax_pack[0] = make_float2(broadcast_var_1, broadcast_var_1);
  #pragma unroll
  for (int rv = 0; rv < 2; ++rv) {
    amax_pack[0] = tl::max2(amax_pack[0], tl::abs2(*(float2*)(X + (rv * 2))));
  }
  amax[0] = max((amax_pack[0]).x, max((amax_pack[0]).y, (0x0p+0f/*0.000000e+00*/ - (amax_pack[0]).y)));
  amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);
  float floored = max(amax[0], 0x1.a36e2eb1c432dp-14f/*1.000000e-04*/);
  float v__7 = floored * 0x1.2492492492492p-9f/*2.232143e-03*/;
  uint v__8 = ((uint)(254 - ((int)((((*(uint *)(&(v__7))) - (uint)1) >> (uint)23) + (uint)1)))) << (uint)23;
  inv[0] = (*(float *)(&(v__8)));
  float v__9 = floored * 0x1.2492492492492p-9f/*2.232143e-03*/;
  uint v__10 = ((uint)((int)((((*(uint *)(&(v__9))) - (uint)1) >> (uint)23) + (uint)1))) << (uint)23;
  scale[0] = (*(float *)(&(v__10)));
  {
    bfloat16_t O_shared_local_cast_8[4];
    uint2 __6;
    float4 __7;
      float4 __8;
      fp8_e4_4_t __9;
      float4 __10;
        float4 v__11 = *(float4*)(X + 0);
        float4 v__12 = make_float4(inv[0], inv[0], inv[0], inv[0]);
        *(float2*)(&(__10.x)) = tl::mul2(*(float2*)(&(v__11.x)), *(float2*)(&(v__12.x)));
        *(float2*)(&(__10.z)) = tl::mul2(*(float2*)(&(v__11.z)), *(float2*)(&(v__12.z)));
      (reinterpret_cast<__nv_fp8x2_storage_t*>(&__9))[0] = __nv_cvt_float2_to_fp8x2(((float2*)(&__10))[0], __NV_SATFINITE, __NV_E4M3);
      (reinterpret_cast<__nv_fp8x2_storage_t*>(&__9))[1] = __nv_cvt_float2_to_fp8x2(((float2*)(&__10))[1], __NV_SATFINITE, __NV_E4M3);
      ((float2*)(&__8))[0] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&__9))[0], __NV_E4M3);
      ((float2*)(&__8))[1] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&__9))[1], __NV_E4M3);
      float4 v__13 = make_float4(scale[0], scale[0], scale[0], scale[0]);
      *(float2*)(&(__7.x)) = tl::mul2(*(float2*)(&(__8.x)), *(float2*)(&(v__13.x)));
      *(float2*)(&(__7.z)) = tl::mul2(*(float2*)(&(__8.z)), *(float2*)(&(v__13.z)));
    (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&__7))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&__7))[1]);
    *(uint2*)(O_shared_local_cast_8 + 0) = __6;
    *(uint2*)(O_shared + (((int)threadIdx.x) * 4)) = *(uint2*)(O_shared_local_cast_8 + 0);
  }
  *(float4*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blockIdx.x)) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4))) = make_float4((float)O_shared[((int)threadIdx.x) * 4], (float)O_shared[((int)threadIdx.x) * 4 + 1], (float)O_shared[((int)threadIdx.x) * 4 + 2], (float)O_shared[((int)threadIdx.x) * 4 + 3]);
}

