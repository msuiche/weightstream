#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void mhc_project_streams_kernel(bfloat16_t* __restrict__ C, const float* __restrict__ Fn, float* __restrict__ P, float* __restrict__ Q, const bfloat16_t* __restrict__ R, bfloat16_t* __restrict__ R_out, float* __restrict__ S, const bfloat16_t* __restrict__ X, const float* __restrict__ comb, const float* __restrict__ post, const float* __restrict__ pre_in, int M);
extern "C" __global__ void __launch_bounds__(128, 1) mhc_project_streams_kernel(bfloat16_t* __restrict__ C, const float* __restrict__ Fn, float* __restrict__ P, float* __restrict__ Q, const bfloat16_t* __restrict__ R, bfloat16_t* __restrict__ R_out, float* __restrict__ S, const bfloat16_t* __restrict__ X, const float* __restrict__ comb, const float* __restrict__ post, const float* __restrict__ pre_in, int M) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* F_low = ((void*)((char*)buf_dyn_shmem + 0));
  void* F_mid = ((void*)((char*)buf_dyn_shmem + 16384));
  void* F_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  float C_local[4];
  float coef[24];
  bfloat16_t xv[16];
  bfloat16_t rv[64];
  float sq[16];
  float col[16];
  bfloat16_t nv[16];
  float av[16];
  float red[1];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(C_local + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(((float*)F_shared) + (((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 768)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(((float*)F_mid) + (((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 768)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(((float*)F_low) + (((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 768)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  }
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    float condval;
    if ((1 <= M)) {
      condval = pre_in[((min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)), (((int64_t)M) - (int64_t)1)) * (int64_t)4) + ((int64_t)k))];
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    coef[k] = condval;
  }
  #pragma unroll
  for (int k_1 = 0; k_1 < 4; ++k_1) {
    float condval_1;
    if ((1 <= M)) {
      condval_1 = post[((min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)), (((int64_t)M) - (int64_t)1)) * (int64_t)4) + ((int64_t)k_1))];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    coef[(k_1 + 4)] = condval_1;
  }
  #pragma unroll
  for (int k_2 = 0; k_2 < 16; ++k_2) {
    float condval_2;
    if ((1 <= M)) {
      condval_2 = comb[((min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)), (((int64_t)M) - (int64_t)1)) * (int64_t)16) + ((int64_t)k_2))];
    } else {
      condval_2 = 0x0p+0f/*0.000000e+00*/;
    }
    coef[(k_2 + 8)] = condval_2;
  }
  for (int v = 0; v < 2; ++v) {
    bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_3;
    if ((1 <= M)) {
      condval_3 = *(uint4*)(X + ((((min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)), (((int64_t)M) - (int64_t)1)) * (int64_t)5120) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16)) + (((int64_t)v) * (int64_t)8)));
    } else {
      condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
    }
    *(uint4*)(xv + (v * 8)) = condval_3;
  }
  #pragma unroll
  for (int m = 0; m < 4; ++m) {
    for (int v_1 = 0; v_1 < 2; ++v_1) {
      bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_4;
      if ((1 <= M)) {
        condval_4 = *(uint4*)(R + (((((min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)), (((int64_t)M) - (int64_t)1)) * (int64_t)20480) + (((int64_t)m) * (int64_t)5120)) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16)) + (((int64_t)v_1) * (int64_t)8)));
      } else {
        condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
      }
      *(uint4*)(rv + ((m * 16) + (v_1 * 8))) = condval_4;
    }
  }
  #pragma unroll
  for (int v_2 = 0; v_2 < 16; ++v_2) {
    sq[v_2] = 0x0p+0f/*0.000000e+00*/;
    col[v_2] = 0x0p+0f/*0.000000e+00*/;
  }
  #pragma unroll
  for (int n = 0; n < 4; ++n) {
    #pragma unroll
    for (int v_3 = 0; v_3 < 16; ++v_3) {
      av[v_3] = (coef[(n + 4)] * ((float)xv[v_3]));
      #pragma unroll
      for (int m_1 = 0; m_1 < 4; ++m_1) {
        av[v_3] = (av[v_3] + (coef[(((m_1 * 4) + n) + 8)] * ((float)rv[((m_1 * 16) + v_3)])));
      }
      nv[v_3] = ((bfloat16_t)av[v_3]);
    }
    if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M) {
      for (int v_4 = 0; v_4 < 2; ++v_4) {
        *(uint4*)(R_out + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)327680) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)20480)) + (((int64_t)n) * (int64_t)5120)) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16)) + (((int64_t)v_4) * (int64_t)8))) = *(uint4*)(nv + (v_4 * 8));
      }
    }
    #pragma unroll
    for (int v_5 = 0; v_5 < 16; ++v_5) {
      av[v_5] = ((float)nv[v_5]);
      ((float*)A_shared)[((((((((((int)threadIdx.x) & 7) >> 1) * 512) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (v_5 >> 3)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + ((v_5 & 7) >> 2)) & 1) * 4)) + (v_5 & 3))] = av[v_5];
      sq[v_5] = (sq[v_5] + (av[v_5] * av[v_5]));
      col[v_5] = (col[v_5] + (coef[n] * av[v_5]));
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 6; ++i_1) {
      float4 f = *(float4*)(Fn + (((((i_1 * 81920) + ((((int)threadIdx.x) >> 5) * 20480)) + (n * 5120)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 31) * 4)));
      uint broadcast_var_6 = (uint)4294959104;
      uint4 __1;
        float4 _reinterpret_tmp = f;
        uint4 v_ = *(uint4 *)(&(_reinterpret_tmp));
        uint4 v__1 = make_uint4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
        __1.x = (v_.x & v__1.x);
        __1.y = (v_.y & v__1.y);
        __1.z = (v_.z & v__1.z);
        __1.w = (v_.w & v__1.w);
      uint4 _reinterpret_tmp_1 = __1;
      *(float4*)(((float*)F_shared) + ((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i_1 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4))) = (*(float4 *)(&(_reinterpret_tmp_1)));
      uint broadcast_var_7 = (uint)4294959104;
      uint broadcast_var_8 = (uint)4294959104;
      uint4 __2;
        float4 __3;
          uint4 __4;
            float4 _reinterpret_tmp_2 = f;
            uint4 v__2 = *(uint4 *)(&(_reinterpret_tmp_2));
            uint4 v__3 = make_uint4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
            __4.x = (v__2.x & v__3.x);
            __4.y = (v__2.y & v__3.y);
            __4.z = (v__2.z & v__3.z);
            __4.w = (v__2.w & v__3.w);
          uint4 _reinterpret_tmp_3 = __4;
          float4 v__4 = *(float4 *)(&(_reinterpret_tmp_3));
          *(float2*)(&(__3.x)) = tl::sub2(*(float2*)(&(f.x)), *(float2*)(&(v__4.x)));
          *(float2*)(&(__3.z)) = tl::sub2(*(float2*)(&(f.z)), *(float2*)(&(v__4.z)));
        float4 _reinterpret_tmp_4 = __3;
        uint4 v__5 = *(uint4 *)(&(_reinterpret_tmp_4));
        uint4 v__6 = make_uint4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
        __2.x = (v__5.x & v__6.x);
        __2.y = (v__5.y & v__6.y);
        __2.z = (v__5.z & v__6.z);
        __2.w = (v__5.w & v__6.w);
      uint4 _reinterpret_tmp_5 = __2;
      *(float4*)(((float*)F_mid) + ((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i_1 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4))) = (*(float4 *)(&(_reinterpret_tmp_5)));
      uint broadcast_var_9 = (uint)4294959104;
      uint broadcast_var_10 = (uint)4294959104;
      uint broadcast_var_11 = (uint)4294959104;
      float4 __5;
        float4 __6;
          uint4 __7;
            float4 _reinterpret_tmp_6 = f;
            uint4 v__7 = *(uint4 *)(&(_reinterpret_tmp_6));
            uint4 v__8 = make_uint4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
            __7.x = (v__7.x & v__8.x);
            __7.y = (v__7.y & v__8.y);
            __7.z = (v__7.z & v__8.z);
            __7.w = (v__7.w & v__8.w);
          uint4 _reinterpret_tmp_7 = __7;
          float4 v__9 = *(float4 *)(&(_reinterpret_tmp_7));
          *(float2*)(&(__6.x)) = tl::sub2(*(float2*)(&(f.x)), *(float2*)(&(v__9.x)));
          *(float2*)(&(__6.z)) = tl::sub2(*(float2*)(&(f.z)), *(float2*)(&(v__9.z)));
        uint4 __8;
          float4 __9;
            uint4 __10;
              float4 _reinterpret_tmp_8 = f;
              uint4 v__10 = *(uint4 *)(&(_reinterpret_tmp_8));
              uint4 v__11 = make_uint4(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10);
              __10.x = (v__10.x & v__11.x);
              __10.y = (v__10.y & v__11.y);
              __10.z = (v__10.z & v__11.z);
              __10.w = (v__10.w & v__11.w);
            uint4 _reinterpret_tmp_9 = __10;
            float4 v__12 = *(float4 *)(&(_reinterpret_tmp_9));
            *(float2*)(&(__9.x)) = tl::sub2(*(float2*)(&(f.x)), *(float2*)(&(v__12.x)));
            *(float2*)(&(__9.z)) = tl::sub2(*(float2*)(&(f.z)), *(float2*)(&(v__12.z)));
          float4 _reinterpret_tmp_10 = __9;
          uint4 v__13 = *(uint4 *)(&(_reinterpret_tmp_10));
          uint4 v__14 = make_uint4(broadcast_var_11, broadcast_var_11, broadcast_var_11, broadcast_var_11);
          __8.x = (v__13.x & v__14.x);
          __8.y = (v__13.y & v__14.y);
          __8.z = (v__13.z & v__14.z);
          __8.w = (v__13.w & v__14.w);
        uint4 _reinterpret_tmp_11 = __8;
        float4 v__15 = *(float4 *)(&(_reinterpret_tmp_11));
        *(float2*)(&(__5.x)) = tl::sub2(*(float2*)(&(__6.x)), *(float2*)(&(v__15.x)));
        *(float2*)(&(__5.z)) = tl::sub2(*(float2*)(&(__6.z)), *(float2*)(&(v__15.z)));
      *(float4*)(((float*)F_low) + ((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i_1 * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 4))) = __5;
    }
    __syncthreads();
    {
      float A_local[4];
      float B_local[2];
      for (int ki = 0; ki < 16; ++ki) {
        tl::ptx_ldmatrix_x4((&(((float*)A_shared)[((((((ki >> 2) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x2((&(((float*)F_shared)[(((((((ki >> 2) * 1024) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      }
    }
    {
      float A_local_1[4];
      float B_local_1[2];
      for (int ki_1 = 0; ki_1 < 16; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((float*)A_shared)[((((((ki_1 >> 2) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(A_local_1[0])));
        tl::ptx_ldmatrix_x2((&(((float*)F_mid)[(((((((ki_1 >> 2) * 1024) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(B_local_1[0])));
        tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
      }
    }
    {
      float A_local_2[4];
      float B_local_2[2];
      for (int ki_2 = 0; ki_2 < 16; ++ki_2) {
        tl::ptx_ldmatrix_x4((&(((float*)A_shared)[((((((ki_2 >> 2) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(A_local_2[0])));
        tl::ptx_ldmatrix_x2((&(((float*)F_low)[(((((((ki_2 >> 2) * 1024) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 3) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4))])), (&(B_local_2[0])));
        tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 0));
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int v_6 = 0; v_6 < 16; ++v_6) {
    nv[v_6] = ((bfloat16_t)col[v_6]);
  }
  if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M) {
    for (int v_7 = 0; v_7 < 2; ++v_7) {
      *(uint4*)(C + (((((((int64_t)((int)blockIdx.y)) * (int64_t)81920) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)5120)) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16)) + (((int64_t)v_7) * (int64_t)8))) = *(uint4*)(nv + (v_7 * 8));
    }
  }
  red[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int v_8 = 0; v_8 < 16; ++v_8) {
    red[0] = (red[0] + (((float)nv[v_8]) * ((float)nv[v_8])));
  }
  red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));
  red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 2, 32));
  red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 1, 32));
  if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)) {
    Q[(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) + (((int64_t)((int)blockIdx.x)) * ((int64_t)M)))] = red[0];
  }
  red[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int v_9 = 0; v_9 < 16; ++v_9) {
    red[0] = (red[0] + sq[v_9]);
  }
  red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));
  red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 2, 32));
  red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 1, 32));
  if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)) {
    S[(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) + (((int64_t)((int)blockIdx.x)) * ((int64_t)M)))] = red[0];
  }
  if (((int)threadIdx.x) < 96) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      if ((((((int)blockIdx.y) * 16) + (i_2 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < M) {
        *(float2*)(P + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)i_2) * (int64_t)192)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)24)) + ((((int64_t)((int)blockIdx.x)) * ((int64_t)M)) * (int64_t)24)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(C_local + (i_2 * 2));
      }
    }
  }
}
