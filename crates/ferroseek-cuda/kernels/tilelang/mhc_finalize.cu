#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void mhc_finalize_kernel(const bfloat16_t* __restrict__ C, const float* __restrict__ P, const float* __restrict__ Q, const float* __restrict__ S, const float* __restrict__ base, float* __restrict__ comb, float* __restrict__ post, float* __restrict__ pre_out, const float* __restrict__ scale, const bfloat16_t* __restrict__ weight, bfloat16_t* __restrict__ y, int M);
extern "C" __global__ void __launch_bounds__(128, 1) mhc_finalize_kernel(const bfloat16_t* __restrict__ C, const float* __restrict__ P, const float* __restrict__ Q, const float* __restrict__ S, const float* __restrict__ base, float* __restrict__ comb, float* __restrict__ post, float* __restrict__ pre_out, const float* __restrict__ scale, const bfloat16_t* __restrict__ weight, bfloat16_t* __restrict__ y, int M) {
  float rms[1];
  float mixes[1];
  extern __shared__ __align__(1024) float mixes_shared[];
  float cm[1];
  float row_max[1];
  float row_sum[1];
  float col_sum[1];
  float total[1];
  bfloat16_t yv[8];
  bfloat16_t wv[8];
  bfloat16_t xv[8];
  if (((int)threadIdx.x) < 32) {
    rms[0] = 0x0p+0f/*0.000000e+00*/;
    for (int i_split = 0; i_split < 40; ++i_split) {
      rms[0] = (rms[0] + S[((((int64_t)i_split) * ((int64_t)M)) + ((int64_t)((int)blockIdx.x)))]);
    }
    rms[0] = rsqrtf(((rms[0] / 0x1.4p+14f/*2.048000e+04*/) + 0x1.79ca10c924223p-67f/*1.000000e-20*/));
    if (((int)threadIdx.x) < 24) {
      mixes[0] = 0x0p+0f/*0.000000e+00*/;
      for (int i_split_1 = 0; i_split_1 < 40; ++i_split_1) {
        mixes[0] = (mixes[0] + P[(((((int64_t)((int)blockIdx.x)) * (int64_t)24) + ((((int64_t)i_split_1) * ((int64_t)M)) * (int64_t)24)) + ((int64_t)((int)threadIdx.x)))]);
      }
      mixes[0] = (mixes[0] * rms[0]);
      mixes_shared[((int)threadIdx.x)] = mixes[0];
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    if (((int)threadIdx.x) < 4) {
      pre_out[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + ((int64_t)((int)threadIdx.x)))] = ((0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - ((mixes_shared[((int)threadIdx.x)] * scale[0]) + base[((int)threadIdx.x)]))))) + 0x1.a36e2eb1c432dp-14f/*1.000000e-04*/);
      post[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + ((int64_t)((int)threadIdx.x)))] = ((0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - ((mixes_shared[(((int)threadIdx.x) + 4)] * scale[1]) + base[(((int)threadIdx.x) + 4)]))))) * 0x1p+1f/*2.000000e+00*/);
    }
    cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadIdx.x) & 15) + 8)]);
    row_max[0] = -CUDART_INF_F;
    row_max[0] = max(row_max[0], cm[0]);
    row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);
    cm[0] = expf((cm[0] - row_max[0]));
    row_sum[0] = 0x0p+0f/*0.000000e+00*/;
    row_sum[0] = (row_sum[0] + cm[0]);
    row_sum[0] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_sum[0]);
    cm[0] = ((cm[0] / row_sum[0]) + 0x1.a36e2eb1c432dp-14f/*1.000000e-04*/);
    col_sum[0] = 0x0p+0f/*0.000000e+00*/;
    col_sum[0] = (col_sum[0] + cm[0]);
    col_sum[0] = tl::AllReduce<tl::SumOp, 16, 4, 0, tl::NamedBarrier<32>>::run(col_sum[0]);
    cm[0] = (cm[0] / (col_sum[0] + 0x1.a36e2eb1c432dp-14f/*1.000000e-04*/));
    for (int __1 = 0; __1 < 18; ++__1) {
      row_sum[0] = 0x0p+0f/*0.000000e+00*/;
      row_sum[0] = (row_sum[0] + cm[0]);
      row_sum[0] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_sum[0]);
      cm[0] = (cm[0] / (row_sum[0] + 0x1.a36e2eb1c432dp-14f/*1.000000e-04*/));
      col_sum[0] = 0x0p+0f/*0.000000e+00*/;
      col_sum[0] = (col_sum[0] + cm[0]);
      col_sum[0] = tl::AllReduce<tl::SumOp, 16, 4, 0, tl::NamedBarrier<32>>::run(col_sum[0]);
      cm[0] = (cm[0] / (col_sum[0] + 0x1.a36e2eb1c432dp-14f/*1.000000e-04*/));
    }
    if ((((int)threadIdx.x) >> 4) == 0) {
      comb[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) & (int64_t)15))] = cm[0];
    }
  } else {
    total[0] = 0x0p+0f/*0.000000e+00*/;
    for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2) {
      total[0] = (total[0] + Q[((((int64_t)i_split_2) * ((int64_t)M)) + ((int64_t)((int)blockIdx.x)))]);
    }
    total[0] = rsqrtf(((total[0] / 0x1.4p+12f/*5.120000e+03*/) + 0x1.79ca10c924223p-67f/*1.000000e-20*/));
    for (int step = 0; step < 7; ++step) {
      if (((((int)threadIdx.x) / 96) + step) < 7) {
        *(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int64_t)768)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) - (int64_t)256));
        *(uint4*)(wv + 0) = *(uint4*)(weight + (((step * 768) + (((int)threadIdx.x) * 8)) - 256));
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
          xv[v] = ((bfloat16_t)((((float)yv[v]) * total[0]) * ((float)wv[v])));
        }
        *(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int64_t)768)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) - (int64_t)256)) = *(uint4*)(xv + 0);
      }
    }
  }
}
