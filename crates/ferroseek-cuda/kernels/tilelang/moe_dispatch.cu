#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void moe_dispatch_kernel(int* __restrict__ ExpertCount, int* __restrict__ ExpertOffset, float* __restrict__ LayoutWeights, int* __restrict__ NumTiles, int* __restrict__ Order, int* __restrict__ Psum, int* __restrict__ RoutePos, int* __restrict__ TileExpert, const int64_t* __restrict__ TopkIdx, const float* __restrict__ TopkWeights, int rows, int tiles);
extern "C" __global__ void __launch_bounds__(512, 1) moe_dispatch_kernel(int* __restrict__ ExpertCount, int* __restrict__ ExpertOffset, float* __restrict__ LayoutWeights, int* __restrict__ NumTiles, int* __restrict__ Order, int* __restrict__ Psum, int* __restrict__ RoutePos, int* __restrict__ TileExpert, const int64_t* __restrict__ TopkIdx, const float* __restrict__ TopkWeights, int rows, int tiles) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* scan = ((void*)((char*)buf_dyn_shmem + 0));
  void* warp_count = ((void*)((char*)buf_dyn_shmem + 0));
  int expert = 0;
  uint peers = (uint)0;
  int rank = 0;
  int run = 0;
  int held = 0;
  int count = 0;
  int padded = 0;
  int carry = 0;
  int offset = 0;
  int used_tiles = 0;
  for (int i = 0; i < 4; ++i) {
    if (((i * 16) + (((int)threadIdx.x) >> 5)) < 51) {
      ((int*)warp_count)[((i * 512) + ((int)threadIdx.x))] = 0;
    }
  }
  __syncthreads();
  int condval;
  if ((((int)threadIdx.x) < (rows * 6))) {
    condval = ((int)TopkIdx[min(((int64_t)((int)threadIdx.x)), ((((int64_t)rows) * (int64_t)6) - (int64_t)1))]);
  } else {
    condval = -1;
  }
  expert = condval;
  peers = __match_any_sync((uint)4294967295, expert);
  rank = ((int)__popc((peers & (((uint)1 << ((uint)(((int)threadIdx.x) & 31))) - (uint)1))));
  if ((0 <= expert) & (rank == 0)) {
    ((int*)warp_count)[(((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)96) + ((int64_t)expert))] = ((int)__popc(peers));
  }
  __syncthreads();
  if (((int)threadIdx.x) < 96) {
    run = 0;
    for (int w = 0; w < 16; ++w) {
      held = ((int*)warp_count)[((w * 96) + ((int)threadIdx.x))];
      ((int*)warp_count)[((w * 96) + ((int)threadIdx.x))] = run;
      run = (run + held);
    }
    ((int*)warp_count)[(((int)threadIdx.x) + 1536)] = run;
  }
  __syncthreads();
  if (0 <= expert) {
    rank = (rank + ((int*)warp_count)[(((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)96) + ((int64_t)expert))]);
  }
  int expert_1 = expert;
  int rank_1 = rank;
  int condval_1;
  if ((((int)threadIdx.x) < 96)) {
    condval_1 = ((int*)warp_count)[(((int)threadIdx.x) + 1536)];
  } else {
    condval_1 = 0;
  }
  count = condval_1;
  int count_1 = count;
  int condval_2;
  if ((((int)threadIdx.x) < 96)) {
    condval_2 = (((count_1 + 31) >> 5) * 32);
  } else {
    condval_2 = 0;
  }
  padded = condval_2;
  __syncthreads();
  ((int*)scan)[((int)threadIdx.x)] = padded;
  __syncthreads();
  for (int s = 0; s < 9; ++s) {
    int condval_3;
    if (((1 << s) <= ((int)threadIdx.x))) {
      condval_3 = ((int*)scan)[max((((int)threadIdx.x) - (1 << s)), 0)];
    } else {
      condval_3 = 0;
    }
    carry = condval_3;
    __syncthreads();
    ((int*)scan)[((int)threadIdx.x)] = (((int*)scan)[((int)threadIdx.x)] + carry);
    __syncthreads();
  }
  offset = (((int*)scan)[((int)threadIdx.x)] - padded);
  used_tiles = (((int*)scan)[511] >> 5);
  if (((int)threadIdx.x) < 96) {
    ExpertCount[((int)threadIdx.x)] = count_1;
    ExpertOffset[((int)threadIdx.x)] = offset;
    Psum[((int)threadIdx.x)] = (offset + count_1);
    for (int i_1 = 0; i_1 < (padded - count_1); ++i_1) {
      if (0 <= ((offset + count_1) + i_1)) {
        if (((offset + count_1) + i_1) < (tiles * 32)) {
          Order[((((int64_t)offset) + ((int64_t)count_1)) + ((int64_t)i_1))] = -1;
        }
      }
      if (0 <= ((offset + count_1) + i_1)) {
        if (((offset + count_1) + i_1) < (tiles * 32)) {
          LayoutWeights[((((int64_t)offset) + ((int64_t)count_1)) + ((int64_t)i_1))] = 0x0p+0f/*0.000000e+00*/;
        }
      }
    }
    for (int t = 0; t < (padded >> 5); ++t) {
      if (0 <= ((offset >> 5) + t)) {
        if (((offset >> 5) + t) < tiles) {
          TileExpert[((offset >> 5) + t)] = ((int)threadIdx.x);
        }
      }
    }
  }
  if (((int)threadIdx.x) == 0) {
    NumTiles[0] = used_tiles;
  }
  for (int i_2 = 0; i_2 < ((tiles + 511) >> 9); ++i_2) {
    if ((((i_2 * 512) + ((int)threadIdx.x)) < tiles) & (used_tiles <= ((i_2 * 512) + ((int)threadIdx.x)))) {
      TileExpert[((((int64_t)i_2) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = -1;
    }
  }
  int offset_1 = offset;
  __syncthreads();
  if (((int)threadIdx.x) < 96) {
    ((int*)scan)[((int)threadIdx.x)] = offset_1;
  }
  __syncthreads();
  if (((int)threadIdx.x) < (rows * 6)) {
    if (0 <= expert_1) {
      if (0 <= (((int*)scan)[((int64_t)expert_1)] + rank_1)) {
        if ((((int*)scan)[((int64_t)expert_1)] + rank_1) < (tiles * 32)) {
          Order[(((int64_t)((int*)scan)[((int64_t)expert_1)]) + ((int64_t)rank_1))] = ((int)threadIdx.x);
        }
      }
      if (0 <= (((int*)scan)[((int64_t)expert_1)] + rank_1)) {
        if ((((int*)scan)[((int64_t)expert_1)] + rank_1) < (tiles * 32)) {
          LayoutWeights[(((int64_t)((int*)scan)[((int64_t)expert_1)]) + ((int64_t)rank_1))] = TopkWeights[((int)threadIdx.x)];
        }
      }
      RoutePos[((int)threadIdx.x)] = (((int*)scan)[((int64_t)expert_1)] + rank_1);
    } else {
      RoutePos[((int)threadIdx.x)] = -1;
    }
  }
}

