// RMSNorm (model-spec §2/§3): y = x * rsqrt(mean(x^2) + 1e-20) * w
// One block per row, fp32 in/out. The weight-less rstd variant feeds
// hc_mixes (one statistic over the flattened 4x5120 stream).

#include <cstdint>

extern "C" __global__ void rmsnorm(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int dim, float eps)
{
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * dim;
    float* outr = out + (size_t)row * dim;

    __shared__ float partial[256];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xr[i];
        acc += v * v;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    float rstd = rsqrtf(partial[0] / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        outr[i] = xr[i] * rstd * w[i];
    }
}

// Just the rstd (hc_mixes statistic, model-spec §2: mean over flattened stream)
extern "C" __global__ void rstd_rows(
    const float* __restrict__ x,
    float* __restrict__ rstd,
    int dim, float eps)
{
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * dim;
    __shared__ float partial[256];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xr[i];
        acc += v * v;
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) rstd[row] = rsqrtf(partial[0] / (float)dim + eps);
}
