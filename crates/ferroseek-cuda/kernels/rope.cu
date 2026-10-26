// RoPE (model-spec §3): complex rotation of adjacent pairs over the last
// rope_dim dims, fp32. cos/sin tables are host-precomputed per layer class
// (plain theta 10000 vs compress theta 160000 + YaRN). The conjugate
// variant removes the query-position rotation on the attention output.

#include <cstdint>

// x: [n_rows, dim], rope applied to x[:, dim-rope_dim : dim]
// tables: [position_index][pair] with position_index = pos_offset + row
extern "C" __global__ void rope(
    float* __restrict__ x,
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t,
    int n_rows, int dim, int rope_dim,
    int pos_offset, int conjugate)
{
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (row >= n_rows || pair >= rope_dim / 2) return;

    int pos = pos_offset + row;
    float c = cos_t[(size_t)pos * (rope_dim / 2) + pair];
    float s = sin_t[(size_t)pos * (rope_dim / 2) + pair];
    if (conjugate) s = -s;

    float* base = x + (size_t)row * dim + (dim - rope_dim) + 2 * pair;
    float x0 = base[0];
    float x1 = base[1];
    base[0] = x0 * c - x1 * s;
    base[1] = x1 * c + x0 * s;
}
