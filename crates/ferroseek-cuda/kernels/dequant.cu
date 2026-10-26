// FP4 E2M1 dequant: packed weights (2/byte, low nibble = even K) + UE8M0
// scales per 32-element block on K → f32. Mirror of
// ferroseek_core::quant::dequant_fp4_block; the CPU reference gates it.

#include <cstdint>

__device__ __constant__ float FP4_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float e8m0_to_f32(uint8_t e) {
    // 2^(e-127); 0xFF is NaN upstream, never passed here
    return exp2f((float)e - 127.0f);
}

extern "C" __global__ void fp4_dequant(
    const uint8_t* __restrict__ packed,   // n_blocks * 16 bytes
    const uint8_t* __restrict__ scales,   // n_blocks * 1 byte (E8M0)
    float* __restrict__ out,              // n_blocks * 32 floats
    int n_blocks)
{
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= n_blocks) return;
    float scale = e8m0_to_f32(scales[block]);
    const uint8_t* src = packed + (size_t)block * 16;
    float* dst = out + (size_t)block * 32;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint8_t byte = src[i];
        dst[2 * i]     = FP4_TABLE[byte & 0x0F] * scale;
        dst[2 * i + 1] = FP4_TABLE[byte >> 4] * scale;
    }
}
