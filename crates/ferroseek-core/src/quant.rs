//! Quantized dtype primitives (host side).
//!
//! On-disk formats in the 1017 checkpoint (verified against shard headers
//! and `inference/convert.py`):
//! - attention projections: FP8 E4M3 weights, UE8M0 scales per 32x32 block
//! - expert weights: FP4 E2M1, two per byte (low nibble = even element),
//!   UE8M0 scale per 1x32 block on the reduce dim
//! - scales themselves: E8M0, a bare power-of-two exponent

/// E2M1 lookup, index = 4-bit pattern (convert.py:FP4_TABLE).
pub const FP4_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Unpack one packed-FP4 byte into (even, odd) elements.
#[inline]
pub fn unpack_fp4(byte: u8) -> (f32, f32) {
    (
        FP4_TABLE[(byte & 0x0F) as usize],
        FP4_TABLE[(byte >> 4) as usize],
    )
}

/// FP8 E4M3 (finite-only variant: no inf, S.1111.111 = NaN), bias 7.
#[inline]
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        // subnormal: 2^-6 * m/8 = m/512
        sign * (mant as f32) * (1.0 / 512.0)
    } else {
        sign * f32::exp2(exp as f32 - 7.0) * (1.0 + mant as f32 / 8.0)
    }
}

/// UE8M0 scale: 2^(e-127). 0xFF is NaN.
#[inline]
pub fn e8m0_to_f32(byte: u8) -> f32 {
    if byte == 0xFF {
        return f32::NAN;
    }
    f32::exp2(byte as f32 - 127.0)
}

/// Dequant one FP4 block of 32 elements from 16 packed bytes.
pub fn dequant_fp4_block(packed: &[u8], scale: f32, out: &mut [f32; 32]) {
    debug_assert_eq!(packed.len(), 16);
    for (i, &byte) in packed.iter().enumerate() {
        let (lo, hi) = unpack_fp4(byte);
        out[2 * i] = lo * scale;
        out[2 * i + 1] = hi * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_nibble_order_matches_convert_py() {
        // low nibble is the even element, high nibble the odd one
        let (lo, hi) = unpack_fp4(0x10);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.5);
        let (lo, hi) = unpack_fp4(0xF7);
        assert_eq!(lo, 6.0); // 0x7 -> 6.0
        assert_eq!(hi, -6.0); // 0xF -> -6.0
    }

    #[test]
    fn e4m3_known_values() {
        assert_eq!(e4m3_to_f32(0x38), 1.0); // S=0 E=0111 M=000
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0); // max finite
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert!(e4m3_to_f32(0x7F).is_nan());
        assert_eq!(e4m3_to_f32(0x01), 0.001953125); // smallest subnormal
    }

    #[test]
    fn e8m0_is_power_of_two() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
        assert_eq!(e8m0_to_f32(137), 1024.0);
        assert!(e8m0_to_f32(0xFF).is_nan());
    }

    #[test]
    fn fp4_block_applies_scale() {
        let packed = [0x11u8; 16]; // every element = 0.5
        let mut out = [0.0f32; 32];
        dequant_fp4_block(&packed, 4.0, &mut out);
        assert!(out.iter().all(|&v| v == 2.0));
    }
}
