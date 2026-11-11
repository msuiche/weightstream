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

/// ceil(log2(x)) for finite positive normal f32, via the exponent field
/// (kernel.py `fast_log2_ceil`). Denormals/zero/negative input is a caller bug.
#[inline]
pub fn log2_ceil(x: f32) -> i32 {
    debug_assert!(x.is_finite() && x > 0.0);
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mant = bits & 0x7F_FFFF;
    if mant != 0 {
        exp + 1
    } else {
        exp
    }
}

/// 2^e for integer e (kernel.py `fast_pow2`). Saturating rather than inf.
#[inline]
pub fn pow2i(e: i32) -> f32 {
    f32::exp2(e as f32)
}

/// UE8M0 activation scale for FP4/FP8 dynamic quant: `2^ceil(log2(amax/max))`
/// (kernel.py `fast_round_scale`). Returns the scale as f32; the UE8M0 byte
/// would be `log2_ceil(amax/max) + 127`.
#[inline]
pub fn ue8m0_round_scale(amax: f32, fp_max: f32) -> f32 {
    pow2i(log2_ceil(amax / fp_max))
}

/// BF16 → f32 (exact).
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 → BF16 bits, round-to-nearest-even.
#[inline]
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + round) >> 16) as u16
}

/// f32 → FP8 E4M3 byte, round-to-nearest-even, saturating at ±448 (the
/// finite-only `e4m3fn` convention; S.1111.111 stays NaN). NaN maps to
/// 0x7F with its sign.
pub fn f32_to_e4m3(v: f32) -> u8 {
    let bits = v.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    if v.is_nan() {
        return sign | 0x7F;
    }
    let a = v.abs();
    if a >= 448.0 {
        return sign | 0x7E;
    }
    // Subnormal region: grid step 1/512 up to 2^-6.
    if a < pow2i(-6) {
        let q = a * 512.0; // exact (power-of-two scale)
        let m = q.round_ties_even() as i32; // 0..=8
        if m == 8 {
            return sign | 0x38; // 2^-6 = E=1 M=0
        }
        return sign | m as u8;
    }
    // Normal region: value = m8 * 2^(e-3), m8 in [8,16).
    let e = ((bits >> 23) & 0xFF) as i32 - 127; // floor(log2 a), a >= 2^-6 is normal
    let q = a * pow2i(3 - e); // a / 2^(e-3), exact
    let m8 = q.round_ties_even() as i32; // 8..=16
    let (exp_field, mant) = if m8 == 16 {
        (e + 8, 0)
    } else {
        (e + 7, m8 - 8)
    };
    debug_assert!((1..=15).contains(&exp_field) && (0..=7).contains(&mant));
    debug_assert!(!(exp_field == 15 && mant == 7)); // NaN pattern unreachable from finite input
    sign | ((exp_field as u8) << 3) | mant as u8
}

/// f32 → FP4 E2M1 nibble code (0..15, sign in bit 3), round-to-nearest-even,
/// saturating at ±6. Mirror of `FP4_TABLE` for encoding.
pub fn f32_to_e2m1(v: f32) -> u8 {
    let bits = v.to_bits();
    let sign = if (bits >> 31) & 1 == 1 { 0x8 } else { 0x0 };
    let a = v.abs();
    let code = if a <= 0.25 {
        0 // tie at 0.25 rounds to even-mantissa 0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    sign | code
}

/// Pack two FP4 codes into one stored byte: low nibble = even element,
/// high nibble = odd (convert.py layout).
#[inline]
pub fn pack_fp4(even: u8, odd: u8) -> u8 {
    (even & 0x0F) | (odd << 4)
}

/// Dequantize a whole FP8 matrix with UE8M0 scales per 32x32 block into a
/// row-major f32 matrix (weight-side inverse of `fp8_gemm`, kernel.py:207).
pub fn dequant_fp8_blockwise(w: &[u8], scales: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols);
    let sb_rows = rows.div_ceil(32);
    let sb_cols = cols.div_ceil(32);
    assert_eq!(scales.len(), sb_rows * sb_cols);
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let s = e8m0_to_f32(scales[(r / 32) * sb_cols + c / 32]);
            out[r * cols + c] = e4m3_to_f32(w[r * cols + c]) * s;
        }
    }
    out
}

/// Dequantize a packed-FP4 matrix `[rows, cols/2]` with one UE8M0 scale per
/// row per 32-element K block (expert-weight layout, convert.py:13-34).
pub fn dequant_fp4_packed(w: &[u8], scales: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(cols % 2, 0);
    assert_eq!(w.len(), rows * cols / 2);
    assert_eq!(scales.len(), rows * cols.div_ceil(32));
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for kb in 0..cols.div_ceil(32) {
            let s = e8m0_to_f32(scales[r * cols.div_ceil(32) + kb]);
            for j in 0..32 {
                let c = kb * 32 + j;
                if c >= cols {
                    break;
                }
                let byte = w[r * (cols / 2) + c / 2];
                let v = if c % 2 == 0 {
                    FP4_TABLE[(byte & 0x0F) as usize]
                } else {
                    FP4_TABLE[(byte >> 4) as usize]
                };
                out[r * cols + c] = v * s;
            }
        }
    }
    out
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

    #[test]
    fn log2_ceil_and_pow2_roundtrip() {
        assert_eq!(log2_ceil(1.0), 0);
        assert_eq!(log2_ceil(0.99), 0);
        assert_eq!(log2_ceil(1.0 / 448.0), -8); // 2^-8.8
        assert_eq!(log2_ceil(1.0 / 512.0), -9);
        assert_eq!(log2_ceil(448.0), 9); // 2^8.8 -> 9
        assert_eq!(pow2i(-9), 1.0 / 512.0);
        assert_eq!(pow2i(9), 512.0);
        // ue8m0 scale: amax 100 with fp8 max 448 -> 2^ceil(log2(0.223)) = 2^-2 = 0.25
        assert_eq!(ue8m0_round_scale(100.0, 448.0), 0.25);
    }

    #[test]
    fn bf16_roundtrip_is_exact() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 3.140625, -123.0, 1.0078125] {
            assert_eq!(bf16_to_f32(f32_to_bf16(v)), v, "value {v}");
        }
        // RNE around 1.0 (bf16 step 2^-7): below midpoint rounds down, above rounds up
        assert_eq!(f32_to_bf16(1.0 + 3.0 / 1024.0), f32_to_bf16(1.0));
        assert_eq!(
            bf16_to_f32(f32_to_bf16(1.0 + 5.0 / 1024.0)),
            1.0 + 8.0 / 1024.0
        );
    }

    #[test]
    fn e4m3_encode_is_round_trip_and_rne() {
        // every representable value round-trips
        for b in 0u8..=0x7F {
            if b == 0x7F {
                continue; // NaN
            }
            let v = e4m3_to_f32(b);
            assert_eq!(f32_to_e4m3(v), b, "byte {b:#x}");
            assert_eq!(f32_to_e4m3(-v), b | 0x80, "byte {b:#x}");
        }
        // saturation
        assert_eq!(f32_to_e4m3(1000.0), 0x7E);
        assert_eq!(f32_to_e4m3(-1000.0), 0xFE);
        assert_eq!(f32_to_e4m3(448.0), 0x7E);
        // grid step at 0.5 (E=6) is 1/16: nearest, then RNE at the midpoint
        assert_eq!(f32_to_e4m3(0.5 + 1.0 / 512.0), 0x30); // nearest 0.5 (E=6,M=0)
        assert_eq!(f32_to_e4m3(0.5 + 1.0 / 32.0), 0x30); // midpoint tie -> even M=0
        assert_eq!(f32_to_e4m3(0.5 + 1.0 / 32.0 + 1e-6), 0x31); // above tie -> 0.5625 (E=6,M=1)
                                                                // subnormal grid step 1/512: tie at 1.5/512 rounds to 2/512 (even mantissa)
        assert_eq!(f32_to_e4m3(1.5 / 512.0), 0x02);
        assert_eq!(f32_to_e4m3(2.5 / 512.0), 0x02); // tie -> 2/512, not 3/512
    }

    #[test]
    fn e2m1_encode_matches_table_and_rne_ties() {
        for code in 0u8..16 {
            let v = FP4_TABLE[code as usize];
            // magnitude round-trips; code 8 is -0.0 and encodes as +0 code 0
            assert_eq!(f32_to_e2m1(v) & 0x7, code & 0x7, "code {code}");
            if v != 0.0 {
                assert_eq!(f32_to_e2m1(v), code, "code {code}");
            }
        }
        // tie-to-even on the e2m1 grid
        assert_eq!(f32_to_e2m1(0.25), 0); // 0 has even mantissa
        assert_eq!(f32_to_e2m1(0.75), 2); // 1.0 has even mantissa
        assert_eq!(f32_to_e2m1(1.25), 2);
        assert_eq!(f32_to_e2m1(1.75), 4); // 2.0 even
        assert_eq!(f32_to_e2m1(2.5), 4);
        assert_eq!(f32_to_e2m1(3.5), 6); // 4.0 even
        assert_eq!(f32_to_e2m1(5.0), 6);
        // saturation at +-6
        assert_eq!(f32_to_e2m1(7.0), 0x7);
        assert_eq!(f32_to_e2m1(-7.0), 0xF);
        assert_eq!(FP4_TABLE[f32_to_e2m1(-0.25) as usize], 0.0); // tie -> 0 not -0.5
    }

    #[test]
    fn blockwise_dequant_matches_elementwise() {
        // 64x64 fp8 with random-ish bytes, 32x32 scales
        let rows = 64;
        let cols = 64;
        let w: Vec<u8> = (0..rows * cols).map(|i| (i * 7 % 251) as u8).collect();
        let s: Vec<u8> = (0..2 * 2).map(|i| 120 + i as u8).collect();
        let dense = dequant_fp8_blockwise(&w, &s, rows, cols);
        for r in [0, 31, 32, 63] {
            for c in [0, 33, 63] {
                let want = e4m3_to_f32(w[r * cols + c]) * e8m0_to_f32(s[(r / 32) * 2 + c / 32]);
                assert_eq!(dense[r * cols + c], want);
            }
        }
        // fp4 packed, 2 rows x 64 cols, one scale per row per 32-K block
        let packed = [0x12u8; 64]; // each byte: even elem code 2 (=1.0), odd elem code 1 (=0.5)
                                   // scales: [row0 blk0, row0 blk1, row1 blk0, row1 blk1] = [1.0, 2.0, 4.0, 8.0]
        let scales = [127u8, 128, 129, 130];
        let dense4 = dequant_fp4_packed(&packed, &scales, 2, 64);
        assert_eq!(dense4[0], 1.0); // code 2 * scale 1.0
        assert_eq!(dense4[1], 0.5); // code 1 * scale 1.0
        assert_eq!(dense4[32], 2.0); // row 0, second block: scale 2.0
        assert_eq!(dense4[64], 4.0); // row 1, first block: scale 4.0
        assert_eq!(dense4[95], 2.0); // row 1 col 31: odd elem 0.5 * scale 4.0
    }
}
