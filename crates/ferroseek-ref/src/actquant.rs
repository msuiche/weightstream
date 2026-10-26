//! Activation quantization round-trips (spec §7). These are semantic, not
//! cosmetic: every cached K/V and the attention output pass through them,
//! so the reference must reproduce the exact grid rounding.
//!
//! - FP8 dynamic quant, block 32, UE8M0 power-of-2 scales, amax floor 1e-4,
//!   clamp to ±448 (kernel.py:40-124). `inplace=True` semantics: snap back to
//!   the original dtype (f32 here).
//! - FP4 with E8M0 scales (indexer q/k, block 32, amax floor 6·2^-126) and
//!   with E4M3 scales (compressed KV, block 16, amax floor 6·2^-9, scale
//!   NOT power-of-2) (kernel.py:127-204, spec §12.12).
//! - The GEMM path (`linear()` on an fp8/fp4 weight) quantizes activations
//!   non-inplace; [`fp8_act_values`] returns exactly the dequantized values
//!   those GEMMs consume.

use ferroseek_core::quant::{
    e4m3_to_f32, e8m0_to_f32, f32_to_e2m1, f32_to_e4m3, log2_ceil, pow2i, unpack_fp4,
};

/// Snap `x` in place to the FP8 grid with UE8M0 scales, per `block` elements
/// along the last dim (act_quant, inplace=True; spec §7).
pub fn fp8_snap(x: &mut [f32], block: usize) {
    assert!(
        x.len() % block == 0,
        "fp8_snap: len {} % block {}",
        x.len(),
        block
    );
    for g in x.chunks_exact_mut(block) {
        let mut amax = 0.0f32;
        for &v in g.iter() {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        if amax < 1e-4 {
            amax = 1e-4;
        }
        let s = pow2i(log2_ceil(amax / 448.0));
        for v in g.iter_mut() {
            let q = (*v / s).clamp(-448.0, 448.0);
            *v = e4m3_to_f32(f32_to_e4m3(q)) * s;
        }
    }
}

/// Per-block FP8 activation quantization feeding a quantized GEMM
/// (non-inplace act_quant; spec §7). Returns the dequantized activation —
/// exactly the values the fp8/fp4 gemv consumes: the scales are powers of
/// two, so distributing them into the products is bit-identical to the
/// kernel's per-block scale-corrected accumulation.
pub fn fp8_act_values(x: &[f32], block: usize) -> Vec<f32> {
    assert!(x.len() % block == 0);
    let mut out = x.to_vec();
    fp8_snap(&mut out, block);
    out
}

/// FP4 snap with E8M0 power-of-2 scales (indexer q/k; spec §7).
pub fn fp4_snap_e8m0(x: &mut [f32], block: usize) {
    assert!(x.len() % block == 0);
    for g in x.chunks_exact_mut(block) {
        let mut amax = 0.0f32;
        for &v in g.iter() {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        if amax < 6.0 * pow2i(-126) {
            amax = 6.0 * pow2i(-126);
        }
        let s = pow2i(log2_ceil(amax / 6.0));
        for v in g.iter_mut() {
            let q = (*v / s).clamp(-6.0, 6.0);
            *v = e2m1_val(f32_to_e2m1(q)) * s;
        }
    }
}

/// FP4 snap with E4M3 scales (compressed KV; spec §7, §12.12 — the exact
/// amax floor matters, and the scale is e4m3-rounded, not a power of two).
pub fn fp4_snap_e4m3(x: &mut [f32], block: usize) {
    assert!(x.len() % block == 0);
    for g in x.chunks_exact_mut(block) {
        let mut amax = 0.0f32;
        for &v in g.iter() {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        if amax < 6.0 * pow2i(-9) {
            amax = 6.0 * pow2i(-9);
        }
        // s = e4m3(amax / 6) — NOT power-of-2
        let s = e4m3_to_f32(f32_to_e4m3(amax / 6.0));
        for v in g.iter_mut() {
            let q = (*v / s).clamp(-6.0, 6.0);
            *v = e2m1_val(f32_to_e2m1(q)) * s;
        }
    }
}

#[inline]
fn e2m1_val(code: u8) -> f32 {
    ferroseek_core::quant::FP4_TABLE[(code & 0x0F) as usize]
}

/// FP8 GEMV: per k-block, `Σ (qa[k] * e4m3(w))` scaled by `s_a·s_b` and
/// accumulated in f32 in block order (fp8_gemm semantics, kernel.py:207-307;
/// spec §7). Both scales are powers of two, so this is bit-identical to the
/// kernel's separate scale-corrected accumulator.
pub fn fp8_gemv(w: &[u8], scales: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(cols % 32, 0);
    assert_eq!(w.len(), rows * cols);
    let kb_n = cols / 32;
    assert_eq!(scales.len(), rows.div_ceil(32) * kb_n);
    let qa = fp8_act_values(x, 32);
    let mut y = vec![0.0f32; rows];
    for n in 0..rows {
        let row = &w[n * cols..(n + 1) * cols];
        let srow = (n / 32) * kb_n;
        let mut acc = 0.0f32;
        for kb in 0..kb_n {
            let s_b = e8m0_to_f32(scales[srow + kb]);
            let mut blk = 0.0f32;
            for j in 0..32 {
                let c = kb * 32 + j;
                blk += qa[c] * (e4m3_to_f32(row[c]) * s_b);
            }
            acc += blk;
        }
        y[n] = acc;
    }
    y
}

/// FP4 GEMV against packed E2M1 expert weights with one UE8M0 scale per row
/// per 32-K block (fp4_gemm semantics, kernel.py:477-591; activation block
/// 32 UE8M0, weight block 32 UE8M0, fp32 accumulate). Nibbles: low = even K
/// element, high = odd (convert.py:13-34).
pub fn fp4_gemv(w: &[u8], scales: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(cols % 32, 0);
    assert_eq!(w.len(), rows * cols / 2);
    let kb_n = cols / 32;
    assert_eq!(scales.len(), rows * kb_n);
    let qa = fp8_act_values(x, 32);
    let mut y = vec![0.0f32; rows];
    let half = cols / 2;
    for n in 0..rows {
        let row = &w[n * half..(n + 1) * half];
        let mut acc = 0.0f32;
        for kb in 0..kb_n {
            let s_b = e8m0_to_f32(scales[n * kb_n + kb]);
            let mut blk = 0.0f32;
            for j in 0..32 {
                let c = kb * 32 + j;
                let byte = row[c / 2];
                let v = if c % 2 == 0 {
                    unpack_fp4(byte).0
                } else {
                    unpack_fp4(byte).1
                };
                blk += qa[c] * (v * s_b);
            }
            acc += blk;
        }
        y[n] = acc;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num;

    #[test]
    fn fp8_snap_snaps_to_grid() {
        let mut x = vec![100.0f32, -100.0, 3.0, 0.001];
        x.resize(32, 0.0);
        fp8_snap(&mut x, 32);
        // amax 100 -> s = 0.25; 100/0.25 = 400 ties on the e4m3 grid between
        // 384 (even mantissa) and 416 (odd) -> rounds to 384 -> 96.0
        assert_eq!(x[0], 96.0);
        assert_eq!(x[1], -96.0);
        assert_eq!(x[2], 3.0); // 3/0.25 = 12 exact
                               // tiny value snaps onto the grid: e4m3(0.004) = 0.00390625 -> *0.25
        assert!((x[3] - 0.0009765625).abs() < 1e-9, "{:?}", x[3]);
        assert!(x[4..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fp8_snap_zero_block_uses_floor() {
        let mut x = vec![0.0f32; 32];
        fp8_snap(&mut x, 32);
        assert!(x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fp4_snap_e8m0_matches_grid() {
        // amax 6 -> s = 2^ceil(log2(1)) = 1; values snap to the e2m1 grid
        let mut x = vec![6.0f32, 2.4, 0.3, -5.9];
        x.resize(32, 0.0);
        fp4_snap_e8m0(&mut x, 32);
        assert_eq!(x[0], 6.0);
        assert_eq!(x[1], 2.0); // 2.4 < 2.5 tie -> 2
        assert_eq!(x[2], 0.5); // 0.3 -> 0.5
        assert_eq!(x[3], -6.0);
    }

    #[test]
    fn fp4_e4m3_scale_is_not_power_of_two() {
        // amax 5 -> s = e4m3(5/6) = 0.8125 (NOT a power of two); 5/0.8125 =
        // 6.15 -> clamps to 6 on the e2m1 grid -> 6 * 0.8125 = 4.875
        let mut x = vec![5.0f32, 0.0];
        x.resize(16, 0.0);
        fp4_snap_e4m3(&mut x, 16);
        assert!((x[0] - 4.875).abs() < 1e-6, "{}", x[0]);
        assert!(x[1..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fp8_gemv_matches_dense_dequant() {
        let rows: usize = 32;
        let cols: usize = 32;
        let w: Vec<u8> = (0..rows * cols)
            .map(|i| {
                let b = (i % 200) as u8;
                if b == 0x7F {
                    0x7E
                } else {
                    b
                }
            })
            .collect();
        let scales = vec![127u8; rows.div_ceil(32) * (cols / 32)];
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32) * 0.01 - 0.15) * 448.0 / 200.0)
            .collect();
        let y = fp8_gemv(&w, &scales, rows, cols, &x);
        let qa = fp8_act_values(&x, 32);
        for n in 0..rows {
            let mut want = 0.0f32;
            for c in 0..cols {
                want += qa[c] * e4m3_to_f32(w[n * cols + c]);
            }
            assert!((y[n] - want).abs() < 1e-4, "row {n}: {} vs {}", y[n], want);
        }
    }

    #[test]
    fn fp4_gemv_matches_dense_dequant() {
        let rows = 4;
        let cols = 64;
        let w: Vec<u8> = (0..rows * cols / 2).map(|i| (i % 251) as u8).collect();
        let scales = vec![127u8; rows * (cols / 32)];
        let x: Vec<f32> = (0..cols)
            .map(|i| (i as f32 * 0.02 - 0.6) * 6.0 / 4.0)
            .collect();
        let y = fp4_gemv(&w, &scales, rows, cols, &x);
        let qa = fp8_act_values(&x, 32);
        let dense = ferroseek_core::quant::dequant_fp4_packed(&w, &scales, rows, cols);
        for n in 0..rows {
            let want = num::dot(&dense[n * cols..(n + 1) * cols], &qa);
            assert!((y[n] - want).abs() < 1e-4, "row {n}: {} vs {}", y[n], want);
        }
    }
}
