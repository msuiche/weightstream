//! Rotary embeddings (spec §3, model.py:368-406).
//!
//! Frequencies live in f32 exactly like the reference's fp32 tables: the
//! base table is `1 / theta_f32^(2i/d)` computed with f32 pow, and a
//! position's angle is `pos_f32 * freq_i`. Adjacent elements form complex
//! pairs rotated by that angle. Two variants exist (model.py:681-688):
//! ratio-0 layers use plain theta 10000; compressed layers use
//! compress_rope_theta + YaRN (original_seq_len 65536, factor 16,
//! beta_fast 32, beta_slow 1, no mscale/attention-scale correction).

use crate::config::RefConfig;

/// Per-pair base frequencies for one rope variant.
pub struct Freqs {
    /// `freq[i]` is the angular frequency of complex pair `i`
    /// (`i = 0..rope_dim/2`).
    pub freq: Vec<f32>,
    pub rope_dim: usize,
}

impl Freqs {
    /// `precompute_freqs_cis` base table (model.py:368-386), position axis
    /// applied at rotation time.
    pub fn new(
        rope_dim: u32,
        theta: f64,
        original_seq_len: u32,
        factor: f64,
        beta_fast: u32,
        beta_slow: u32,
    ) -> Self {
        let d = rope_dim as i32;
        let half = (rope_dim / 2) as usize;
        let theta32 = theta as f32;
        let mut freq: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / theta32.powf((2 * i) as f32 / d as f32))
            .collect();
        if original_seq_len > 0 {
            // the dim whose wavelength completes `rotations` turns over the
            // training context; low/high computed in f64 (python math.log)
            let corrected = |rotations: f64| {
                (rope_dim as f64)
                    * ((original_seq_len as f64) / (rotations * 2.0 * std::f64::consts::PI)).ln()
                    / (2.0 * theta.ln())
            };
            let low = (corrected(beta_fast as f64).floor() as i64).max(0) as i32;
            let high = ((corrected(beta_slow as f64).ceil() as i64).min(d as i64 - 1)) as i32;
            let divisor = ((high - low) as f32).max(1e-3);
            for (i, f) in freq.iter_mut().enumerate() {
                let ramp = (((i as i32 - low) as f32) / divisor).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                *f = (*f / factor as f32) * (1.0 - smooth) + *f * smooth;
            }
        }
        Self {
            freq,
            rope_dim: rope_dim as usize,
        }
    }

    /// The variant a layer uses (model.py:681-688, spec §3 / §12.3).
    pub fn for_layer(cfg: &RefConfig, ratio: u32) -> Self {
        if ratio > 0 {
            Self::new(
                cfg.rope_dim,
                cfg.compress_rope_theta,
                cfg.original_seq_len,
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
            )
        } else {
            // plain rope, YaRN off
            Self::new(cfg.rope_dim, cfg.rope_theta, 0, 1.0, 1, 1)
        }
    }

    /// Rotate the last `rope_dim` elements of `x` at position `pos`
    /// (adjacent pairs as complex numbers, model.py:392-406). `inverse`
    /// conjugates the rotation — how the attention output gets the query's
    /// rotation removed (spec §3, model.py:781-791).
    pub fn apply_tail(&self, x: &mut [f32], pos: u64, inverse: bool) {
        debug_assert!(x.len() >= self.rope_dim);
        let off = x.len() - self.rope_dim;
        for i in 0..self.rope_dim / 2 {
            let ang = pos as f32 * self.freq[i];
            let (c, mut s) = (ang.cos(), ang.sin());
            if inverse {
                s = -s;
            }
            let a = x[off + 2 * i];
            let b = x[off + 2 * i + 1];
            x[off + 2 * i] = a * c - b * s;
            x[off + 2 * i + 1] = a * s + b * c;
        }
    }

    /// `apply_tail` over every head of a `[n_heads][head_dim]` buffer — the
    /// q/o path rotates only the rope tail of each head.
    pub fn apply_heads(&self, x: &mut [f32], head_dim: usize, pos: u64, inverse: bool) {
        for h in 0..x.len() / head_dim {
            self.apply_tail(&mut x[h * head_dim..(h + 1) * head_dim], pos, inverse);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_rope_matches_power_table() {
        let f = Freqs::new(64, 10000.0, 0, 1.0, 1, 1);
        let t = 10000.0f32;
        for i in 0..32 {
            let want = 1.0f32 / t.powf((2 * i) as f32 / 64.0);
            assert!((f.freq[i] - want).abs() <= 1e-6 * want.abs(), "i={i}");
        }
    }

    #[test]
    fn inverse_rotation_roundtrips() {
        let f = Freqs::new(8, 10000.0, 0, 1.0, 1, 1);
        let orig = [1.0f32, -2.0, 0.5, 3.0, -0.25, 4.0, 0.125, -8.0];
        let mut x = orig;
        f.apply_tail(&mut x, 1234, false);
        f.apply_tail(&mut x, 1234, true);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn yarn_blends_high_dims_only() {
        let plain = Freqs::new(64, 160000.0, 0, 16.0, 32, 1);
        let yarn = Freqs::new(64, 160000.0, 65536, 16.0, 32, 1);
        let last = yarn.freq.last().unwrap();
        assert!((last - plain.freq.last().unwrap() / 16.0).abs() < 1e-5 * last.abs());
        assert!((yarn.freq[0] - plain.freq[0]).abs() <= 1e-6 * plain.freq[0].abs());
    }
}
