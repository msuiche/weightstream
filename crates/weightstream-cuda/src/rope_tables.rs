//! RoPE frequency tables, mirroring `precompute_freqs_cis`
//! (model.py:367-386): plain theta, or YaRN for the compress layers
//! (theta 160000, orig 65536, factor 16, beta_fast 32, beta_slow 1,
//! no mscale).

pub struct FreqsCis {
    pub cos: Vec<f32>, // [max_pos, rope_dim/2]
    pub sin: Vec<f32>,
    pub rope_dim: usize,
    pub max_pos: usize,
}

pub struct Yarn {
    pub original_seq_len: usize,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

/// theta=10000, no YaRN — layers 0-1 and MTP (compress_ratio == 0).
pub fn plain(rope_dim: usize, max_pos: usize) -> FreqsCis {
    build(rope_dim, 10_000.0, None, max_pos)
}

/// theta=160000 + YaRN — layers 2-39 (compress_ratio > 0).
pub fn compress(rope_dim: usize, max_pos: usize) -> FreqsCis {
    build(
        rope_dim,
        160_000.0,
        Some(Yarn {
            original_seq_len: 65536,
            factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }),
        max_pos,
    )
}

pub fn build(rope_dim: usize, theta: f32, yarn: Option<Yarn>, max_pos: usize) -> FreqsCis {
    let half = rope_dim / 2;
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / theta.powf(2.0 * i as f32 / rope_dim as f32))
        .collect();
    if let Some(y) = yarn {
        let dim = rope_dim as f32;
        let corrected = |rotations: f32| {
            dim * ((y.original_seq_len as f32) / (rotations * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * theta.ln())
        };
        let low = corrected(y.beta_fast).floor().max(0.0);
        let high = corrected(y.beta_slow).ceil().min(dim - 1.0);
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / y.factor * (1.0 - smooth) + *f * smooth;
        }
    }
    let mut cos = vec![0.0f32; max_pos * half];
    let mut sin = vec![0.0f32; max_pos * half];
    for pos in 0..max_pos {
        for (i, &f) in freqs.iter().enumerate() {
            let angle = pos as f32 * f;
            cos[pos * half + i] = angle.cos();
            sin[pos * half + i] = angle.sin();
        }
    }
    FreqsCis {
        cos,
        sin,
        rope_dim,
        max_pos,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_theta_first_column() {
        // pos 0 is always angle 0 → cos 1, sin 0
        let t = plain(64, 8);
        for i in 0..32 {
            assert_eq!(t.cos[i], 1.0);
            assert_eq!(t.sin[i], 0.0);
        }
        // pos 1, i 0 → freq 1
        assert!((t.cos[32] - 1.0f32.cos()).abs() < 1e-7);
    }

    #[test]
    fn yarn_band_edges() {
        // corrected dims for theta 160000: low = floor(15.42) = 15,
        // high = ceil(24.65) = 25 → i<15 unchanged, i>25 divided by 16.
        // Compare via the pos=1 SIN row: sin(f) ≈ f stays representable
        // where cos(f) ≈ 1 loses the angle to f32 rounding.
        let dim = 64;
        let t = build(dim, 160_000.0, None, 2);
        let y = compress(dim, 2);
        let half = 32;
        for i in 0..half {
            let f_plain = t.sin[half + i];
            let f_yarn = y.sin[half + i];
            if i < 15 {
                assert!((f_plain - f_yarn).abs() < 1e-7, "i={i}");
            } else if i > 25 {
                assert!((f_plain / f_yarn - 16.0).abs() < 1e-3, "i={i}");
            }
        }
    }
}
