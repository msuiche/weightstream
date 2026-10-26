//! Sampling (spec §10, model.py:1288-1295, generate.py:120). Greedy argmax is
//! the default decode path; [`GumbelSampler`] mirrors the reference's
//! Gumbel-max trick with the reference seed constant 33377335 available.
//!
//! Divergence (spec §12.10, documented): torch draws `Exp(1)` from its
//! Philox stream; this engine draws from a splitmix64 stream, so sampled
//! (temperature > 0) sequences match the reference's *distribution* but not
//! its bit stream. Temperature 0 is exact.

use crate::rng::splitmix64;

/// Greedy selection: index of the maximum logit (first wins on ties).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

/// Gumbel-max sampling: `argmax(softmax(logits/T) / Exp(1))`
/// (model.py:1288-1295). Temperature 0 is argmax.
pub struct GumbelSampler {
    state: u64,
}

impl GumbelSampler {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Seeded with the reference constant (`torch.manual_seed(33377335)`).
    pub fn reference() -> Self {
        Self::new(crate::REFERENCE_SEED)
    }

    fn exp1(&mut self) -> f32 {
        // -ln(1-U) with a 53-bit uniform from splitmix64
        let u = (splitmix64(&mut self.state) >> 11) as f64 / (1u64 << 53) as f64;
        (-(1.0 - u).ln()) as f32
    }

    /// One sample from `logits` at `temperature`.
    pub fn sample(&mut self, logits: &[f32], temperature: f32) -> u32 {
        if temperature == 0.0 {
            return argmax(logits);
        }
        let t = temperature.max(1e-5);
        // softmax fp32 over the temperature-scaled logits
        let mut m = f32::NEG_INFINITY;
        for &v in logits {
            if v > m {
                m = v;
            }
        }
        let mut probs = Vec::with_capacity(logits.len());
        let mut sum = 0.0f32;
        for &v in logits {
            let p = ((v / t) - (m / t)).exp();
            probs.push(p);
            sum += p;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        // argmax(probs / Exp(1))
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            let s = p / self.exp1();
            if s > bv {
                bv = s;
                best = i;
            }
        }
        best as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_first_wins() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[3.0, 3.0, 1.0]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY; 3]), 0);
    }

    #[test]
    fn gumbel_temperature_zero_is_argmax() {
        let mut g = GumbelSampler::reference();
        assert_eq!(g.sample(&[1.0, 5.0, 2.0], 0.0), 1);
    }

    #[test]
    fn gumbel_low_temperature_mostly_picks_max() {
        let mut g = GumbelSampler::new(7);
        let logits: Vec<f32> = (0..64).map(|i| if i == 33 { 30.0 } else { 0.0 }).collect();
        let hits = (0..20).filter(|_| g.sample(&logits, 0.01) == 33).count();
        assert!(hits >= 18, "{hits}/20");
    }
}
