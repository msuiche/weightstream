//! Small latency-summary helpers (p50/p99 over microsecond samples).

/// Linear-interpolated percentile over a slice that will be sorted in place.
/// `p` in 0..=100.
pub fn percentile(samples: &mut [f64], p: f64) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    if n == 1 {
        return samples[0];
    }
    let pos = (p / 100.0) * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    samples[lo] * (1.0 - frac) + samples[hi] * frac
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Summary {
    pub n: usize,
    pub min_us: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
}

pub fn summarize(mut samples: Vec<f64>) -> Summary {
    if samples.is_empty() {
        return Summary { n: 0, min_us: f64::NAN, p50_us: f64::NAN, p99_us: f64::NAN, max_us: f64::NAN, mean_us: f64::NAN };
    }
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p50 = percentile(&mut samples, 50.0);
    let p99 = percentile(&mut samples, 99.0);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    Summary { n: samples.len(), min_us: min, p50_us: p50, p99_us: p99, max_us: max, mean_us: mean }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_basic() {
        let mut s: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert_eq!(percentile(&mut s, 0.0), 0.0);
        assert_eq!(percentile(&mut s, 50.0), 49.5);
        assert_eq!(percentile(&mut s, 99.0), 98.01);
        assert_eq!(percentile(&mut s, 100.0), 99.0);
    }

    #[test]
    fn summary_shape() {
        let s = summarize(vec![5.0, 1.0, 3.0]);
        assert_eq!(s.n, 3);
        assert_eq!(s.min_us, 1.0);
        assert_eq!(s.max_us, 5.0);
        assert_eq!(s.mean_us, 3.0);
    }
}
