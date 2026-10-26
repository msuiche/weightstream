//! Hyper-connection stream mechanics (spec §2, model.py:940-996,
//! kernel.py:406-474).
//!
//! The residual stream is `hc_mult` parallel copies. Each sublayer derives
//! three coefficient sets from one projection of the flattened stream:
//! `pre` (collapse), `post` (scale the sublayer output), and `comb` (a
//! Sinkhorn-balanced mixing matrix over the copies). A sublayer's mixes are
//! consumed by the *next* sublayer.

use crate::num;

/// The three coefficient sets per token (spec §2).
pub struct HcMixes {
    /// `pre[j]`: weight of copy j when collapsing into the next-but-one
    /// sublayer input (sigmoid + eps).
    pub pre: Vec<f32>,
    /// `post[j]`: output scale for copy j (2·sigmoid).
    pub post: Vec<f32>,
    /// `comb[src * hc + dst]`: doubly-stochastic-ish mixing matrix.
    pub comb: Vec<f32>,
}

/// One projection of the flattened stream into all coefficient sets,
/// including the Sinkhorn balancing (exactly 20 iterations on 1017: one
/// row-softmax+eps / col-norm pair, then 19 row/col pairs; every divide
/// adds eps in the denominator, only the first softmax adds it to the
/// entries).
pub fn hc_mixes(
    h: &[f32],
    hc_fn: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    hc: usize,
    dim: usize,
    norm_eps: f32,
    sinkhorn_iters: usize,
    eps: f32,
) -> HcMixes {
    let mix_hc = (2 + hc) * hc;
    let hc_dim = hc * dim;
    debug_assert_eq!(h.len(), hc_dim);
    debug_assert_eq!(hc_fn.len(), mix_hc * hc_dim);
    debug_assert_eq!(hc_base.len(), mix_hc);
    debug_assert_eq!(hc_scale.len(), 3);

    // ONE rms statistic over the whole flattened hc*d stream (model.py:953-956)
    let mut sumsq = 0.0f32;
    for &v in h {
        sumsq += v * v;
    }
    let rstd = num::rsqrt(sumsq / hc_dim as f32 + norm_eps);

    let mut mixes = vec![0.0f32; mix_hc];
    for j in 0..mix_hc {
        mixes[j] = num::dot(h, &hc_fn[j * hc_dim..(j + 1) * hc_dim]) * rstd;
    }

    let mut pre = vec![0.0f32; hc];
    let mut post = vec![0.0f32; hc];
    for j in 0..hc {
        pre[j] = num::sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + eps;
        post[j] = 2.0 * num::sigmoid(mixes[hc + j] * hc_scale[1] + hc_base[hc + j]);
    }

    // comb_raw[j][k]: j = SOURCE copy, k = DEST copy
    let mut comb = vec![0.0f32; hc * hc];
    for j in 0..hc {
        for k in 0..hc {
            let idx = 2 * hc + j * hc + k;
            comb[j * hc + k] = mixes[idx] * hc_scale[2] + hc_base[idx];
        }
    }

    // Sinkhorn (kernel.py:439-458)
    // comb = softmax_row(comb_raw) + eps
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        for v in row.iter_mut() {
            *v = *v / s + eps;
        }
    }
    // comb = comb / (colsum + eps) — colsum over sources
    col_norm(&mut comb, hc, eps);
    // 19 more row/col pairs
    for _ in 1..sinkhorn_iters {
        row_norm(&mut comb, hc, eps);
        col_norm(&mut comb, hc, eps);
    }

    HcMixes { pre, post, comb }
}

fn row_norm(comb: &mut [f32], hc: usize, eps: f32) {
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let s: f32 = row.iter().sum();
        for v in row.iter_mut() {
            *v /= s + eps;
        }
    }
}

fn col_norm(comb: &mut [f32], hc: usize, eps: f32) {
    for k in 0..hc {
        let mut s = 0.0f32;
        for j in 0..hc {
            s += comb[j * hc + k];
        }
        for j in 0..hc {
            comb[j * hc + k] /= s + eps;
        }
    }
}

/// Collapse the copies into one vector: `y[d] = Σ_j pre[j]·h[j][d]`
/// (model.py:959-962).
pub fn hc_pre(h: &[f32], pre: &[f32], hc: usize, dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; dim];
    for j in 0..hc {
        let pj = pre[j];
        let row = &h[j * dim..(j + 1) * dim];
        for d in 0..dim {
            y[d] += pj * row[d];
        }
    }
    y
}

/// Expand back out, mixing the residual through `comb`:
/// `y[dst][d] = post[dst]·x[d] + Σ_src comb[src][dst]·residual[src][d]`
/// (model.py:964-967).
pub fn hc_post(
    x: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    hc: usize,
    dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; hc * dim];
    for dst in 0..hc {
        let out = &mut y[dst * dim..(dst + 1) * dim];
        let p = post[dst];
        for d in 0..dim {
            out[d] = p * x[d];
        }
        for src in 0..hc {
            let c = comb[src * hc + dst];
            let row = &residual[src * dim..(src + 1) * dim];
            for d in 0..dim {
                out[d] += c * row[d];
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinkhorn_reaches_doubly_stochastic() {
        // adversarial-ish raw comb, hc = 2
        let hc = 2usize;
        let dim = 4usize;
        let mix_hc = (2 + hc) * hc;
        let mut h = vec![0.0f32; hc * dim];
        for (i, v) in h.iter_mut().enumerate() {
            *v = ((i * 37) % 11) as f32 * 0.3 - 1.0;
        }
        let mut hc_fn = vec![0.0f32; mix_hc * hc * dim];
        let mut hc_base = vec![0.0f32; mix_hc];
        for (i, v) in hc_fn.iter_mut().enumerate() {
            *v = ((i * 17) % 7) as f32 * 0.1 - 0.3;
        }
        for (i, v) in hc_base.iter_mut().enumerate() {
            *v = ((i * 13) % 5) as f32 * 0.2 - 0.5;
        }
        let hc_scale = [1.0f32, 0.5, 2.0];
        let m = hc_mixes(&h, &hc_fn, &hc_scale, &hc_base, hc, dim, 1e-20, 20, 1e-6);
        // rows and columns each sum to ~1 (within eps slack)
        for j in 0..hc {
            let s: f32 = (0..hc).map(|k| m.comb[j * hc + k]).sum();
            assert!((s - 1.0).abs() < 1e-3, "row {j} sum {s}");
        }
        for k in 0..hc {
            let s: f32 = (0..hc).map(|j| m.comb[j * hc + k]).sum();
            assert!((s - 1.0).abs() < 1e-3, "col {k} sum {s}");
        }
        // pre in (eps, 1+eps], post in (0, 2)
        for j in 0..hc {
            assert!(m.pre[j] > 0.0 && m.pre[j] < 1.0 + 1e-5);
            assert!(m.post[j] > 0.0 && m.post[j] <= 2.0);
        }
    }

    #[test]
    fn pre_post_shapes() {
        assert_eq!(
            hc_pre(&vec![1.0f32; 8], &[1.0, 0.0, 0.0, 0.0], 4, 2),
            vec![1.0f32, 1.0]
        );
        let comb = vec![0.0f32; 16]; // residual is zero, comb irrelevant
        let y = hc_post(
            &[2.0f32, 2.0],
            &[0.0f32; 8],
            &[1.0, 1.0, 1.0, 1.0],
            &comb,
            4,
            2,
        );
        assert_eq!(y, vec![2.0f32; 8]);
    }
}
