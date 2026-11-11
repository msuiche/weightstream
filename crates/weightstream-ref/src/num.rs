//! Small f32 linear-algebra and activation primitives. Plain `Vec<f32>`, no
//! external math crates — this is the reference path, so the accumulation
//! order defined here (sequential over K, sequential over output rows) is
//! the normative one GPU kernels are compared against.

/// Dot product, sequential over elements.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Row-major dense mat-vec: `y = W[n, k] * x` (f32; spec's bf16 weights are
/// held dequantized as f32).
pub fn gemv(w: &[f32], n: usize, k: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.len(), n * k);
    debug_assert_eq!(x.len(), k);
    let mut y = vec![0.0f32; n];
    for r in 0..n {
        y[r] = dot(&w[r * k..(r + 1) * k], x);
    }
    y
}

/// RMSNorm with weight (model.py:281-293). fp32 internal math, one variance
/// statistic per row.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), w.len());
    let n = x.len();
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let rstd = rsqrt(sum / n as f32 + eps);
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = w[i] * (x[i] * rstd);
    }
    out
}

/// In-place softmax over a slice, max-subtracted, fp32.
pub fn softmax(x: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > m {
            m = v;
        }
    }
    let mut s = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    for v in x.iter_mut() {
        *v /= s;
    }
}

/// 1/sqrt(x) (torch.rsqrt).
#[inline]
pub fn rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// softplus with torch's 20.0 overflow threshold.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}
