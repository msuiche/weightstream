#![recursion_limit = "512"]

//! weightstream-gpu: the GPU decode engine for DeepSeek-V4.1-Flash-1017 on
//! DGX Spark (GB10). Mirrors `weightstream-ref`'s decode step one-for-one,
//! driving `weightstream-cuda`'s kernels; every deviation from the CPU
//! reference is either a documented reduction-order difference or a host
//! glue fallback.
//!
//! Host-only by default (builds with no CUDA toolchain); `--features cuda`
//! links the driver and enables [`engine::GpuModel`]. The gate is
//! `cargo test -p weightstream-gpu --features cuda` on a DGX Spark: the same
//! `SyntheticWeights` run through `RefModel` (CPU) and `GpuModel` (GPU)
//! must agree within the documented tolerance.

#[cfg(feature = "cuda")]
pub mod dump;
#[cfg(feature = "cuda")]
pub mod engine;
#[cfg(feature = "cuda")]
pub mod launch;
#[cfg(feature = "cuda")]
pub mod stream;
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub mod tp4;

#[cfg(all(test, feature = "cuda"))]
mod tests;

/// True when built with driver support (does not probe hardware).
pub fn built_with_cuda() -> bool {
    cfg!(feature = "cuda")
}
