//! CUDA runtime for ferroseek. Host-only by default; the `cuda` feature
//! links the driver (cust) and enables GPU paths — build/test it on a
//! DGX Spark with `--features cuda`.

#[cfg(feature = "cuda")]
pub mod gpu;

pub mod rope_tables;

/// True when built with driver support (does not probe hardware).
pub fn built_with_cuda() -> bool {
    cfg!(feature = "cuda")
}
