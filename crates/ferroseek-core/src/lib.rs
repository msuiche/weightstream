//! ferroseek-core: model configuration and memory sizing.
//!
//! Host-side, GPU-free. Everything here must stay testable on any machine;
//! CUDA lives in later crates.

pub mod config;
pub mod kv;

pub use config::ModelConfig;
