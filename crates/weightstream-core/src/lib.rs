//! weightstream-core: model configuration and memory sizing.
//!
//! Host-side, GPU-free. Everything here must stay testable on any machine;
//! CUDA lives in later crates.

pub mod config;
pub mod glp;
pub mod kv;
pub mod loader;
pub mod quant;
pub mod safetensors;
pub mod weights;

pub use config::ModelConfig;
pub use loader::Checkpoint;
