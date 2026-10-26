//! Synthetic test weights flow through the exact same store interface and
//! dequantization code path, so the gate is honest for tests too.
//!
//! # Known divergences from the reference (by design; GPU kernels are
//! validated against THIS engine with tolerance)
//!
//! - **f32 instead of bf16 intermediates.** Every value the reference holds
//!   in bf16 (stream h, q/kv, attention P, MoE activations) stays f32 here.
//!   Quantization round-trips that are *semantic* (activation snaps, fp8/fp4
//!   weights, scale handling) are reproduced exactly.
//! - **P is not cast to bf16** before the PV product (kernel.py:377 casts
//!   `acc_s` to bf16; spec §3 describes an fp32 online softmax).
//! - **Summation order.** Dot products accumulate sequentially over K; the
//!   online softmax walks indices in chunks of 64 in index order (the
//!   kernel.py:311-403 shape; exact fp32 equality would need the same
//!   intra-block order, spec §12.7); expert outputs accumulate in ascending
//!   expert id. Within-block fp8·fp8 products are exact in f32, so per-block
//!   accumulation is bit-identical to the kernel's scale-corrected
//!   accumulator modulo intra-block order and subnormal underflow.
//! - **Sampling RNG.** `sample()`'s `Exp(1)` draws come from splitmix64, not
//!   torch's Philox stream (spec §12.10): same distribution, not the same
//!   bits. Temperature 0 (greedy) is exact.
//! - **Top-k tie-break** is (value desc, index asc) — deterministic, may
//!   differ from torch's unspecified order (spec §12.6).
//! - **`decode` returns `io::Result`** because checkpoint reads can fail,
//!   and is decode-only: prefill is repeated decode (per-token outputs are
//!   identical to the reference's chunked prefill; see `RefModel::decode`).
//! - **Engram token map** is injectable (`RefModel::set_token_map`);
//!   identity is the default until the tokenizer-derived map (spec §11) is
//!   installed. Hash multipliers replicate numpy PCG64 bit-exactly.
//! - **f32 trig** (`cos`/`sin` for RoPE) may differ from torch's by ≤1 ULP.

#![recursion_limit = "512"]

//! ferroseek-ref: single-rank, pure-f32 CPU decode engine for
//! DeepSeek-V4.1-Flash-1017.
//!
//! This crate is the **numerics gate**: every GPU kernel must reproduce its
//! outputs. Correctness and faithfulness to `docs/model-spec.md` outrank
//! speed. All arithmetic is f32 — the reference's bf16 intermediates are
//! promoted by design, while every semantic quantization round-trip the
//! spec calls out (fp8/fp4 activation snaps, quantized weights, scale
//! handling) is reproduced exactly.
//!
//! Weights are never fully dequantized: quantized tensors stay in their
//! on-disk byte layout behind the [`store::WeightStore`] trait and are read
//! on demand (mmap page cache for real checkpoints); bf16/f32 tensors are
//! converted to f32 and kept in a bounded LRU ([`store::DenseLru`]).
//! Synthetic test weights flow through the exact same store interface and
//! dequantization code path, so the gate is honest for tests too.

pub mod actquant;
pub mod attn;
pub mod config;
pub mod engram;
pub mod hc;
pub mod layout;
pub mod model;
pub mod moe;
pub mod num;
pub mod rng;
pub mod rope;
pub mod sample;
pub mod store;

#[cfg(test)]
mod tests;

pub(crate) mod dump;

pub use config::RefConfig;
pub use model::RefModel;
pub use sample::{argmax, GumbelSampler};
pub use store::{DType, SyntheticWeights, TensorView, WeightStore};

/// Reference sampling seed (`torch.manual_seed(33377335)`, generate.py:120;
/// spec §10). Used by [`GumbelSampler::reference`].
pub const REFERENCE_SEED: u64 = 33377335;
