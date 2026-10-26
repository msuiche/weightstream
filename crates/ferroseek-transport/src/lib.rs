//! ferroseek-transport: one-shot RoCE collectives for the 4-node DGX Spark
//! ring (ferroseek TP4 transport, docs/transport.md).
//!
//! Layering:
//! - `nodes`  — the recipe's nodes-ring4 JSON -> ring topology (pure, tested
//!   everywhere)
//! - `ctrl`   — management-TCP rendezvous and phase barriers
//! - `verbs`  — hand-written libibverbs FFI + a thin safe wrapper (Linux
//!   only; transcribed from the rdma-core 50 headers on the nodes)
//! - `ring`   — the two-hop fp32 all-reduce and per-rail ping/pong
//! - `bench`  — the 4-rank test runner (`ferroseek-transport bench --ring …`)
//!
//! The ibverbs path is `#[cfg(target_os = "linux")]`; everything else
//! compiles and unit-tests on any host. Tests that need the real ring are
//! additionally gated on `FERROSEEK_RING` at runtime.

pub mod ctrl;
pub mod nodes;
pub mod stats;

#[cfg(target_os = "linux")]
pub mod counter;
#[cfg(target_os = "linux")]
pub mod bench;
#[cfg(target_os = "linux")]
pub mod ring;
#[cfg(target_os = "linux")]
pub mod verbs;
