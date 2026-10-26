//! Env-gated detail dumps for the GPU-vs-CPU gate bisection
//! (`FERROSEEK_DUMP_GPU`). Same file naming as `ferroseek_ref::dump` so the
//! two engines' dumps pair up; no-ops without the env var, so decode is
//! untouched in production.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Current decode position (set by `GpuModel::decode`).
static POS: AtomicU64 = AtomicU64::new(u64::MAX);

static DIR: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();

pub(crate) fn set_pos(p: u64) {
    POS.store(p, Ordering::Relaxed);
}

pub(crate) fn pos() -> u64 {
    POS.load(Ordering::Relaxed)
}

fn write(name: String, bytes: Vec<u8>) {
    let dir = DIR.get_or_init(|| std::env::var_os("FERROSEEK_DUMP_GPU").map(std::path::PathBuf::from));
    if let Some(d) = dir {
        let _ = std::fs::write(d.join(name), bytes);
    }
}

pub(crate) fn f32s(name: String, vals: &[f32]) {
    write(name, vals.iter().flat_map(|v| v.to_le_bytes()).collect());
}

pub(crate) fn i32s(name: String, vals: &[i32]) {
    write(name, vals.iter().flat_map(|v| v.to_le_bytes()).collect());
}

pub(crate) fn u8s(name: String, vals: &[u8]) {
    write(name, vals.to_vec());
}
