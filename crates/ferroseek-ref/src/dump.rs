//! Env-gated detail dumps for the synthetic bisection harness
//! (`FERROSEEK_DUMP`, see `src/bin/dump-synth.rs`). Raw little-endian files
//! named `d_pos{p}_*{l}` per decode position and layer; no-ops without the
//! env var, so production decode is untouched.

use std::sync::atomic::{AtomicU64, Ordering};

/// Current decode position, for dump sites without a `pos` parameter
/// (decode is single-position-at-a-time; set by `RefModel::decode`).
static POS: AtomicU64 = AtomicU64::new(u64::MAX);

pub(crate) fn set_pos(p: u64) {
    POS.store(p, Ordering::Relaxed);
}

pub(crate) fn pos() -> u64 {
    POS.load(Ordering::Relaxed)
}

pub(crate) fn f32s(name: String, vals: &[f32]) {
    if let Some(dir) = std::env::var_os("FERROSEEK_DUMP") {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(std::path::Path::new(&dir).join(name), bytes);
    }
}

pub(crate) fn i32s(name: String, vals: &[i32]) {
    if let Some(dir) = std::env::var_os("FERROSEEK_DUMP") {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(std::path::Path::new(&dir).join(name), bytes);
    }
}
