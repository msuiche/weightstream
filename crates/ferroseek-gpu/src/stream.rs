//! Unified-memory weight streaming (docs/streaming.md).
//!
//! GB10 reads file-backed page-cache pages directly at 163 GB/s when the
//! mapping is registered with `cuMemHostRegister(DEVICEMAP)`; anonymous host
//! memory only reaches 37 GB/s. So the streaming mode keeps the checkpoint's
//! shard mmaps where they are, registers every shard mapping once at model
//! load, and gives the GEMM kernels device pointers straight into the page
//! cache — the 475 GB 1017 snapshot never gets copied or uploaded.
//!
//! cust 0.3 does not wrap `cuMemHostRegister`, but it re-exports the raw
//! driver bindings (`cust::sys`, cust_raw), so this module is a small safe
//! veneer over four calls: register, get-device-pointer, unregister, and
//! error formatting.
//!
//! Enable with `GpuModel::from_checkpoint_streaming`, or the
//! `FERROSEEK_STREAM=1` env var / `--stream` flag at the CLI. Without it,
//! every weight is a device copy (upload mode) — the synthetic gate runs
//! both ways.

use cust::sys::{
    cuGetErrorString, cuMemHostGetDevicePointer_v2, cuMemHostRegister_v2, cuMemHostUnregister,
    cudaError_enum, CU_MEMHOSTREGISTER_DEVICEMAP,
};
use std::io;
use std::sync::Mutex;

const PAGE_ALIGN: usize = 4096;

type CuResult = cudaError_enum;

fn cu_ok(r: CuResult) -> bool {
    // CUDA_SUCCESS = 0
    (r as i32) == 0
}

fn cuda_err(code: CuResult, what: &str) -> io::Error {
    let mut msg: *const std::os::raw::c_char = std::ptr::null();
    let text = unsafe {
        if cu_ok(cuGetErrorString(code, &mut msg)) && !msg.is_null() {
            std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned()
        } else {
            format!("CUresult {}", code as i32)
        }
    };
    io::Error::new(io::ErrorKind::Other, format!("{what}: {text}"))
}

/// One registered host range, unregistered on drop.
struct Registration {
    base: usize,
}

impl Drop for Registration {
    fn drop(&mut self) {
        unsafe {
            // best effort; the process usually exits before this matters
            let _ = cuMemHostUnregister(self.base as *mut _);
        }
    }
}

/// The registered host mappings backing a streaming model. Registration
/// deliberately precedes weight loading (streamed tensors query their
/// device pointers during the load pass) and is separate from the weight
/// store that owns the mapped memory — the GpuModel holds both, the store
/// in a lifetime-carrying box.
pub struct HostArena {
    registrations: Vec<Registration>,
    /// Successfully registered (page-aligned) host ranges with the
    /// host->device address delta for each — streamed tensors must fall
    /// inside one of these; anything else falls back to staging/uploads.
    pub ranges: Vec<(usize, usize, i64)>,
    /// Every mapped shard `(base, end, path)` — registered or not. The
    /// per-use staging path resolves a host pointer back to its backing
    /// file so cold tensors read with one sequential `read()` instead of
    /// page-cache faults (which crawl at ~1-4 MB/s inside a process that
    /// holds tens of GB of pinned device memory on GB10).
    files: Vec<(usize, usize, std::path::PathBuf)>,
    total_bytes: u64,
}

// SAFETY: the registered ranges are read-only checkpoint mmaps (or a
// synthetic store's private allocations); device kernels read them while
// the owning GpuModel is alive, and the host side never mutates them
// concurrently.
unsafe impl Send for HostArena {}

static DEVICE_POINTER_LOCK: Mutex<()> = Mutex::new(());

/// Register `[ptr, ptr+len)` as device-mapped host memory. Page-aligns the
/// range (registration wants page-granular ranges; a shard's whole mmap is
/// already aligned, so this is a no-op there). If the driver rejects a
/// large range, splits it in half recursively down to 64 MiB.
fn register_range(ptr: usize, len: usize, what: &str) -> io::Result<Vec<Registration>> {
    let base = ptr & !(PAGE_ALIGN - 1);
    let end = (ptr + len + PAGE_ALIGN - 1) & !(PAGE_ALIGN - 1);
    let r =
        unsafe { cuMemHostRegister_v2(base as *mut _, end - base, CU_MEMHOSTREGISTER_DEVICEMAP) };
    if cu_ok(r) {
        return Ok(vec![Registration { base }]);
    }
    if len <= 64 << 20 {
        return Err(cuda_err(r, what));
    }
    let half = len / 2;
    let mut out = register_range(ptr, half, what)?;
    out.extend(register_range(ptr + half, len - half, what)?);
    Ok(out)
}

/// Walk every page of `[base, end)` so the process's PTEs exist before the
/// device ever reads the range. The checkpoint's mmaps are lazy and the
/// page cache is warm, so this is a PTE walk, not I/O — but skipping it is
/// fatal at decode time: a registered range whose pages are unmapped in
/// this process makes the GPU demand-fault them through HMM mid-step, and
/// GB10's RM wedges under that storm (NVRM "unhandled interrupt" floods,
/// observed on all four ranks at decode@0 with 64-110 GB registered).
fn prefault_range(base: usize, end: usize) {
    let mut acc = 0u64;
    let mut p = base;
    while p < end {
        // SAFETY: the range is the checkpoint's own mmap; reading one
        // byte per page only populates PTEs.
        acc = acc.wrapping_add(unsafe { *(p as *const u8) } as u64);
        p += PAGE_ALIGN;
    }
    std::hint::black_box(acc);
}

/// Device pointer for a host address inside a registered range. Under UVA
/// (GB10, CUDA 13) this is the identity mapping; queried per tensor so
/// nothing depends on that.
pub fn device_pointer(host: *const u8) -> io::Result<u64> {
    // cuMemHostGetDevicePointer needs a current context; serialize with
    // other context juggling at load time
    let _guard = DEVICE_POINTER_LOCK.lock().unwrap();
    let mut dptr: u64 = 0;
    let r = unsafe { cuMemHostGetDevicePointer_v2(&mut dptr, host as *mut _, 0) };
    if !cu_ok(r) {
        return Err(cuda_err(r, "cuMemHostGetDevicePointer"));
    }
    Ok(dptr)
}

impl HostArena {
    /// Register host regions as device-mapped, oldest-shard-first, until
    /// the driver refuses (observed on GB10 after ~120 GB: "range already
    /// mapped") or the budget is hit. Two caps matter on this box:
    ///
    /// - the driver stops accepting ranges after ~120 GB of registrations;
    /// - every registered byte is PAGE-LOCKED host memory, and GB10's
    ///   device allocations share the same physical RAM. As the sole
    ///   tenant ~100 GB works (measured), but leave real headroom for
    ///   device allocations and the kernel: 96 GB is the default.
    ///   `FERROSEEK_STREAM_BUDGET_GB` overrides (0 disables registration
    ///   entirely — everything then uploads or stages per use).
    pub fn new(regions: &[(usize, usize, std::path::PathBuf)]) -> io::Result<Self> {
        let budget: u64 = std::env::var("FERROSEEK_STREAM_BUDGET_GB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(96);
        let budget = budget * (1 << 30);
        let mut registrations = Vec::new();
        let mut ranges = Vec::new();
        let files = regions
            .iter()
            .map(|&(ptr, len, ref path)| {
                let base = ptr & !(PAGE_ALIGN - 1);
                let end = (ptr + len + PAGE_ALIGN - 1) & !(PAGE_ALIGN - 1);
                (base, end, path.clone())
            })
            .collect();
        let mut total = 0u64;
        let mut failed = 0usize;
        for (i, &(ptr, len, _)) in regions.iter().enumerate() {
            if total >= budget {
                failed += 1;
                continue;
            }
            let base = ptr & !(PAGE_ALIGN - 1);
            let end = (ptr + len + PAGE_ALIGN - 1) & !(PAGE_ALIGN - 1);
            match register_range(ptr, len, &format!("cuMemHostRegister shard {i}")) {
                Ok(regs) => {
                    total += len as u64;
                    // Fault the range's pages CPU-side before any kernel can
                    // read them (see prefault_range): GB10's RM wedges if the
                    // GPU demand-faults a registered range mid-step. With a
                    // warm page cache this is a PTE walk at memory speed.
                    let t = std::time::Instant::now();
                    prefault_range(base, end);
                    eprintln!(
                        "streaming: shard {i} prefaulted {:.1} GB in {:?}",
                        len as f64 / (1u64 << 30) as f64,
                        t.elapsed()
                    );
                    // one driver call per range: UVA maps host->device with
                    // a constant delta inside a registered mapping, so every
                    // tensor pointer derives arithmetically (46k+ tensors per
                    // real model make per-tensor queries the load bottleneck)
                    let probe = base as *const u8;
                    let dptr = device_pointer(probe)?;
                    ranges.push((base, end, dptr as i64 - base as i64));
                    registrations.extend(regs);
                }
                Err(e) => {
                    failed += 1;
                    eprintln!("streaming: shard {i} ({:.1} GB) not registered ({e}); its big tensors stage per use", len as f64 / (1u64 << 30) as f64);
                }
            }
        }
        if failed > 0 {
            eprintln!("streaming: {failed} shard(s) unregistered — routed experts stage per use, engram gathers host-side, small tensors upload");
        }
        Ok(HostArena {
            registrations,
            ranges,
            files,
            total_bytes: total,
        })
    }

    /// Registered host bytes.
    pub fn bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Device pointer for a host address inside a registered range
    /// (None when the address is not registered).
    pub fn device_ptr(&self, ptr: *const u8) -> Option<u64> {
        let p = ptr as usize;
        self.ranges
            .iter()
            .find(|&&(b, e, _)| p >= b && p < e)
            .map(|&(_, _, d)| (p as i64 + d) as u64)
    }

    /// Backing file and byte offset for `[ptr, ptr+len)` in any mapped shard
    /// (the mmaps cover the whole files, so `ptr - base` is the file offset).
    /// None for addresses outside the checkpoint (the synthetic store).
    pub fn file_range(&self, ptr: *const u8, len: usize) -> Option<(std::path::PathBuf, u64)> {
        let p = ptr as usize;
        self.files
            .iter()
            .find(|&&(b, e, _)| p >= b && p + len <= e)
            .map(|&(b, _, ref path)| (path.clone(), (p - b) as u64))
    }
}

/// `true` when the streaming mode was requested via `FERROSEEK_STREAM=1`.
pub fn env_enabled() -> bool {
    std::env::var("FERROSEEK_STREAM")
        .map(|v| v == "1")
        .unwrap_or(false)
}
