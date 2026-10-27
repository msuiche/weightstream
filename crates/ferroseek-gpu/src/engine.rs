//! [`GpuModel`]: the device-resident decode engine. Mirrors
//! `ferroseek_ref::RefModel::decode` one sublayer at a time; every kernel
//! launch follows the CPU reference's op order, so only intra-kernel
//! reduction order (block_reduce trees vs the CPU's sequential
//! accumulation) can differ. Host glue (marked TODO(gpu)): two-level
//! top-k selection, MoE route readback, window index lists, engram
//! hashing.
//!
//! # Buffer layout
//!
//! - Weights upload once at build, keeping on-disk quantized formats
//!   (fp8+ue8m0, packed fp4+ue8m0, bf16 gate); `wo_a`, `embed`, `head`,
//!   norms, hc params and compressor weights dequantize/upload as f32.
//! - Per layer one contiguous `kv_all[max_seq/ratio + window, head_dim]`:
//!   rows `[0, window)` are the sliding-window ring, the tail holds this
//!   layer's copy of its publisher's compressed cache (the kv source
//!   fans its completed latent out to every layer in its band, because
//!   sparse_attn gathers from a single kv pointer).
//! - Per kv source: the index-key cache and the CSA open-pair fp32 state.
//! - Per step scratch is allocated once at build and reused.
//!
//! # Divergences from the CPU engine (measured on the mini-model gate)
//!
//! - Reduction order inside kernels: ~1e-6 noise in the f32 chain.
//! - Transcendentals: device `expf`/`rsqrtf` vs Rust libm — ULP-level.
//! - nvcc FMA contraction (rope): 1 ULP.
//! - `weights_proj` combine scale folds into the weights at load instead
//!   of after the dot; selection is order-based, insensitive.
//! - Net effect: logits agree within ~8e-7 on clean steps. The fp8/fp4
//!   activation snaps can flip a code when a value sits within that noise
//!   of a boundary, amplifying to ~1e-3 in the logits for that step —
//!   rare, inherent to cross-libm f32, and an order of magnitude below
//!   structural divergence (the bugs this gate exists to catch measured
//!   1e-1..2e-1). See `tests::GATE_ABS_TOL` for the tolerance ladder.

use crate::launch as k;
use cust::memory::{DeviceBuffer, DevicePointer};
use cust::prelude::*;
use ferroseek_core::loader::Checkpoint;
use ferroseek_core::ModelConfig;
use ferroseek_cuda::gpu::Cuda;
use ferroseek_ref::attn::{select_candidate_blocks, select_topk, window_topk_idxs};
use ferroseek_ref::config::RefConfig;
use ferroseek_ref::engram::{hash_ids, EngramLayout};
use ferroseek_ref::model::{validate_dims, validate_layout};
use ferroseek_ref::store::{bytes_to_f32, CkptStore, DType, SyntheticWeights, WeightStore};
use rayon::prelude::*;
use std::io;
use std::time::Instant;

/// Env-gated coarse decode profiler (`FERROSEEK_PROF=1`): wall time per
/// section of a decode step, aggregated over the run and reported through
/// `GpuModel::prof_stats` at exit. Zero overhead when the env is unset
/// (one bool check per section boundary).
#[derive(Default)]
pub struct Prof {
    pub on: bool,
    pub attn_us: u64,
    pub moe_us: u64,
    pub gate_us: u64, // moe routing readback (the per-layer D2H sync)
    pub engram_us: u64,
    pub head_us: u64,
    pub other_us: u64,
    pub steps: u64,
}

impl Prof {
    pub fn from_env() -> Self {
        Prof {
            on: std::env::var_os("FERROSEEK_PROF").is_some(),
            ..Default::default()
        }
    }
    fn add_us(&mut self, slot: usize, t: Instant) {
        if !self.on {
            return;
        }
        let us = t.elapsed().as_micros() as u64;
        match slot {
            0 => self.attn_us += us,
            1 => self.moe_us += us,
            2 => self.gate_us += us,
            3 => self.engram_us += us,
            4 => self.head_us += us,
            _ => self.other_us += us,
        }
    }
    pub fn summary(&self) -> String {
        if !self.on {
            return String::new();
        }
        let f = |us: u64| us as f64 / 1e3 / self.steps.max(1) as f64;
        format!(
            "prof: steps={} attn={:.1}ms moe={:.1}ms gate={:.1}ms engram={:.1}ms head={:.1}ms other={:.1}ms",
            self.steps,
            f(self.attn_us),
            f(self.moe_us),
            f(self.gate_us),
            f(self.engram_us),
            f(self.head_us),
            f(self.other_us)
        )
    }
}

/// Default decode-position bound; caches are preallocated for it
/// (docs/kernels.md: never reallocated mid-sequence).
pub const DEFAULT_MAX_SEQ_LEN: u64 = 65536;

/// Expert/head sharding plan. TP1 today; the upload loops are structured
/// per-expert/per-tensor so a TP4 slice (96 experts/rank, 16 heads/rank,
/// vocab-sharded embed/head, fp32 combine) slots in without restructuring.
/// The full 1017 checkpoint (~272 GB of experts alone) does not fit one
/// rank's 128 GB unified memory, so real-checkpoint decode requires
/// sharding before `from_checkpoint` can hold it.
#[derive(Clone, Copy, Debug)]
pub struct Sharding {
    pub world_size: u32,
    pub rank: u32,
}

impl Default for Sharding {
    fn default() -> Self {
        Sharding {
            world_size: 1,
            rank: 0,
        }
    }
}

/// Local slice of a sharded tensor (spec §3 TP4 note). `Rows` is a
/// contiguous output-row slice of a row-major matrix; `Cols` is a
/// strided column (K-dim) slice. Scale tiles follow at 1/32 granularity —
/// every TP4 slice lands on 32-row/col boundaries.
#[derive(Clone, Copy, Debug)]
pub enum WShard {
    All,
    Rows { start: usize, len: usize },
    Cols { start: usize, len: usize },
}

impl WShard {
    fn is_all(&self) -> bool {
        matches!(self, WShard::All)
    }
}

/// Sharded weight bytes: a borrow for the unsharded case (zero-copy, TP1
/// path unchanged), an owned gather for slices.
enum Sharded<'b> {
    All(&'b [u8]),
    Owned(Vec<u8>),
}

impl Sharded<'_> {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Sharded::All(b) => b,
            Sharded::Owned(v) => v,
        }
    }
}

/// Slice a row-major fp8 weight's payload per `shard`. `view` gives the
/// logical [rows, cols]; the scale tile is 32 (spec §12).
fn shard_weight<'b>(view: &'b [u8], shape: &[usize], shard: WShard) -> io::Result<Sharded<'b>> {
    shard_weight_impl(view, shape, shard)
}

/// Test seam: the same fn, pub for the byte-identity probe.
pub fn shard_weight_pub<'b>(
    view: &'b [u8],
    shape: &[usize],
    shard: WShard,
) -> io::Result<Sharded<'b>> {
    shard_weight_impl(view, shape, shard)
}

fn shard_weight_impl<'b>(
    view: &'b [u8],
    shape: &[usize],
    shard: WShard,
) -> io::Result<Sharded<'b>> {
    if shard.is_all() {
        return Ok(Sharded::All(view));
    }
    let (rows, cols) = (shape[0], shape[1]);
    if view.len() != rows * cols {
        return ioerr(format!(
            "weight bytes {} do not tile [{rows}, {cols}]",
            view.len()
        ));
    }
    match shard {
        WShard::Rows { start, len } => {
            if start % 32 != 0 || len % 32 != 0 || start + len > rows {
                return ioerr(format!(
                    "row shard [{start}, {start}+{len}) not 32-aligned in [{rows}]"
                ));
            }
            Ok(Sharded::Owned(
                view[start * cols..(start + len) * cols].to_vec(),
            ))
        }
        WShard::Cols { start, len } => {
            if start % 32 != 0 || len % 32 != 0 || start + len > cols {
                return ioerr(format!(
                    "col shard [{start}, {start}+{len}) not 32-aligned in [{cols}]"
                ));
            }
            let mut out = Vec::with_capacity(rows * len);
            for r in 0..rows {
                out.extend_from_slice(&view[r * cols + start..r * cols + start + len]);
            }
            Ok(Sharded::Owned(out))
        }
        WShard::All => unreachable!(),
    }
}

/// Slice the matching ue8m0 scale payload (`[rows/32, cols/32]`) the same
/// way, keeping the shard's weight tiles aligned with their scale tiles.
fn shard_scale<'b>(view: &'b [u8], shape: &[usize], shard: WShard) -> io::Result<Sharded<'b>> {
    shard_scale_impl(view, shape, shard)
}

/// Test seam: the same fn, pub for the byte-identity probe.
pub fn shard_scale_pub<'b>(
    view: &'b [u8],
    shape: &[usize],
    shard: WShard,
) -> io::Result<Sharded<'b>> {
    shard_scale_impl(view, shape, shard)
}

fn shard_scale_impl<'b>(view: &'b [u8], shape: &[usize], shard: WShard) -> io::Result<Sharded<'b>> {
    if shard.is_all() {
        return Ok(Sharded::All(view));
    }
    let (srows, scols) = (shape[0], shape[1]);
    if view.len() != srows * scols {
        return ioerr(format!(
            "scale bytes {} do not tile [{srows}, {scols}]",
            view.len()
        ));
    }
    match shard {
        WShard::Rows { start, len } => {
            let s0 = start / 32;
            let s1 = s0 + len / 32;
            Ok(Sharded::Owned(view[s0 * scols..s1 * scols].to_vec()))
        }
        WShard::Cols { start, len } => {
            let c0 = start / 32;
            let clen = len / 32;
            let mut out = Vec::with_capacity(srows * clen);
            for r in 0..srows {
                out.extend_from_slice(&view[r * scols + c0..r * scols + c0 + clen]);
            }
            Ok(Sharded::Owned(out))
        }
        WShard::All => unreachable!(),
    }
}

/// TP4 collectives (docs/transport.md): the fp32 rank-ordered all-reduce at
/// the spec's reduction points and the rank-ordered logits all-gather.
/// Results must be bit-identical on every rank — the engine relies on that
/// for identical routing, caches and sampling with no extra broadcasts.
/// `None` (TP1) skips every call site, so single-node decode is unchanged.
pub trait Collectives {
    fn allreduce(&mut self, input: &[f32]) -> io::Result<Vec<f32>>;
    fn allgather(&mut self, contribution: &[f32]) -> io::Result<Vec<f32>>;
    /// Latency accounting for run reports ("" when untracked).
    fn stats(&self) -> String {
        String::new()
    }
}

type D<T> = DeviceBuffer<T>;

/// Weight bytes as the GEMM kernels see them: either an owned device
/// buffer (upload mode) or a device pointer into registered host memory
/// (streaming mode, docs/streaming.md). Pointer semantics are identical,
/// so kernels and call sites cannot tell the modes apart.
pub struct W8 {
    ptr: Option<k::U8>,
    _own: Option<D<u8>>,
    /// Unregistered + too big to upload (routed experts, engram tables on
    /// the real checkpoint): per-use copy source. `ptr` is `None` — the
    /// consumer stages a device copy before each use (moe_step) or gathers
    /// rows host-side (engram).
    host: Option<(*const u8, usize)>,
}

/// Fault every page of a mapped tensor into this process before a device
/// copy: cudaMemcpy's GUP path faults file-backed pages synchronously at
/// ~0.2-4 MB/s on GB10 under memory pressure (measured: a 42 MB wq_b copy
/// took 10.9 s cold, 0.5 ms warm), so the load pass and per-use staging
/// pre-touch the range at memory speed instead.
fn prefault(bytes: &[u8]) {
    let mut acc = 0u64;
    for chunk in bytes.chunks(4096) {
        acc = acc.wrapping_add(chunk[0] as u64);
    }
    std::hint::black_box(acc);
}

/// One-shot device copy of a checkpoint payload. Cold file-backed pages
/// fault catastrophically slowly inside a process holding tens of GB of
/// pinned device memory on GB10 (~1-4 MB/s through the copy's GUP path and
/// through plain page faults alike, while a fresh process reads the same
/// ranges at ~130 MB/s) — so when the arena can resolve the payload back
/// to its shard file, read it with one sequential `read()` into anon
/// memory and copy from there (measured: anon-to-device at ~10 GB/s).
fn read_up(c: &Cuda, arena: Option<&crate::stream::HostArena>, bytes: &[u8]) -> io::Result<D<u8>> {
    if let Some((path, off)) = arena.and_then(|a| a.file_range(bytes.as_ptr(), bytes.len())) {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(&path)?;
        f.seek(SeekFrom::Start(off))?;
        let mut buf = vec![0u8; bytes.len()];
        f.read_exact(&mut buf)?;
        return up_u8(c, &buf);
    }
    prefault(bytes);
    up_u8(c, bytes)
}

impl W8 {
    fn device(buf: D<u8>) -> Self {
        Self {
            ptr: Some(buf.as_device_ptr()),
            _own: Some(buf),
            host: None,
        }
    }

    /// Device pointer into a registered host payload (streaming mode).
    /// The bytes stay in the checkpoint's shard mmap (or the synthetic
    /// store's allocation); the arena inside GpuModel keeps them alive.
    /// Unregistered payloads (the driver stops accepting registrations
    /// after ~120 GB) fall back to the raw host address: under UVA the
    /// identity mapping holds, and GB10's HMM faults unregistered host
    /// pages on device access.
    fn streamed(
        bytes: &[u8],
        arena: Option<&crate::stream::HostArena>,
        c: &Cuda,
    ) -> io::Result<Self> {
        if let Some(d) = arena.and_then(|a| a.device_ptr(bytes.as_ptr())) {
            Ok(Self {
                ptr: Some(DevicePointer::from_raw(d)),
                _own: None,
                host: None,
            })
        } else {
            Ok(Self::device(read_up(c, arena, bytes)?))
        }
    }

    /// Like `streamed`, but unregistered payloads become host-backed
    /// (per-use staging) instead of uploading — for tensors too large to
    /// hold on the device (routed experts, the engram tables).
    fn streamed_or_host(
        bytes: &[u8],
        arena: Option<&crate::stream::HostArena>,
        c: &Cuda,
    ) -> io::Result<Self> {
        if let Some(d) = arena.and_then(|a| a.device_ptr(bytes.as_ptr())) {
            Ok(Self {
                ptr: Some(DevicePointer::from_raw(d)),
                _own: None,
                host: None,
            })
        } else {
            Ok(Self::host_backed(bytes))
        }
    }

    /// Unconditionally host-backed, never device-resident: the engram
    /// embed tables (a 98 GB fp8 slab on the real checkpoint) are gathered
    /// row-by-row host-side in every mode, including upload mode.
    fn host_backed(bytes: &[u8]) -> Self {
        Self {
            ptr: None,
            _own: None,
            host: Some((bytes.as_ptr(), bytes.len())),
        }
    }

    fn ptr(&self) -> k::U8 {
        self.ptr
            .expect("host-backed W8 has no device pointer; stage it first")
    }

    /// Device pointer, using the staged copy when host-backed (the copy
    /// borrows `staged` for its lifetime; empty when already resident).
    fn ptr_or<'b>(&'b self, staged: &'b D<u8>) -> k::U8 {
        match self.ptr {
            Some(p) => p,
            None => staged.as_device_ptr(),
        }
    }

    /// Host payload when this tensor is host-backed (None = device/registered).
    fn host(&self) -> Option<(*const u8, usize)> {
        self.host
    }

    /// One-shot device copy of a host-backed payload; empty for tensors
    /// that already live on the device or in a registered mapping.
    fn stage(&self, c: &Cuda, arena: Option<&crate::stream::HostArena>) -> io::Result<D<u8>> {
        match self.host {
            Some((ptr, len)) => {
                // SAFETY: the pointer aliases the weight store's mmap,
                // owned by the model's keepalive; the copy happens while the
                // model lives.
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                read_up(c, arena, slice)
            }
            None => DeviceBuffer::from_slice(&[]).map_err(cuda_err),
        }
    }
}

/// fp8 weight + ue8m0 scales, exactly the on-disk bytes.
struct Q8 {
    w: W8,
    s: W8,
}

/// Packed fp4 expert weight + per-1x32-K ue8m0 scales.
struct Q4 {
    w: W8,
    s: W8,
}

struct GpuCompressor {
    /// compressor wkv dequantized to f32 (bf16 for ratio 1, f32/bf16 for
    /// ratio 2 — the ref's dense path decodes the same way)
    wkv: D<f32>,
    wgate: Option<D<f32>>,
    norm_w: D<f32>,
}

struct GpuIndexer {
    wq_b: Q8,
    /// weights_proj dequantized to f32 with the
    /// `(index_head_dim * index_n_heads)^-0.5` combine factor folded in at
    /// load (documented divergence: scale applied per-element instead of
    /// after the dot; selection is order-based so it is insensitive).
    weights_proj: D<f32>,
}

struct GpuKOwner {
    wk: D<u16>,
    k_norm: D<f32>,
}

struct GpuEngram {
    embed_w: W8,
    embed_s: W8,
    wkv: Q8,
    /// q_weight * k_weight per hc copy (the merge kernel takes one product
    /// row per call).
    w_prod: D<f32>,
}

struct GpuLayer {
    layer: u32,
    ratio: u32,
    kv_src: Option<usize>,
    idx_src: bool,
    owns_k: bool,
    cand_src: bool,
    uses_cand: bool,
    engram: Option<usize>,
    /// 0 = plain theta table, 1 = compress (YaRN) table (spec §3).
    freq: usize,
    attn_norm_w: D<f32>,
    ffn_norm_w: D<f32>,
    hc_attn_fn: D<f32>,
    hc_ffn_fn: D<f32>,
    hc_attn_base: D<f32>,
    hc_ffn_base: D<f32>,
    hc_attn_scale: D<f32>,
    hc_ffn_scale: D<f32>,
    wq_a: Q8,
    q_norm_w: D<f32>,
    wq_b: Q8,
    wkv: Q8,
    kv_norm_w: D<f32>,
    attn_sink: D<f32>,
    /// Host-dequantized f32 (spec §11).
    wo_a: D<u16>,
    wo_b: Q8,
    gate_w: D<u16>,
    gate_bias: D<f32>,
    experts: Vec<[Q4; 3]>,
    /// Grouped-expert slabs (upload mode): contiguous [e_len] tables per
    /// weight kind for device-side dispatch (moe_grouped.cu). None in
    /// streaming mode (per-use staging keeps per-expert buffers).
    w1_all: Option<D<u8>>,
    w1s_all: Option<D<u8>>,
    w2_all: Option<D<u8>>,
    w2s_all: Option<D<u8>>,
    w3_all: Option<D<u8>>,
    w3s_all: Option<D<u8>>,
    shared_experts: [Q8; 3],
    compressor: Option<GpuCompressor>,
    indexer: Option<GpuIndexer>,
    k_owner: Option<GpuKOwner>,
    engram_w: Option<GpuEngram>,
}

/// Per-step scratch, allocated once at build.
struct Scratch {
    h: [D<f32>; 2],
    x: D<f32>,
    qr: D<f32>,
    q: D<f32>,
    kvrow: D<f32>,
    o: D<f32>,
    fl: D<f32>,
    attn_out: D<f32>,
    y_moe: D<f32>,
    gate_buf: D<f32>,
    up_buf: D<f32>,
    // grouped-expert path (moe_grouped.cu): slot-strided buffers; h_e/
    // expert_y/latent/krow serve the host-dispatch path (streaming mode)
    h_e: D<f32>,
    expert_y: D<f32>,
    latent: D<f32>,
    krow: D<f32>,
    local_slots: D<i32>,
    local_experts: D<i32>,
    local_count: D<i32>,
    h_slots: D<f32>,
    q_slots: D<u8>,
    qs_slots: D<u8>,
    qidx: D<f32>,
    widx: D<f32>,
    score: D<f32>,
    eflat: D<f32>,
    ekv: D<f32>,
    hash_ids: D<i64>,
    // quantized activations per feed: x, qr, fl, h_e, engram flat
    act_x: D<u8>,
    act_x_s: D<u8>,
    act_qr: D<u8>,
    act_qr_s: D<u8>,
    act_fl: D<u8>,
    act_fl_s: D<u8>,
    act_h: D<u8>,
    act_h_s: D<u8>,
    act_e: D<u8>,
    act_e_s: D<u8>,
    gate_w_out: D<f32>,
    gate_i_out: D<i32>,
    gate_scores: D<f32>,
    hc_dots: D<f32>,
    win_idxs: D<i32>,
    shared_idxs: D<i32>,
    pre_attn: D<f32>,
    pre_ffn: D<f32>,
    post: D<f32>,
    comb: D<f32>,
    zeros_h: D<f32>,
    ones_hc: D<f32>,
    comb_zero: D<f32>,
    emb_row: D<f32>,
    logits: D<f32>,
    argmax_i: D<i32>,
    argmax_v: D<f32>,
    /// Pinned host staging (docs: pageable copies drain the queue at ~1 ms
    /// a piece on GB10; ~240 per step made a 300 ms token). Sizes:
    /// coll = max(dim, vocab/world) f32 for collective staging,
    /// gate = top_k f32 + top_k i32 for the moe routing readback,
    /// small = 4 KiB for engram hash/upload traffic.
    coll_stage: crate::stream::PinnedBuf,
    gate_stage: crate::stream::PinnedBuf,
    small_stage: crate::stream::PinnedBuf,
}

/// Device caches, preallocated at max_seq_len (docs/kernels.md). Per layer
/// one contiguous `kv_all`: rows `[0, window)` are the ring, the tail holds
/// this layer's copy of its publisher's compressed cache (fan-out copy at
/// publish time — sparse_attn takes a single kv pointer). Per kv source:
/// the index-key cache and the CSA open-pair state.
struct GpuCaches {
    kv_all: Vec<D<f32>>,
    index_k: Vec<D<f32>>,
    comp_kv: Vec<Option<D<f32>>>,
    comp_score: Vec<Option<D<f32>>>,
}

/// Cross-layer shared state (model.py:1169-1183): one slot per published
/// artifact, host-tracked lengths.
#[derive(Clone)]
struct RunState {
    compress_len: Vec<usize>,
    index_k_src: usize,
    candidates: Vec<bool>,
    shared_n_sel: usize,
    /// Debug: when Some, every attention sublayer records (layer, x).
    capture_x: Option<Vec<(u32, Vec<f32>)>>,
}

/// Prefix-cache snapshot (see `GpuModel::snapshot`): device copies of
/// every per-request cache plus the run state and position, restorable
/// with `GpuModel::restore`.
pub struct CacheSnapshot {
    pub pos: u64,
    kv_all: Vec<D<f32>>,
    index_k: Vec<D<f32>>,
    comp_kv: Vec<D<f32>>,
    comp_score: Vec<D<f32>>,
    run: RunState,
    eng_cache: Vec<i64>,
}

/// The GPU decode engine.
///
/// `'a` is the weight-store lifetime in streaming mode: the model borrows
/// the `Checkpoint` whose shard mmaps back the streamed weights. Upload
/// mode is `'static` (every weight is a device copy).
pub struct GpuModel<'a> {
    pub cfg: RefConfig,
    layers: Vec<GpuLayer>,
    caches: GpuCaches,
    engram: EngramLayout,
    eng_cache: Vec<i64>,
    token_map: Option<Vec<i64>>,
    /// [plain, compress] rope tables: [max_pos, rope_dim/2].
    rope_cos: [D<f32>; 2],
    rope_sin: [D<f32>; 2],
    emb: Option<D<f32>>,
    head: D<f32>,
    final_norm: D<f32>,
    scratch: Scratch,
    /// Publisher fan-out: for kv-source ordinal s, the layers attending its
    /// compressed rows (source included).
    bands: Vec<Vec<usize>>,
    run: RunState,
    pos: u64,
    max_seq: u64,
    /// TP hook (see [`Sharding`]); TP1 for now.
    pub sharding: Sharding,
    /// Streaming mode (docs/streaming.md): the registered shard mappings
    /// (`arena`) and the weight store that owns the mapped memory. `None`
    /// = upload mode, where every weight is a device copy.
    arena: Option<crate::stream::HostArena>,
    store_keepalive: Option<Box<dyn WeightStore + 'a>>,
    /// TP4 transport (docs/transport.md). None = TP1: no collective is
    /// issued and every reduction point falls through to the local sum.
    pub comm: Option<Box<dyn Collectives>>,
    /// Decode-section profiler (`FERROSEEK_PROF`, see [`Prof`]).
    pub prof: Prof,
    /// Prefix cache (serving): snapshots of per-request state keyed by
    /// the token ids that produced them, so a request sharing a prefix
    /// prefills only the remainder (see `decode_prefill`). Bounded —
    /// each snapshot is ~82 MB of device memory.
    prefix_cache: Vec<(Vec<u32>, CacheSnapshot)>,
}

const PREFIX_CACHE_MAX: usize = 8;

// ---------------------------------------------------------------------------
// upload helpers

fn ioerr<T>(m: String) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, m))
}

fn cuda_err(e: cust::error::CudaError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
}

fn view_bytes(store: &mut dyn WeightStore, name: &str) -> io::Result<Vec<u8>> {
    Ok(store.get(name)?.bytes.to_vec())
}

fn view_f32(store: &mut dyn WeightStore, name: &str) -> io::Result<Vec<f32>> {
    let v = store.get(name)?;
    bytes_to_f32(&v).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
}

fn up_f32(_c: &Cuda, host: &[f32]) -> io::Result<D<f32>> {
    DeviceBuffer::from_slice(host).map_err(cuda_err)
}

fn up_u16(_c: &Cuda, host: &[u16]) -> io::Result<D<u16>> {
    DeviceBuffer::from_slice(host).map_err(cuda_err)
}

fn up_u8(_c: &Cuda, host: &[u8]) -> io::Result<D<u8>> {
    DeviceBuffer::from_slice(host).map_err(cuda_err)
}

fn up_zero_f32(_c: &Cuda, n: usize) -> io::Result<D<f32>> {
    DeviceBuffer::zeroed(n.max(1)).map_err(cuda_err)
}

fn load_q8(
    store: &mut dyn WeightStore,
    c: &Cuda,
    arena: Option<&crate::stream::HostArena>,
    wname: &str,
    sname: &str,
    shard: WShard,
) -> io::Result<Q8> {
    if let Some(a) = arena {
        let (wv, sv) = store.get2(wname, sname)?;
        let wb = shard_weight(wv.bytes, &wv.shape, shard)?;
        let sb = shard_scale(sv.bytes, &sv.shape, shard)?;
        Ok(Q8 {
            w: W8::streamed(wb.as_slice(), Some(a), c)?,
            s: W8::streamed(sb.as_slice(), Some(a), c)?,
        })
    } else {
        // upload mode (synthetic gate): snapshot both slices to owned bytes
        // first — the store lends one view at a time
        let wb: Vec<u8> = {
            let v = store.get(wname)?;
            shard_weight(v.bytes, &v.shape, shard)?.as_slice().to_vec()
        };
        let sb: Vec<u8> = {
            let v = store.get(sname)?;
            shard_scale(v.bytes, &v.shape, shard)?.as_slice().to_vec()
        };
        Ok(Q8 {
            w: W8::device(up_u8(c, &wb)?),
            s: W8::device(up_u8(c, &sb)?),
        })
    }
}

fn load_q4(
    store: &mut dyn WeightStore,
    c: &Cuda,
    arena: Option<&crate::stream::HostArena>,
    wname: &str,
    sname: &str,
) -> io::Result<Q4> {
    if let Some(a) = arena {
        let (wv, sv) = store.get2(wname, sname)?;
        // routed experts are per-use staged when unregistered: 384 x 3 per
        // layer cannot fit the device, and the raw host mapping faults
        // kernels (HMM is not enabled for kernel args on this stack)
        Ok(Q4 {
            w: W8::streamed_or_host(wv.bytes, Some(a), c)?,
            s: W8::streamed_or_host(sv.bytes, Some(a), c)?,
        })
    } else {
        Ok(Q4 {
            w: W8::device(up_u8(c, &view_bytes(store, wname)?)?),
            s: W8::device(up_u8(c, &view_bytes(store, sname)?)?),
        })
    }
}

/// wo_a: bf16 dense (converted export, spec §11) or fp8+scale (raw HF
/// export) — dequantize host-side to f32 either way, like the ref's dense
/// path. TP4 loads the rank's contiguous o-group block: rows
/// `[row_start, row_start+row_len)` of `[o_groups*o_lora, gdim]`.
fn load_wo_a(
    store: &mut dyn WeightStore,
    c: &Cuda,
    name: &str,
    rows: usize,
    cols: usize,
    row_start: usize,
    row_len: usize,
) -> io::Result<D<u16>> {
    let v = store.get(name)?;
    let dtype = v.dtype;
    let full = v.bytes.to_vec();
    drop(v);
    let esz = dtype_size(dtype);
    if full.len() != rows * cols * esz {
        return ioerr(format!(
            "{name}: wo_a byte length {} does not match [{rows}, {cols}]",
            full.len()
        ));
    }
    if row_start % 32 != 0 || row_len % 32 != 0 || row_start + row_len > rows {
        return ioerr(format!(
            "{name}: wo_a row range [{row_start}, +{row_len}) not 32-aligned in [{rows}]"
        ));
    }
    let full_rows = rows;
    let bytes = full[row_start * cols * esz..(row_start + row_len) * cols * esz].to_vec();
    let rows = row_len;
    match dtype {
        // bf16 stays bf16 on device: bf16_gemv converts per element (exact),
        // and 40 layers of f32 dequantization would cost 5.4 GB device on a
        // 128 GB-unified box that shares RAM with everything else
        DType::Bf16 => {
            let host: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            up_u16(c, &host)
        }
        DType::Fp8E4m3 => {
            let sname = format!("{}.scale", name.trim_end_matches(".weight"));
            let sview = store.get(&sname)?;
            let sbytes_full = sview.bytes.to_vec();
            let sshape = sview.shape.clone();
            drop(sview);
            // scale shape [rows/B, cols/B] with B in {32, 128} (spec §12.9);
            // infer B from the shape rather than assuming 32
            if sshape.len() != 2 {
                return ioerr(format!("{sname}: bad scale shape {sshape:?}"));
            }
            // B from the FULL weight's shape; the shard keeps 32-alignment
            // (checked above) so it is also B-aligned for B in {32, 128}
            let b_rows = full_rows.checked_div(sshape[0]).unwrap_or(0);
            let b_cols = cols.checked_div(sshape[1]).unwrap_or(0);
            if b_rows == 0 || b_rows != b_cols {
                return ioerr(format!(
                    "{sname}: scale shape {sshape:?} does not tile [{full_rows}, {cols}]",
                ));
            }
            let s0 = row_start / b_rows;
            let s1 = s0 + row_len / b_rows;
            if row_start % b_rows != 0 || row_len % b_rows != 0 {
                return ioerr(format!(
                    "{sname}: wo_a shard [{row_start}, +{row_len}) not {b_rows}-aligned"
                ));
            }
            let scols = sshape[1];
            let sbytes = sbytes_full[s0 * scols..s1 * scols].to_vec();
            let host_f32 = dequant_fp8_general(&bytes, &sbytes, rows, cols, b_rows);
            let host: Vec<u16> = host_f32
                .iter()
                .map(|&v| ferroseek_core::quant::f32_to_bf16(v))
                .collect();
            up_u16(c, &host)
        }
        d => ioerr(format!("{name}: unexpected wo_a dtype {d:?}")),
    }
}

/// fp8 [rows, cols] dequant with ue8m0 scales per `b x b` block
/// (b in {32, 128}; generalization of quant::dequant_fp8_blockwise).
fn dequant_fp8_general(w: &[u8], scales: &[u8], rows: usize, cols: usize, b: usize) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols);
    let sb_rows = rows.div_ceil(b);
    let sb_cols = cols.div_ceil(b);
    assert_eq!(
        scales.len(),
        sb_rows * sb_cols,
        "scale bytes do not tile the weight"
    );
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let s = ferroseek_core::quant::e8m0_to_f32(scales[(r / b) * sb_cols + c / b]);
            out[r * cols + c] = ferroseek_core::quant::e4m3_to_f32(w[r * cols + c]) * s;
        }
    }
    out
}

fn dtype_size(d: DType) -> usize {
    match d {
        DType::Bf16 => 2,
        DType::F32 => 4,
        DType::Fp8E4m3 | DType::E8m0 | DType::Fp4Packed | DType::U8 => 1,
    }
}

fn upload_layer(
    cfg: &RefConfig,
    store: &mut dyn WeightStore,
    c: &Cuda,
    l: usize,
    kv_src: Option<usize>,
    arena: Option<&crate::stream::HostArena>,
    sh: &Sharding,
) -> io::Result<GpuLayer> {
    let p = format!("layers.{l}");
    let dim = cfg.dim as usize;
    let hd = cfg.head_dim as usize;
    let inh = cfg.index_n_heads as usize;
    let ihd = cfg.index_head_dim as usize;
    let ratio = cfg.compress_ratios[l];
    // TP4 locals (spec §3): 16 heads, 2 o-groups, 8 index heads, 96 experts
    let world = sh.world_size as usize;
    let rank = sh.rank as usize;
    let heads_l = cfg.n_heads as usize / world;
    let og_l = cfg.o_groups as usize / world;
    let inh_l = inh / world;
    let e_start = rank * cfg.n_routed as usize / world;
    let e_len = cfg.n_routed as usize / world;
    let idx_src = cfg.index_sources.contains(&(l as u32)) && ratio > 0;

    let layer = l as u32;

    // engram weights (q_weight * k_weight precomputed per copy)
    let engram_w = if cfg.engram_layers.iter().any(|&e| e == layer) {
        let qw = view_f32(store, &format!("{p}.engram.q_weight"))?;
        let kw = view_f32(store, &format!("{p}.engram.k_weight"))?;
        if qw.len() != kw.len() {
            return ioerr("engram q/k weight mismatch".into());
        }
        let prod: Vec<f32> = qw.iter().zip(&kw).map(|(&a, &b)| a * b).collect();
        Some(GpuEngram {
            embed_w: W8::host_backed(store.get(&format!("{p}.engram.embed.weight"))?.bytes),
            embed_s: W8::host_backed(store.get(&format!("{p}.engram.embed.scale"))?.bytes),
            wkv: load_q8(
                store,
                c,
                arena,
                &format!("{p}.engram.wkv.weight"),
                &format!("{p}.engram.wkv.scale"),
                WShard::All,
            )?,
            w_prod: up_f32(c, &prod)?,
        })
    } else {
        None
    };

    let compressor = match kv_src {
        Some(_) => {
            let wkv = up_f32(
                c,
                &view_f32(store, &format!("{p}.attn.compressor.wkv.weight"))?,
            )?;
            let wgate = if ratio > 1 {
                Some(up_f32(
                    c,
                    &view_f32(store, &format!("{p}.attn.compressor.wgate.weight"))?,
                )?)
            } else {
                None
            };
            Some(GpuCompressor {
                wkv,
                wgate,
                norm_w: up_f32(
                    c,
                    &view_f32(store, &format!("{p}.attn.compressor.norm.weight"))?,
                )?,
            })
        }
        None => None,
    };

    let indexer = if idx_src {
        Some(GpuIndexer {
            wq_b: load_q8(
                store,
                c,
                arena,
                &format!("{p}.attn.indexer.wq_b.weight"),
                &format!("{p}.attn.indexer.wq_b.scale"),
                WShard::Rows {
                    start: rank * inh_l * ihd,
                    len: inh_l * ihd,
                },
            )?,
            weights_proj: {
                // head shard: rows [rank*inh_l, +inh_l) of [inh, dim],
                // scale factor folded at load (documented divergence)
                let w = view_f32(store, &format!("{p}.attn.indexer.weights_proj.weight"))?;
                let f = (ihd as f32) * (inh as f32);
                let r0 = rank * inh_l * dim;
                let r1 = r0 + inh_l * dim;
                let scaled: Vec<f32> = w[r0..r1].iter().map(|&v| v / f.sqrt()).collect();
                up_f32(c, &scaled)?
            },
        })
    } else {
        None
    };

    let k_owner = match kv_src {
        Some(_) => Some(GpuKOwner {
            wk: up_u16(c, &{
                let v = store.get(&format!("{p}.attn.indexer.wk.weight"))?;
                let n = ihd * hd;
                if v.bytes.len() != n * 2 {
                    return ioerr("indexer wk size mismatch".into());
                }
                v.bytes
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect::<Vec<u16>>()
            })?,
            k_norm: up_f32(
                c,
                &view_f32(store, &format!("{p}.attn.indexer.k_norm.weight"))?,
            )?,
        }),
        None => None,
    };

    let mut experts = Vec::with_capacity(e_len);
    // Fused grouped-expert path (upload mode only): one contiguous
    // [e_len] slab per weight kind so the device-side dispatch kernel can
    // address expert rows by local id (moe_grouped.cu). Streaming mode
    // keeps the per-expert buffers (per-use staging).
    let mut w1_all: Option<D<u8>> = None;
    let mut w1s_all: Option<D<u8>> = None;
    let mut w2_all: Option<D<u8>> = None;
    let mut w2s_all: Option<D<u8>> = None;
    let mut w3_all: Option<D<u8>> = None;
    let mut w3s_all: Option<D<u8>> = None;
    if arena.is_none() && e_len > 0 && sh.world_size == 4 {
        let inter = cfg.inter_dim as usize;
        let dim = cfg.dim as usize;
        let one_w = |store: &mut dyn WeightStore, name: &str| -> io::Result<Vec<u8>> {
            Ok(store.get(name)?.bytes.to_vec())
        };
        let mut b1 = Vec::with_capacity(e_len * inter * dim / 2);
        let mut b1s = Vec::with_capacity(e_len * inter * dim / 32);
        let mut b2 = Vec::with_capacity(e_len * dim * inter / 2);
        let mut b2s = Vec::with_capacity(e_len * dim * inter / 32);
        let mut b3 = Vec::with_capacity(e_len * inter * dim / 2);
        let mut b3s = Vec::with_capacity(e_len * inter * dim / 32);
        for e in e_start..e_start + e_len {
            b1.extend(one_w(store, &format!("{p}.ffn.experts.{e}.w1.weight"))?);
            b1s.extend(one_w(store, &format!("{p}.ffn.experts.{e}.w1.scale"))?);
            b2.extend(one_w(store, &format!("{p}.ffn.experts.{e}.w2.weight"))?);
            b2s.extend(one_w(store, &format!("{p}.ffn.experts.{e}.w2.scale"))?);
            b3.extend(one_w(store, &format!("{p}.ffn.experts.{e}.w3.weight"))?);
            b3s.extend(one_w(store, &format!("{p}.ffn.experts.{e}.w3.scale"))?);
        }
        w1_all = Some(up_u8(c, &b1)?);
        w1s_all = Some(up_u8(c, &b1s)?);
        w2_all = Some(up_u8(c, &b2)?);
        w2s_all = Some(up_u8(c, &b2s)?);
        w3_all = Some(up_u8(c, &b3)?);
        w3s_all = Some(up_u8(c, &b3s)?);
        eprintln!(
            "build: layer {l} experts fused into grouped slabs ({} x 3, {:.1} GB)",
            e_len,
            (b1.len() + b2.len() + b3.len()) as f64 / (1u64 << 30) as f64
        );
    }
    if w1_all.is_none() {
        for e in e_start..e_start + e_len {
            let w1 = load_q4(
                store,
                c,
                arena,
                &format!("{p}.ffn.experts.{e}.w1.weight"),
                &format!("{p}.ffn.experts.{e}.w1.scale"),
            )?;
            let w2 = load_q4(
                store,
                c,
                arena,
                &format!("{p}.ffn.experts.{e}.w2.weight"),
                &format!("{p}.ffn.experts.{e}.w2.scale"),
            )?;
            let w3 = load_q4(
                store,
                c,
                arena,
                &format!("{p}.ffn.experts.{e}.w3.weight"),
                &format!("{p}.ffn.experts.{e}.w3.scale"),
            )?;
            experts.push([w1, w2, w3]);
        }
    }

    // per-field loads (the struct literal binds them in order)
    let attn_norm_w = up_f32(c, &view_f32(store, &format!("{p}.attn_norm.weight"))?)?;
    let ffn_norm_w = up_f32(c, &view_f32(store, &format!("{p}.ffn_norm.weight"))?)?;
    let hc_attn_fn = up_f32(c, &view_f32(store, &format!("{p}.hc_attn_fn"))?)?;
    let hc_ffn_fn = up_f32(c, &view_f32(store, &format!("{p}.hc_ffn_fn"))?)?;
    let hc_attn_base = up_f32(c, &view_f32(store, &format!("{p}.hc_attn_base"))?)?;
    let hc_ffn_base = up_f32(c, &view_f32(store, &format!("{p}.hc_ffn_base"))?)?;
    let hc_attn_scale = up_f32(c, &view_f32(store, &format!("{p}.hc_attn_scale"))?)?;
    let hc_ffn_scale = up_f32(c, &view_f32(store, &format!("{p}.hc_ffn_scale"))?)?;
    let wq_a = load_q8(
        store,
        c,
        arena,
        &format!("{p}.attn.wq_a.weight"),
        &format!("{p}.attn.wq_a.scale"),
        WShard::All,
    )?;
    let q_norm_w = up_f32(c, &view_f32(store, &format!("{p}.attn.q_norm.weight"))?)?;
    let wq_b = load_q8(
        store,
        c,
        arena,
        &format!("{p}.attn.wq_b.weight"),
        &format!("{p}.attn.wq_b.scale"),
        WShard::Rows {
            start: rank * heads_l * hd,
            len: heads_l * hd,
        },
    )?;
    let wkv = load_q8(
        store,
        c,
        arena,
        &format!("{p}.attn.wkv.weight"),
        &format!("{p}.attn.wkv.scale"),
        WShard::All,
    )?;
    let kv_norm_w = up_f32(c, &view_f32(store, &format!("{p}.attn.kv_norm.weight"))?)?;
    let attn_sink = {
        // per-local-head sink slice (spec §3: [16] at TP4)
        let sink = view_f32(store, &format!("{p}.attn.attn_sink"))?;
        let r0 = rank * heads_l;
        up_f32(c, &sink[r0..r0 + heads_l])?
    };
    let wo_a = load_wo_a(
        store,
        c,
        &format!("{p}.attn.wo_a.weight"),
        cfg.o_groups as usize * cfg.o_lora as usize,
        cfg.n_heads as usize * hd / cfg.o_groups as usize,
        rank * og_l * cfg.o_lora as usize,
        og_l * cfg.o_lora as usize,
    )?;
    let wo_b = load_q8(
        store,
        c,
        arena,
        &format!("{p}.attn.wo_b.weight"),
        &format!("{p}.attn.wo_b.scale"),
        // row-parallel: the rank's K-slice of o_groups*o_lora
        WShard::Cols {
            start: rank * og_l * cfg.o_lora as usize,
            len: og_l * cfg.o_lora as usize,
        },
    )?;
    let gate_w = up_u16(c, &{
        let v = store.get(&format!("{p}.ffn.gate.weight"))?;
        let n = cfg.n_routed as usize * dim;
        if v.bytes.len() != n * 2 {
            return ioerr("gate weight size mismatch".into());
        }
        v.bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect::<Vec<u16>>()
    })?;
    let gate_bias = up_f32(c, &view_f32(store, &format!("{p}.ffn.gate.bias"))?)?;
    let shared_experts = [
        load_q8(
            store,
            c,
            arena,
            &format!("{p}.ffn.shared_experts.w1.weight"),
            &format!("{p}.ffn.shared_experts.w1.scale"),
            WShard::All,
        )?,
        load_q8(
            store,
            c,
            arena,
            &format!("{p}.ffn.shared_experts.w2.weight"),
            &format!("{p}.ffn.shared_experts.w2.scale"),
            WShard::All,
        )?,
        load_q8(
            store,
            c,
            arena,
            &format!("{p}.ffn.shared_experts.w3.weight"),
            &format!("{p}.ffn.shared_experts.w3.scale"),
            WShard::All,
        )?,
    ];

    let layer = GpuLayer {
        layer,
        ratio,
        kv_src,
        idx_src,
        owns_k: idx_src && kv_src.is_some(),
        cand_src: cfg.candidate_source >= 0 && (layer as i64) == cfg.candidate_source && idx_src,
        uses_cand: cfg.candidate_source >= 0 && (layer as i64) > cfg.candidate_source && idx_src,
        engram: cfg.engram_layers.iter().position(|&e| e == layer),
        freq: if ratio > 0 { 1 } else { 0 },
        attn_norm_w,
        ffn_norm_w,
        hc_attn_fn,
        hc_ffn_fn,
        hc_attn_base,
        hc_ffn_base,
        hc_attn_scale,
        hc_ffn_scale,
        wq_a,
        q_norm_w,
        wq_b,
        wkv,
        kv_norm_w,
        attn_sink,
        wo_a,
        wo_b,
        gate_w,
        gate_bias,
        experts,
        w1_all,
        w1s_all,
        w2_all,
        w2s_all,
        w3_all,
        w3s_all,
        shared_experts,
        compressor,
        indexer,
        k_owner,
        engram_w,
    };
    Ok(layer)
}

// ---------------------------------------------------------------------------
// decode

impl GpuModel<'static> {
    /// Install the tokenizer-derived compressed id map (engram; spec §11).
    pub fn set_token_map(&mut self, map: Vec<i64>) {
        self.token_map = Some(map);
    }

    /// Next position `decode` expects.
    pub fn next_position(&self) -> u64 {
        self.pos
    }

    /// Compressed rows this layer OWNS (kv sources; spec §5) — `None` for
    /// consumers, mirroring `RefModel::compress_len`.
    pub fn compress_len(&self, layer: u32) -> Option<usize> {
        let li = &self.layers[layer as usize];
        li.kv_src.map(|s| self.run.compress_len[s])
    }

    /// One sliding-window ring row as stored on device (gate debugging).
    pub fn window_row(&self, layer: u32, slot: usize) -> Vec<f32> {
        let hd = self.cfg.head_dim as usize;
        let row = self.caches.kv_all[layer as usize].index(slot * hd..(slot + 1) * hd);
        let mut out = vec![0.0f32; hd];
        row.copy_to(&mut out).expect("window row copy");
        out
    }

    /// One published compressed-KV row as stored on device (gate debugging).
    pub fn compress_row(&self, layer: u32, j: usize) -> Option<Vec<f32>> {
        let hd = self.cfg.head_dim as usize;
        let win = self.cfg.window_size as usize;
        let row =
            self.caches.kv_all[layer as usize].index(win * hd + j * hd..win * hd + (j + 1) * hd);
        let mut out = vec![0.0f32; hd];
        row.copy_to(&mut out).ok()?;
        Some(out)
    }

    /// One owned index-key row as stored on device (gate debugging).
    pub fn index_k_row(&self, layer: u32, j: usize) -> Option<Vec<f32>> {
        let li = &self.layers[layer as usize];
        let ihd = self.cfg.index_head_dim as usize;
        let src = li.kv_src?;
        let row = self.caches.index_k[src].index(j * ihd..(j + 1) * ihd);
        let mut out = vec![0.0f32; ihd];
        row.copy_to(&mut out).ok()?;
        Some(out)
    }

    /// Index-key rows owned by `layer`'s kv source (spec §4).
    pub fn index_k_len(&self, layer: u32) -> Option<usize> {
        let li = &self.layers[layer as usize];
        li.kv_src.map(|s| self.run.compress_len[s])
    }

    /// Engram layout, for tests and tooling.
    pub fn engram_layout(&self) -> &EngramLayout {
        &self.engram
    }

    /// Reset for a fresh request (serving): back to position 0 with the
    /// same weights. Zeroes every per-request cache the decode path
    /// writes — the kv rings, the published compressed rows, the index-key
    /// caches, the compressor open-pair state and its NEG_INF score
    /// sentinels, the engram hash cache, the run state — everything the
    /// build allocates per-model but the step treats as per-request.
    /// KV memory is preallocated at build (docs/kernels.md) and simply
    /// re-zeroed here; positions always decode from 0.
    pub fn reset(&mut self) -> io::Result<()> {
        let guard = Cuda::global();
        let c = guard
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "cuda init"))?;
        let zero = |buf: &mut D<f32>| -> io::Result<()> {
            let _ = buf.set_zero();
            Ok(())
        };
        for buf in self.caches.kv_all.iter_mut() {
            zero(buf)?;
        }
        for buf in self.caches.index_k.iter_mut() {
            zero(buf)?;
        }
        for opt in self.caches.comp_kv.iter_mut() {
            if let Some(buf) = opt {
                zero(buf)?;
            }
        }
        // comp_score's sentinel: every entry NEG_INF (the open-pair search
        // skips scores until the first write, build's exact init)
        for opt in self.caches.comp_score.iter_mut() {
            if let Some(buf) = opt {
                let _ = c.stream.synchronize();
                let n = buf.len();
                let host = vec![f32::NEG_INFINITY; n];
                buf.copy_from(&host).map_err(cuda_err)?;
            }
        }
        self.eng_cache.clear();
        let n_sources = self.run.compress_len.len();
        self.run = RunState {
            compress_len: vec![0; n_sources],
            index_k_src: 0,
            candidates: Vec::new(),
            shared_n_sel: 0,
            capture_x: None,
        };
        self.pos = 0;
        Ok(())
    }

    /// Prefix-cache snapshot: the full per-request decode state at the
    /// current position. `restore` replays it, so a request sharing the
    /// token prefix pays only the tokens past the prefix (a 1000-token
    /// prompt is ~90 s of prefill at 90 ms/token; the snapshot is ~82 MB
    /// of device-to-device copies, well under a millisecond). Same
    /// weights, so the restored state is exactly the state a fresh decode
    /// of that prefix would have produced.
    pub fn snapshot(&self) -> io::Result<CacheSnapshot> {
        let clone_buf = |b: &D<f32>| -> io::Result<D<f32>> {
            let mut out = unsafe { DeviceBuffer::uninitialized(b.len()) }.map_err(cuda_err)?;
            out.copy_from(b).map_err(cuda_err)?;
            Ok(out)
        };
        let mut kv_all = Vec::with_capacity(self.caches.kv_all.len());
        for b in &self.caches.kv_all {
            kv_all.push(clone_buf(b)?);
        }
        let mut index_k = Vec::with_capacity(self.caches.index_k.len());
        for b in &self.caches.index_k {
            index_k.push(clone_buf(b)?);
        }
        let mut comp_kv = Vec::with_capacity(self.caches.comp_kv.len());
        for b in self.caches.comp_kv.iter().flatten() {
            comp_kv.push(clone_buf(b)?);
        }
        let mut comp_score = Vec::with_capacity(self.caches.comp_score.len());
        for b in self.caches.comp_score.iter().flatten() {
            comp_score.push(clone_buf(b)?);
        }
        Ok(CacheSnapshot {
            pos: self.pos,
            kv_all,
            index_k,
            comp_kv,
            comp_score,
            run: self.run.clone(),
            eng_cache: self.eng_cache.clone(),
        })
    }

    /// Replay a `snapshot` taken after the shared prefix's last token.
    /// `decode` continues from `snapshot.pos` — the caller feeds only the
    /// tokens past the prefix. Every rank must restore the same snapshot
    /// (same prefix, same weights, bit-identical state) before stepping.
    pub fn restore(&mut self, snap: &CacheSnapshot) -> io::Result<()> {
        for (dst, src) in self.caches.kv_all.iter_mut().zip(&snap.kv_all) {
            dst.copy_from(src).map_err(cuda_err)?;
        }
        for (dst, src) in self.caches.index_k.iter_mut().zip(&snap.index_k) {
            dst.copy_from(src).map_err(cuda_err)?;
        }
        for (dst, src) in self.caches.comp_kv.iter_mut().flatten().zip(&snap.comp_kv) {
            dst.copy_from(src).map_err(cuda_err)?;
        }
        for (dst, src) in self
            .caches
            .comp_score
            .iter_mut()
            .flatten()
            .zip(&snap.comp_score)
        {
            dst.copy_from(src).map_err(cuda_err)?;
        }
        self.run = snap.run.clone();
        self.eng_cache = snap.eng_cache.clone();
        self.pos = snap.pos;
        Ok(())
    }

    /// Prefill a whole prompt with the prefix cache: find the longest
    /// cached token prefix, restore its snapshot, decode only the tokens
    /// past it, then cache the full prompt's state for the next request.
    /// Returns the logits after the last prompt token (same contract as
    /// `decode` at the final position). All ranks see the same requests,
    /// so the cache evolves identically on every rank and the lockstep
    /// contract holds. A shared 1000-token prefix is ~90 s of prefill
    /// saved for an ~82 MB restore (device copies, well under a ms).
    pub fn decode_prefill(&mut self, ids: &[u32]) -> io::Result<Vec<f32>> {
        // longest strictly-shorter cached prefix of `ids`
        let mut best: Option<usize> = None;
        let mut best_len = 0usize;
        for (i, (key, _)) in self.prefix_cache.iter().enumerate() {
            if key.len() >= ids.len() {
                continue; // equal keys are stale (generation moved past)
            }
            let shared = key.iter().zip(ids).take_while(|(a, b)| a == b).count();
            if shared == key.len() && shared > best_len {
                best_len = shared;
                best = Some(i);
            }
        }
        let mut logits = Vec::new();
        let start = match best {
            Some(i) => {
                // take the snapshot out (avoids the &self/&mut self
                // overlap), restore, put it back
                let (key, snap) = self.prefix_cache.remove(i);
                let pos = snap.pos;
                self.restore(&snap)?;
                self.prefix_cache
                    .insert(i.min(self.prefix_cache.len()), (key, snap));
                pos
            }
            None => {
                self.reset()?;
                0
            }
        };
        for (n, &t) in ids.iter().enumerate().skip(start as usize) {
            logits = self.decode(t, n as u64)?;
        }
        if !ids.is_empty() {
            if self.prefix_cache.len() >= PREFIX_CACHE_MAX {
                self.prefix_cache.remove(0);
            }
            let snap = self.snapshot()?;
            self.prefix_cache.push((ids.to_vec(), snap));
        }
        Ok(logits)
    }

    /// Debug: capture every attention sublayer's input x (gate tooling).
    pub fn debug_capture_x(&mut self, on: bool) {
        self.run.capture_x = if on { Some(Vec::new()) } else { None };
    }

    /// Debug: take the captured (layer, x) pairs.
    pub fn debug_take_captures(&mut self) -> Vec<(u32, Vec<f32>)> {
        self.run.capture_x.take().unwrap_or_default()
    }

    /// TODO(spec-decode): the DSpark drafter stays out of scope — greedy
    /// main-head decoding is the v1 contract, mirroring `RefModel::drafter`.
    pub fn drafter(&mut self) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "TODO(spec-decode): mtp.* drafter is not implemented",
        ))
    }

    /// One decode step (spec §1): raw fp32 logits for `token` at
    /// `position`. Positions must arrive sequentially from 0.
    pub fn decode(&mut self, token: u32, position: u64) -> io::Result<Vec<f32>> {
        if position != self.pos {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "decode expects sequential positions: got {position}, next is {}",
                    self.pos
                ),
            ));
        }
        if position >= self.max_seq {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "position beyond max_seq_len",
            ));
        }
        self.pos += 1;
        if token >= self.cfg.vocab {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "token id out of vocab",
            ));
        }
        let guard = Cuda::global();
        let c = guard
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "cuda init"))?;
        crate::dump::set_pos(position);

        // streaming mode: embedding row read host-side (10 KB), matching the
        // ref engine's dequantization — no device table to gather from
        let emb_row_host: Option<Vec<f32>> = match (self.emb.is_none(), &mut self.store_keepalive) {
            (true, Some(store)) => {
                let v = store.get("embed.weight")?;
                let start = token as usize * self.cfg.dim as usize * 2;
                if start + self.cfg.dim as usize * 2 > v.bytes.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "token out of vocab",
                    ));
                }
                Some(
                    v.bytes[start..start + self.cfg.dim as usize * 2]
                        .chunks_exact(2)
                        .map(|b| {
                            ferroseek_core::quant::bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))
                        })
                        .collect(),
                )
            }
            _ => None,
        };

        // engram hashes, host-side (the same ref code path)
        let hashes = if self.engram.layer_ids.is_empty() {
            Vec::new()
        } else {
            let token_map = self.token_map.clone();
            hash_ids(
                &self.cfg,
                &self.engram,
                &mut self.eng_cache,
                token_map.as_deref(),
                token,
                position,
            )
        };

        let GpuModel {
            cfg,
            layers,
            caches,
            scratch: s,
            rope_cos,
            rope_sin,
            emb,
            head,
            final_norm,
            bands,
            run,
            engram,
            arena,
            sharding,
            comm,
            prof,
            ..
        } = self;
        let arena_opt: Option<&crate::stream::HostArena> = arena.as_ref();
        let sharding = *sharding;
        let mut comm_opt: Option<&mut (dyn Collectives + 'static)> = comm.as_deref_mut();
        let world = sharding.world_size as usize;

        let dim = cfg.dim as usize;
        let hc = cfg.hc_mult as usize;
        let hd = cfg.head_dim as usize;
        let rd = cfg.rope_dim as usize;
        let heads = cfg.n_heads as usize;
        let win = cfg.window_size as usize;
        let eps = cfg.norm_eps;

        // h init: gather the embedding row, broadcast to hc copies via
        // hc_post(zeros residual, ones post) — the ref's 4 identical copies
        {
            // streaming mode: read the embedding row host-side (no 2.6 GB
            // device table on a RAM-shared box); upload mode gathers from
            // the device copy
            if let Some(row) = &emb_row_host {
                unsafe { s.emb_row.index(0..dim).copy_from(row) }.map_err(cuda_err)?;
            } else {
                let idx = [token as i64];
                let d_idx = DeviceBuffer::from_slice(&idx).map_err(cuda_err)?;
                k::gather_f32(
                    c,
                    k::base(emb.as_ref().unwrap()),
                    k::base(&d_idx),
                    k::base(&s.emb_row),
                    1,
                    dim,
                );
            }
            k::hc_post(
                c,
                k::base(&s.emb_row),
                k::base(&s.zeros_h),
                k::base(&s.ones_hc),
                k::base(&s.comb_zero),
                k::base(&s.h[0]),
                1,
                dim,
            );
            // fresh one-hot mix every step (model.py:1162-1166)
            let mut pre = vec![0.0f32; hc];
            pre[0] = 1.0;
            s.pre_ffn.copy_from(&pre).map_err(cuda_err)?;
            // window-only idxs for ratio-0 layers
            let widxs = window_topk_idxs(win, position);
            s.win_idxs.copy_from(&widxs).map_err(cuda_err)?;
        }

        let mut h_cur = 0usize;
        for li in 0..layers.len() {
            let layer = &layers[li];

            // engram merge before the block (model.py:1264-1266)
            let t_eng = Instant::now();
            if let Some(ei) = layer.engram {
                let e = layer.engram_w.as_ref().unwrap();
                let hash = &hashes[ei];
                let ncols = engram.n_hash_cols;
                let ehd = engram.head_dim;
                // the real checkpoint's engram embed table is a 98 GB fp8
                // slab in the last shards — never registered, so gather its
                // n_hash_cols rows host-side with the ref engine's own
                // fp8_row_scaled (bit-identical dequant, model.py:328-365).
                // Registered tables (synthetic gate) keep the device gather.
                let ids: Vec<i64> = hash.iter().map(|&h| h as i64).collect();
                match (e.embed_w.host(), e.embed_s.host()) {
                    (Some((wp, wlen)), Some((sp, slen))) => {
                        // SAFETY: both alias the weight store's mmap, owned
                        // by store_keepalive for the model's lifetime
                        let (wt, st) = unsafe {
                            (
                                std::slice::from_raw_parts(wp, wlen),
                                std::slice::from_raw_parts(sp, slen),
                            )
                        };
                        // Rows are gathered in parallel: each row starts a
                        // distinct page of the 98 GB slab, and with ~80 GB
                        // of device allocations in the same unified RAM the
                        // slab is mostly cold — a serial gather pays one
                        // NVMe fault per row (~650 us each, 31 ms of a
                        // 125 ms step). Chunked rayon (par_chunks) faults
                        // 24 pages concurrently through the page cache.
                        // Row values are written to disjoint slices, so
                        // the fp32 results are identical to the serial
                        // gather.
                        let rows: Vec<usize> =
                            ids.iter().take(ncols).map(|&id| id as usize).collect();
                        let mut flat = vec![0.0f32; rows.len() * ehd];
                        flat.par_chunks_mut(ehd)
                            .zip(rows.par_iter())
                            .for_each(|(chunk, &r)| {
                                if r * ehd + ehd > wt.len() {
                                    // out-of-range rows are a hard error;
                                    // record and fail after the join (rayon
                                    // closures must be closure-safe)
                                    chunk.fill(f32::NAN);
                                    return;
                                }
                                let row = ferroseek_ref::store::fp8_row_scaled(wt, st, r, ehd);
                                chunk.copy_from_slice(&row);
                            });
                        if flat.iter().any(|v| v.is_nan()) {
                            let bad = rows
                                .iter()
                                .find(|&&r| r * ehd + ehd > wt.len())
                                .copied()
                                .unwrap_or(0);
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!("engram hash row {bad} out of table ({})", wt.len() / ehd),
                            ));
                        }
                        // pinned async staging: the two pageable copy_from
                        // calls each drained the queue (engram measured
                        // 32 ms of a 126 ms step). Layout in the 64 KiB
                        // buffer: [ids i64 x nids | flat f32 x ncols*ehd].
                        let nids = ncols.min(ids.len());
                        let ids_bytes = nids * 8;
                        let flat_bytes = flat.len() * 4;
                        let flat_off = 256usize; // past the id slots, 16B-aligned
                        {
                            let stream = c.stream.as_inner();
                            s.small_stage.write_bytes(
                                0,
                                ids[..nids].as_ptr() as *const u8,
                                ids_bytes,
                            );
                            s.small_stage.write_bytes(
                                flat_off,
                                flat.as_ptr() as *const u8,
                                flat_bytes,
                            );
                            s.small_stage.htod(
                                s.hash_ids.as_device_ptr().as_raw() as u64,
                                ids_bytes,
                                stream,
                            )?;
                            s.small_stage.htod_at(
                                s.eflat.as_device_ptr().as_raw() as u64,
                                flat_bytes,
                                flat_off,
                                stream,
                            )?;
                            s.small_stage.sync(stream)?;
                        }
                    }
                    _ => {
                        k::gather_fp8(
                            c,
                            e.embed_w.ptr(),
                            e.embed_s.ptr(),
                            k::base(&s.hash_ids),
                            k::base(&s.eflat),
                            ncols,
                            ehd,
                        );
                    }
                }
                k::emit8(
                    c,
                    k::base(&s.eflat),
                    k::base(&s.act_e),
                    k::base(&s.act_e_s),
                    ncols * ehd,
                );
                k::fp8_gemm(
                    c,
                    k::base(&s.act_e),
                    k::base(&s.act_e_s),
                    e.wkv.w.ptr(),
                    e.wkv.s.ptr(),
                    k::base(&s.ekv),
                    1,
                    dim * (hc + 1),
                    ncols * ehd,
                );
                let inv = (dim as f32).sqrt().recip();
                for j in 0..hc {
                    k::engram_merge(
                        c,
                        k::row(&s.h[h_cur], j * dim, dim),
                        k::row(&s.ekv, j * dim, dim),
                        k::row(&s.ekv, hc * dim, dim),
                        k::row(&e.w_prod, j * dim, dim),
                        dim,
                        inv,
                    );
                }
            }
            prof.add_us(3, t_eng);

            // attention sublayer wiring (spec §2, model.py:970-996)
            let hc_dim = hc * dim;
            let mix_hc = cfg.mix_hc();
            if world > 1 {
                // split hc (TP4): the 24 dot rows run on 24 SMs; identical
                // per-lane order and shuffle tree as hc_block (gate-verified)
                k::hc_mixes(
                    c,
                    k::base(&s.h[h_cur]),
                    k::base(&layer.hc_attn_fn),
                    k::base(&s.hc_dots),
                    1,
                    hc_dim,
                    mix_hc,
                );
                k::hc_finalize(
                    c,
                    k::base(&s.h[h_cur]),
                    k::base(&s.hc_dots),
                    k::base(&layer.hc_attn_base),
                    k::base(&layer.hc_attn_scale),
                    k::base(&s.pre_attn),
                    k::base(&s.post),
                    k::base(&s.comb),
                    1,
                    hc_dim,
                    mix_hc,
                    cfg.hc_eps,
                );
            } else {
                k::hc_block(
                    c,
                    k::base(&s.h[h_cur]),
                    k::base(&layer.hc_attn_fn),
                    k::base(&layer.hc_attn_base),
                    k::base(&layer.hc_attn_scale),
                    k::base(&s.pre_attn),
                    k::base(&s.post),
                    k::base(&s.comb),
                    1,
                    hc_dim,
                    mix_hc,
                    cfg.hc_eps,
                );
            }
            k::hc_pre(
                c,
                k::base(&s.h[h_cur]),
                k::base(&s.pre_ffn),
                k::base(&s.x),
                1,
                dim,
            );
            k::rmsnorm(
                c,
                k::base(&s.x),
                k::base(&layer.attn_norm_w),
                k::base(&s.x),
                dim,
                eps,
            );

            let t_att = Instant::now();
            attention_step(
                c,
                cfg,
                layer,
                s,
                caches,
                run,
                bands,
                &*rope_cos,
                &*rope_sin,
                position,
                sharding,
                comm_opt.as_deref_mut(),
            )?;
            prof.add_us(0, t_att);

            {
                let h_next = 1 - h_cur;
                k::hc_post(
                    c,
                    k::base(&s.attn_out),
                    k::base(&s.h[h_cur]),
                    k::base(&s.post),
                    k::base(&s.comb),
                    k::base(&s.h[h_next]),
                    1,
                    dim,
                );
                h_cur = h_next;
            }

            // ffn sublayer: its pre feeds the next block's attention
            if world > 1 {
                k::hc_mixes(
                    c,
                    k::base(&s.h[h_cur]),
                    k::base(&layer.hc_ffn_fn),
                    k::base(&s.hc_dots),
                    1,
                    hc_dim,
                    mix_hc,
                );
                k::hc_finalize(
                    c,
                    k::base(&s.h[h_cur]),
                    k::base(&s.hc_dots),
                    k::base(&layer.hc_ffn_base),
                    k::base(&layer.hc_ffn_scale),
                    k::base(&s.pre_ffn),
                    k::base(&s.post),
                    k::base(&s.comb),
                    1,
                    hc_dim,
                    mix_hc,
                    cfg.hc_eps,
                );
            } else {
                k::hc_block(
                    c,
                    k::base(&s.h[h_cur]),
                    k::base(&layer.hc_ffn_fn),
                    k::base(&layer.hc_ffn_base),
                    k::base(&layer.hc_ffn_scale),
                    k::base(&s.pre_ffn),
                    k::base(&s.post),
                    k::base(&s.comb),
                    1,
                    hc_dim,
                    mix_hc,
                    cfg.hc_eps,
                );
            }
            k::hc_pre(
                c,
                k::base(&s.h[h_cur]),
                k::base(&s.pre_attn),
                k::base(&s.x),
                1,
                dim,
            );
            k::rmsnorm(
                c,
                k::base(&s.x),
                k::base(&layer.ffn_norm_w),
                k::base(&s.x),
                dim,
                eps,
            );

            let t_moe = Instant::now();
            moe_step(
                c,
                cfg,
                layer,
                s,
                arena_opt,
                sharding,
                comm_opt.as_deref_mut(),
                prof,
            )?;
            prof.add_us(1, t_moe);

            {
                let h_next = 1 - h_cur;
                k::hc_post(
                    c,
                    k::base(&s.y_moe),
                    k::base(&s.h[h_cur]),
                    k::base(&s.post),
                    k::base(&s.comb),
                    k::base(&s.h[h_next]),
                    1,
                    dim,
                );
                h_cur = h_next;
            }
        }

        // collapse with the last layer's ffn pre, final norm, head
        k::hc_pre(
            c,
            k::base(&s.h[h_cur]),
            k::base(&s.pre_ffn),
            k::base(&s.x),
            1,
            dim,
        );
        k::rmsnorm(
            c,
            k::base(&s.x),
            k::base(final_norm),
            k::base(&s.x),
            dim,
            eps,
        );
        // head: the rank's vocab shard produces the first vocab/world
        // logits; TP4 assembles the full vector with the all-gather so
        // every rank samples identically (no token broadcast needed)
        let t_head = Instant::now();
        let vocab_l = cfg.vocab as usize / world;
        k::f32_gemv(
            c,
            k::base(head),
            k::base(&s.x),
            k::base(&s.logits),
            vocab_l,
            dim,
        );
        if let Some(comm) = comm_opt.as_deref_mut() {
            let stream = c.stream.as_inner();
            s.coll_stage.dtoh(
                s.logits.as_device_ptr().as_raw() as u64,
                vocab_l * 4,
                stream,
            )?;
            s.coll_stage.sync(stream)?;
            let full = comm.allgather(&s.coll_stage.as_f32()[..vocab_l])?;
            s.logits.copy_from(&full).map_err(cuda_err)?;
        }
        let mut out = vec![0.0f32; cfg.vocab as usize];
        s.logits.copy_to(&mut out).map_err(cuda_err)?;
        prof.steps += 1;
        prof.add_us(4, t_head);
        Ok(out)
    }

    /// Greedy decode convenience: `decode` + the device argmax kernel.
    pub fn argmax_next(&mut self, token: u32, position: u64) -> io::Result<u32> {
        self.decode(token, position)?;
        self.argmax_current()
    }

    /// Device argmax over the logits of the most recent `decode` step
    /// (greedy path, spec §10; ties to the lowest index).
    pub fn argmax_current(&mut self) -> io::Result<u32> {
        let guard = Cuda::global();
        let c = guard
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "cuda init"))?;
        let s = &self.scratch;
        k::argmax(
            c,
            k::base(&s.logits),
            k::base(&s.argmax_i),
            k::base(&s.argmax_v),
            self.cfg.vocab as usize,
        );
        let mut idx = [0i32; 1];
        s.argmax_i.copy_to(&mut idx).map_err(cuda_err)?;
        Ok(idx[0] as u32)
    }
}

// ---------------------------------------------------------------------------
// build

impl GpuModel<'static> {
    /// Load from a lazily-mmapped checkpoint. Weights upload in their
    /// on-disk quantized formats; wo_a dequantizes on the host; bf16/f32
    /// params upload as f32 (embedding/head) or stay bf16 where the kernel
    /// takes bf16 (gate).
    pub fn from_checkpoint(ckpt: &mut Checkpoint) -> io::Result<GpuModel<'static>> {
        let cfg = RefConfig::from_model_config(&ckpt.config)?;
        let mut store: Box<dyn WeightStore> = Box::new(CkptStore::new(ckpt));
        validate_dims(&cfg)?;
        validate_layout(&cfg, store.as_mut())?;
        Self::build(
            cfg,
            store.as_mut(),
            DEFAULT_MAX_SEQ_LEN,
            None,
            &Sharding::default(),
        )
    }

    /// Load from deterministically generated synthetic weights — the gate:
    /// the same weights go through `RefModel::from_synthetic` on the CPU.
    pub fn from_synthetic(
        cfg: ModelConfig,
        weights: SyntheticWeights,
    ) -> io::Result<GpuModel<'static>> {
        let rcfg = RefConfig::from_model_config(&cfg)?;
        let mut store: Box<dyn WeightStore> = Box::new(weights);
        validate_dims(&rcfg)?;
        Self::build(rcfg, store.as_mut(), 1024, None, &Sharding::default())
    }

    /// Raise the decode-position bound (must be set before the first step).
    pub fn with_max_seq_len(mut self, n: u64) -> Self {
        assert_eq!(self.pos, 0, "caches are preallocated at build time");
        self.max_seq = n;
        self
    }
}

impl<'a> GpuModel<'a> {
    /// Streaming mode (docs/streaming.md): every shard mapping is
    /// registered with `cuMemHostRegister(DEVICEMAP)` and the fp8/fp4
    /// weight kernels read straight out of the page cache — the 475 GB
    /// snapshot is never copied. The model borrows the checkpoint; keep
    /// both alive together.
    pub fn from_checkpoint_streaming(ckpt: &'a mut Checkpoint) -> io::Result<GpuModel<'a>> {
        let cfg = RefConfig::from_model_config(&ckpt.config)?;
        validate_dims(&cfg)?;
        let dir = ckpt.dir.clone();
        let regions: Vec<(usize, usize, std::path::PathBuf)> = ckpt
            .data_regions()?
            .into_iter()
            .map(|(p, l, name)| (p, l, dir.join(name)))
            .collect();
        let n_regions = regions.len();
        // a CUDA context must exist before cuMemHostRegister; register
        // before loading because streamed tensors query device pointers
        // into these mappings during the load pass. The guard is scoped:
        // `build` re-acquires the global lock (a held lock would deadlock).
        {
            let _guard = Cuda::global();
        }
        let arena = crate::stream::HostArena::new(&regions)?;
        eprintln!(
            "streaming: {} shard mappings registered, {:.1} GB (hot tensors stay on device)",
            n_regions,
            arena.bytes() as f64 / (1u64 << 30) as f64
        );
        let mut store: Box<dyn WeightStore + 'a> = Box::new(CkptStore::new(ckpt));
        let t_v = std::time::Instant::now();
        validate_layout(&cfg, store.as_mut())?;
        eprintln!("streaming: layout validated in {:?}", t_v.elapsed());
        let t_b = std::time::Instant::now();
        let mut model = Self::build(
            cfg,
            store.as_mut(),
            DEFAULT_MAX_SEQ_LEN,
            Some(&arena),
            &Sharding::default(),
        )?;
        eprintln!("streaming: weight pass in {:?}", t_b.elapsed());
        // the borrow above ended when build returned, so the arena and
        // store can move into the model; both keep the shard mmaps alive
        model.arena = Some(arena);
        model.store_keepalive = Some(store);
        Ok(model)
    }

    /// TP4 (spec §3 note, docs/transport.md): rank `sharding.rank` of a
    /// 4-rank ring loads its slice — 16 attention heads, 2 o-groups, 8
    /// index heads, 96 routed experts, one vocab shard of the head — and
    /// the `comm` transport carries the three fp32 all-reduces per layer
    /// plus the logits all-gather. The residual stream, caches, gate,
    /// shared expert and engram stay replicated: every rank runs the
    /// identical step and samples the identical token.
    pub fn from_checkpoint_streaming_tp4(
        ckpt: &'a mut Checkpoint,
        sharding: Sharding,
        comm: Box<dyn Collectives>,
    ) -> io::Result<GpuModel<'a>> {
        if sharding.world_size != 4 {
            return ioerr("TP4 needs world_size 4".into());
        }
        let cfg = RefConfig::from_model_config(&ckpt.config)?;
        validate_dims(&cfg)?;
        let dir = ckpt.dir.clone();
        let regions: Vec<(usize, usize, std::path::PathBuf)> = ckpt
            .data_regions()?
            .into_iter()
            .map(|(p, l, name)| (p, l, dir.join(name)))
            .collect();
        let n_regions = regions.len();
        {
            let _guard = Cuda::global();
        }
        let arena = crate::stream::HostArena::new(&regions)?;
        eprintln!(
            "tp4 rank {}: {} shard mappings, {:.1} GB registered ({}/{} heads, {}/{} experts, vocab shard {}/{})",
            sharding.rank,
            n_regions,
            arena.bytes() as f64 / (1u64 << 30) as f64,
            cfg.n_heads / sharding.world_size,
            cfg.n_heads,
            cfg.n_routed / sharding.world_size,
            cfg.n_routed,
            cfg.vocab / sharding.world_size,
            cfg.vocab
        );
        let mut store: Box<dyn WeightStore + 'a> = Box::new(CkptStore::new(ckpt));
        validate_layout(&cfg, store.as_mut())?;
        let t_b = std::time::Instant::now();
        let mut model = Self::build(
            cfg,
            store.as_mut(),
            DEFAULT_MAX_SEQ_LEN,
            Some(&arena),
            &sharding,
        )?;
        eprintln!(
            "tp4 rank {}: weight pass in {:?}",
            sharding.rank,
            t_b.elapsed()
        );
        model.arena = Some(arena);
        model.store_keepalive = Some(store);
        model.sharding = sharding;
        model.comm = Some(comm);
        Ok(model)
    }

    /// TP4 upload mode: the rank's whole slice lives on the device as
    /// ordinary cudaMalloc memory (GB10: device memory IS unified DRAM,
    /// 226 GB/s measured — docs/streaming.md). The routed-expert slice is
    /// ~69 GB fp4/rank, projections/head/shared expert a few more GB, the
    /// engram tables (189 GB) stay host-backed and gather rows host-side,
    /// so the 475 GB checkpoint fits 4 x 128 GB with zero registration and
    /// zero per-use staging. This is the production path: the DEVICEMAP
    /// registration ceiling (~64-70 GB on driver 580.178.04, above which
    /// the RM wedges in an unhandled-interrupt storm) can never cover the
    /// slice, and per-use staging leaves decode 50x off the streaming
    /// floor.
    pub fn from_checkpoint_tp4(
        ckpt: &'a mut Checkpoint,
        sharding: Sharding,
        comm: Box<dyn Collectives>,
    ) -> io::Result<GpuModel<'a>> {
        if sharding.world_size != 4 {
            return ioerr("TP4 needs world_size 4".into());
        }
        let cfg = RefConfig::from_model_config(&ckpt.config)?;
        validate_dims(&cfg)?;
        let n_experts_local = cfg.n_routed / sharding.world_size;
        let mut store: Box<dyn WeightStore + 'a> = Box::new(CkptStore::new(ckpt));
        validate_layout(&cfg, store.as_mut())?;
        let t_b = std::time::Instant::now();
        let mut model = Self::build(cfg, store.as_mut(), DEFAULT_MAX_SEQ_LEN, None, &sharding)?;
        eprintln!(
            "tp4 rank {}: upload weight pass in {:?} ({}/{} experts device-resident)",
            sharding.rank,
            t_b.elapsed(),
            n_experts_local,
            n_experts_local * sharding.world_size
        );
        model.store_keepalive = Some(store);
        model.sharding = sharding;
        model.comm = Some(comm);
        Ok(model)
    }

    fn build(
        cfg: RefConfig,
        store: &mut dyn WeightStore,
        max_seq: u64,
        arena: Option<&crate::stream::HostArena>,
        sh: &Sharding,
    ) -> io::Result<Self> {
        let guard = Cuda::global();
        let c = guard
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "cuda init"))?;

        let engram = EngramLayout::build(&cfg)?;
        for (i, need) in engram.required_rows().iter().enumerate() {
            if cfg.engram_num_embeddings[i] < *need {
                return ioerr(format!(
                    "engram layer {} needs {need} rows, config says {}",
                    cfg.engram_layers[i], cfg.engram_num_embeddings[i]
                ));
            }
        }

        let dim = cfg.dim as usize;
        let hc = cfg.hc_mult as usize;
        let hd = cfg.head_dim as usize;
        let win = cfg.window_size as usize;
        let n_layers = cfg.n_layers as usize;

        // kv-source wiring + publish bands (mirror of the ref's shared slots)
        let mut kv_src_of = vec![None::<usize>; n_layers];
        for (ord, &s) in cfg.kv_sources.iter().enumerate() {
            kv_src_of[s as usize] = Some(ord);
        }
        let mut publisher = vec![None::<usize>; n_layers];
        let mut cur = None;
        for l in 0..n_layers {
            if kv_src_of[l].is_some() && cfg.compress_ratios[l] > 0 {
                cur = kv_src_of[l];
            }
            if cfg.compress_ratios[l] > 0 {
                if let Some(pord) = cur {
                    if cfg.compress_ratios[l] != cfg.compress_ratios[cfg.kv_sources[pord] as usize]
                    {
                        return ioerr(format!("layer {l} ratio differs from its kv source's"));
                    }
                }
                publisher[l] = cur;
            }
        }
        let mut bands: Vec<Vec<usize>> = vec![Vec::new(); cfg.kv_sources.len()];
        for l in 0..n_layers {
            if let Some(p) = publisher[l] {
                bands[p].push(l);
            }
        }

        // rope tables: two variants shared by all layers (spec §3), built
        let t_phase = std::time::Instant::now();
        // with the ref's own Freqs computation so the tables are
        // bit-identical to the CPU engine's for ANY config (YaRN comes from
        // cfg.original_seq_len — the ferroseek-cuda helpers hardcode the
        // 1017 params, which is wrong for configs with YaRN off)
        let half = cfg.rope_dim as usize / 2;
        let expand = |f: &ferroseek_ref::rope::Freqs| -> (Vec<f32>, Vec<f32>) {
            let rows = max_seq as usize + 1;
            let mut cos = vec![0.0f32; rows * half];
            let mut sin = vec![0.0f32; rows * half];
            for pos in 0..rows {
                for i in 0..half {
                    let ang = pos as f32 * f.freq[i];
                    cos[pos * half + i] = ang.cos();
                    sin[pos * half + i] = ang.sin();
                }
            }
            (cos, sin)
        };
        let plain_f = ferroseek_ref::rope::Freqs::new(cfg.rope_dim, cfg.rope_theta, 0, 1.0, 1, 1);
        let comp_f = ferroseek_ref::rope::Freqs::new(
            cfg.rope_dim,
            cfg.compress_rope_theta,
            cfg.original_seq_len,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
        );
        let (cos_plain, sin_plain) = expand(&plain_f);
        let (cos_comp, sin_comp) = expand(&comp_f);
        let rope_cos = [up_f32(c, &cos_plain)?, up_f32(c, &cos_comp)?];
        let rope_sin = [up_f32(c, &sin_plain)?, up_f32(c, &sin_comp)?];
        eprintln!("build: rope tables in {:?}", t_phase.elapsed());

        let mut layers = Vec::with_capacity(n_layers);
        let t_layers = std::time::Instant::now();
        for l in 0..n_layers {
            let t_l = std::time::Instant::now();
            layers.push(upload_layer(&cfg, store, c, l, kv_src_of[l], arena, sh)?);
            eprintln!("build: layer {l} in {:?}", t_l.elapsed());
        }
        eprintln!("build: {n_layers} layers in {:?}", t_layers.elapsed());

        // embedding + head as f32 (spec §11: head held fp32; embed is
        // row-gathered)
        // streaming mode reads embedding rows host-side (no 2.6 GB device
        // table); upload mode gathers from a device copy
        let t_head = std::time::Instant::now();
        let emb = match arena {
            Some(_) => None,
            None => Some(up_f32(c, &view_f32(store, "embed.weight")?)?),
        };
        let head = {
            // vocab-sharded head (spec §3): the rank's 129280/world rows,
            // bf16 promoted to fp32 like the ref (model.py:1008). The local
            // gemv writes the first vocab/world logits; the full vector is
            // assembled by the all-gather at the end of decode.
            let v = store.get("head.weight")?;
            let row_bytes = dim * dtype_size(v.dtype);
            let vl = cfg.vocab as usize / sh.world_size as usize;
            let r0 = sh.rank as usize * vl;
            if v.dtype != DType::Bf16 {
                return ioerr(format!("head.weight: expected BF16, got {:?}", v.dtype));
            }
            if r0 * row_bytes + vl * row_bytes > v.bytes.len() {
                return ioerr("head.weight: vocab shard out of range".into());
            }
            let host: Vec<f32> = v.bytes[r0 * row_bytes..(r0 + vl) * row_bytes]
                .chunks_exact(2)
                .map(|b| ferroseek_core::quant::bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            drop(v);
            up_f32(c, &host)?
        };
        let final_norm = up_f32(c, &view_f32(store, "norm.weight")?)?;
        eprintln!("build: embed/head/norm in {:?}", t_head.elapsed());

        // caches: per layer kv_all = ring + compressed view; per source
        // index_k + compressor state
        let t_caches = std::time::Instant::now();
        let mut kv_all = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ratio = cfg.compress_ratios[l] as u64;
            let rows = if ratio > 0 {
                win + (max_seq / ratio + 1) as usize
            } else {
                win
            };
            kv_all.push(up_zero_f32(c, rows * hd)?);
        }
        let mut index_k = Vec::new();
        let mut comp_kv = Vec::new();
        let mut comp_score = Vec::new();
        for &s in &cfg.kv_sources {
            let ratio = cfg.compress_ratios[s as usize] as u64;
            index_k.push(up_zero_f32(
                c,
                (max_seq / ratio + 1) as usize * cfg.index_head_dim as usize,
            )?);
            if ratio > 1 {
                comp_kv.push(Some(up_zero_f32(c, ratio as usize * hd)?));
                let score = vec![f32::NEG_INFINITY; ratio as usize * hd];
                comp_score.push(Some(up_f32(c, &score)?));
            } else {
                comp_kv.push(None);
                comp_score.push(None);
            }
        }
        eprintln!("build: caches in {:?}", t_caches.elapsed());

        // scratch
        let t_scratch = std::time::Instant::now();
        let score_cap = cfg
            .kv_sources
            .iter()
            .map(|&s| (max_seq / cfg.compress_ratios[s as usize] as u64 + 1) as usize)
            .max()
            .unwrap_or(1);
        if cfg.inter_dim as usize % 32 != 0 {
            return ioerr("moe_intermediate_size must be 32-divisible (fp4/fp8 K blocks)".into());
        }
        let eflat_len = engram.n_hash_cols * engram.head_dim;
        let act = |n: usize| -> io::Result<(D<u8>, D<u8>)> {
            assert!(n % 32 == 0 && n > 0);
            Ok((
                DeviceBuffer::zeroed(n).map_err(cuda_err)?,
                DeviceBuffer::zeroed(n / 32).map_err(cuda_err)?,
            ))
        };
        let (act_x, act_x_s) = act(dim)?;
        let (act_qr, act_qr_s) = act(cfg.q_lora as usize)?;
        let (act_fl, act_fl_s) = act(cfg.o_groups as usize * cfg.o_lora as usize)?;
        let (act_h, act_h_s) = act(cfg.inter_dim as usize)?;
        let (act_e, act_e_s) = act(eflat_len.max(32))?;
        let scratch = Scratch {
            h: [up_zero_f32(c, hc * dim)?, up_zero_f32(c, hc * dim)?],
            x: up_zero_f32(c, dim)?,
            qr: up_zero_f32(c, cfg.q_lora as usize)?,
            q: up_zero_f32(c, cfg.n_heads as usize * hd)?,
            kvrow: up_zero_f32(c, hd)?,
            o: up_zero_f32(c, cfg.n_heads as usize * hd)?,
            fl: up_zero_f32(c, cfg.o_groups as usize * cfg.o_lora as usize)?,
            attn_out: up_zero_f32(c, dim)?,
            y_moe: up_zero_f32(c, dim)?,
            gate_buf: up_zero_f32(c, cfg.inter_dim as usize)?,
            up_buf: up_zero_f32(c, cfg.inter_dim as usize)?,
            local_slots: DeviceBuffer::zeroed(cfg.n_activated as usize).map_err(cuda_err)?,
            local_experts: DeviceBuffer::zeroed(cfg.n_activated as usize).map_err(cuda_err)?,
            local_count: DeviceBuffer::zeroed(1).map_err(cuda_err)?,
            h_slots: up_zero_f32(c, cfg.n_activated as usize * cfg.inter_dim as usize)?,
            q_slots: DeviceBuffer::zeroed(cfg.n_activated as usize * cfg.inter_dim as usize)
                .map_err(cuda_err)?,
            qs_slots: DeviceBuffer::zeroed(cfg.n_activated as usize * cfg.inter_dim as usize / 32)
                .map_err(cuda_err)?,
            h_e: up_zero_f32(c, cfg.inter_dim as usize)?,
            expert_y: up_zero_f32(c, dim)?,
            latent: up_zero_f32(c, hd)?,
            krow: up_zero_f32(c, cfg.index_head_dim as usize)?,
            qidx: up_zero_f32(c, cfg.index_n_heads as usize * cfg.index_head_dim as usize)?,
            widx: up_zero_f32(c, cfg.index_n_heads as usize)?,
            score: up_zero_f32(c, score_cap)?,
            eflat: up_zero_f32(c, eflat_len.max(1))?,
            ekv: up_zero_f32(c, dim * (hc + 1))?,
            hash_ids: DeviceBuffer::zeroed(engram.n_hash_cols.max(1)).map_err(cuda_err)?,
            act_x,
            act_x_s,
            act_qr,
            act_qr_s,
            act_fl,
            act_fl_s,
            act_h,
            act_h_s,
            act_e,
            act_e_s,
            gate_w_out: up_zero_f32(c, cfg.n_activated as usize)?,
            gate_scores: up_zero_f32(c, cfg.n_routed as usize)?,
            hc_dots: up_zero_f32(c, cfg.mix_hc() as usize)?,
            gate_i_out: DeviceBuffer::zeroed(cfg.n_activated as usize).map_err(cuda_err)?,
            win_idxs: DeviceBuffer::zeroed(win).map_err(cuda_err)?,
            shared_idxs: DeviceBuffer::zeroed(win + cfg.index_topk as usize).map_err(cuda_err)?,
            pre_attn: up_zero_f32(c, hc)?,
            pre_ffn: up_zero_f32(c, hc)?,
            post: up_zero_f32(c, hc)?,
            comb: up_zero_f32(c, hc * hc)?,
            zeros_h: up_zero_f32(c, hc * dim)?,
            ones_hc: up_f32(c, &vec![1.0f32; hc])?,
            comb_zero: up_zero_f32(c, hc * hc)?,
            emb_row: up_zero_f32(c, dim)?,
            logits: up_zero_f32(c, cfg.vocab as usize)?,
            argmax_i: DeviceBuffer::zeroed(1).map_err(cuda_err)?,
            argmax_v: DeviceBuffer::zeroed(1).map_err(cuda_err)?,
            coll_stage: crate::stream::PinnedBuf::new(
                cfg.vocab as usize / sh.world_size as usize * 4 + 4,
            )?,
            gate_stage: crate::stream::PinnedBuf::new(1024)?,
            small_stage: crate::stream::PinnedBuf::new(64 << 10)?,
        };

        let n_sources = cfg.kv_sources.len();
        eprintln!("build: scratch in {:?}", t_scratch.elapsed());
        Ok(GpuModel {
            cfg,
            layers,
            caches: GpuCaches {
                kv_all,
                index_k,
                comp_kv,
                comp_score,
            },
            engram,
            eng_cache: Vec::new(),
            token_map: None,
            rope_cos,
            rope_sin,
            emb,
            head,
            final_norm,
            scratch,
            bands,
            run: RunState {
                compress_len: vec![0; n_sources],
                index_k_src: 0,
                candidates: Vec::new(),
                shared_n_sel: 0,
                capture_x: None,
            },
            pos: 0,
            max_seq,
            sharding: *sh,
            comm: None,
            prof: Prof::from_env(),
            prefix_cache: Vec::new(),
            arena: None,
            store_keepalive: None,
        })
    }
}

// ---------------------------------------------------------------------------
// decode sublayers

/// The attention sublayer (spec §3-§5, model.py:614-791): q path, window KV
/// ring, CSA compressor, DSA indexer, sparse attention, grouped output
/// projection.
#[allow(clippy::too_many_arguments)]
fn attention_step(
    c: &Cuda,
    cfg: &RefConfig,
    layer: &GpuLayer,
    s: &mut Scratch,
    caches: &mut GpuCaches,
    run: &mut RunState,
    bands: &[Vec<usize>],
    rope_cos: &[D<f32>; 2],
    rope_sin: &[D<f32>; 2],
    pos: u64,
    sh: Sharding,
    mut comm: Option<&mut (dyn Collectives + 'static)>,
) -> io::Result<()> {
    let dim = cfg.dim as usize;
    let hd = cfg.head_dim as usize;
    let rd = cfg.rope_dim as usize;
    let win = cfg.window_size as usize;
    let eps = cfg.norm_eps;
    let l = layer.layer as usize;
    // TP4 locals: the rank's head slice (16 of 64), o-group slice (2 of 8),
    // index-head slice (8 of 32). Every head-dim use below is local: the
    // sharded weights (wq_b, wo_a, wo_b, indexer) load as local blocks and
    // the kernels take the local counts.
    let world = sh.world_size as usize;
    let heads = cfg.n_heads as usize / world;
    let og = cfg.o_groups as usize / world;
    let ol = cfg.o_lora as usize;
    let inh = cfg.index_n_heads as usize / world;
    let ihd = cfg.index_head_dim as usize;

    if run.capture_x.is_some() {
        let mut v = vec![0.0f32; dim];
        s.x.index(0..dim).copy_to(&mut v).map_err(cuda_err)?;
        run.capture_x.as_mut().unwrap().push((layer.layer, v));
    }

    // q path: wq_a -> q_norm -> wq_b -> rope (model.py:771-773)
    k::emit8(
        c,
        k::base(&s.x),
        k::base(&s.act_x),
        k::base(&s.act_x_s),
        dim,
    );
    dump_u8(c, &s.act_x, dim, format!("d_pos{pos}_actx{l}.u8"));
    dump_u8(c, &s.act_x_s, dim / 32, format!("d_pos{pos}_actxs{l}.u8"));
    k::fp8_gemm(
        c,
        k::base(&s.act_x),
        k::base(&s.act_x_s),
        layer.wq_a.w.ptr(),
        layer.wq_a.s.ptr(),
        k::base(&s.qr),
        1,
        cfg.q_lora as usize,
        dim,
    );
    k::rmsnorm(
        c,
        k::base(&s.qr),
        k::base(&layer.q_norm_w),
        k::base(&s.qr),
        cfg.q_lora as usize,
        eps,
    );
    k::emit8(
        c,
        k::base(&s.qr),
        k::base(&s.act_qr),
        k::base(&s.act_qr_s),
        cfg.q_lora as usize,
    );
    dump_dev(
        c,
        &s.qr,
        cfg.q_lora as usize,
        format!("d_pos{pos}_qr{l}.f32"),
    );
    dump_u8(
        c,
        &s.act_qr,
        cfg.q_lora as usize,
        format!("d_pos{pos}_actqr{l}.u8"),
    );
    dump_u8(
        c,
        &s.act_qr_s,
        cfg.q_lora as usize / 32,
        format!("d_pos{pos}_actqrs{l}.u8"),
    );
    k::fp8_gemm(
        c,
        k::base(&s.act_qr),
        k::base(&s.act_qr_s),
        layer.wq_b.w.ptr(),
        layer.wq_b.s.ptr(),
        k::base(&s.q),
        1,
        heads * hd,
        cfg.q_lora as usize,
    );
    k::rope_uniform_pos(
        c,
        k::base(&s.q),
        k::base(&rope_cos[layer.freq]),
        k::base(&rope_sin[layer.freq]),
        heads,
        hd,
        rd,
        pos,
        false,
    );
    dump_dev(c, &s.q, heads * hd, format!("d_pos{pos}_q{l}.f32"));

    // window KV: one shared latent, rope tail, fp8 snap, ring write
    k::fp8_gemm(
        c,
        k::base(&s.act_x),
        k::base(&s.act_x_s),
        layer.wkv.w.ptr(),
        layer.wkv.s.ptr(),
        k::base(&s.kvrow),
        1,
        hd,
        dim,
    );
    let slot = (pos % win as u64) as usize;
    let ring = k::row(&caches.kv_all[l], slot * hd, hd);
    k::rmsnorm(
        c,
        k::base(&s.kvrow),
        k::base(&layer.kv_norm_w),
        ring,
        hd,
        eps,
    );
    k::rope(
        c,
        ring,
        k::base(&rope_cos[layer.freq]),
        k::base(&rope_sin[layer.freq]),
        1,
        hd,
        rd,
        pos,
        false,
    );
    k::snap8(c, ring, hd);

    // compressed path (spec §4-§5)
    let mut latent_ready = false;
    if layer.ratio > 0 {
        let ratio = layer.ratio as u64;
        let compress_len = ((pos + 1) / ratio) as usize;

        // CSA compressor: kv sources park wkv/wgate outputs; the latent is
        // produced on group completion
        if let Some(src) = layer.kv_src {
            run.index_k_src = src;
            let comp = layer.compressor.as_ref().unwrap();
            if ratio == 1 {
                k::f32_gemv(
                    c,
                    k::base(&comp.wkv),
                    k::base(&s.x),
                    k::base(&s.latent),
                    hd,
                    dim,
                );
                k::rmsnorm(
                    c,
                    k::base(&s.latent),
                    k::base(&comp.norm_w),
                    k::base(&s.latent),
                    hd,
                    eps,
                );
                latent_ready = true;
            } else {
                let slot_i = (pos % ratio) as usize;
                let kv_state = caches.comp_kv[src].as_ref().unwrap();
                let score_state = caches.comp_score[src].as_mut().unwrap();
                k::f32_gemv(
                    c,
                    k::base(&comp.wkv),
                    k::base(&s.x),
                    k::row(kv_state, slot_i * hd, hd),
                    hd,
                    dim,
                );
                k::f32_gemv(
                    c,
                    k::base(comp.wgate.as_ref().unwrap()),
                    k::base(&s.x),
                    k::row(score_state, slot_i * hd, hd),
                    hd,
                    dim,
                );
                if (pos + 1) % ratio == 0 {
                    k::compressor_finish(
                        c,
                        k::base(kv_state),
                        k::base(score_state),
                        k::base(&s.latent),
                        hd,
                    );
                    // RMS norm of the pooled latent (model.py:485; the
                    // ratio-1 branch norms right after its gemv)
                    k::rmsnorm(
                        c,
                        k::base(&s.latent),
                        k::base(&comp.norm_w),
                        k::base(&s.latent),
                        hd,
                        eps,
                    );
                    latent_ready = true;
                }
            }
            if latent_ready {
                run.compress_len[src] = compress_len;
            }
        }

        // DSA indexer (index sources run their own; consumers reuse the
        // published idxs — model.py:723-738). inh/ihd are the rank's local
        // index-head slice (8 of 32 at TP4); scores all-reduce.
        if layer.idx_src {
            // key side: append on group completion, pre-RoPE latent
            if layer.owns_k && latent_ready {
                let kowner = layer.k_owner.as_ref().unwrap();
                let j = compress_len - 1;
                k::bf16_gemv(
                    c,
                    k::base(&kowner.wk),
                    k::base(&s.latent),
                    k::base(&s.krow),
                    ihd,
                    hd,
                );
                let krow_dst = k::row(&caches.index_k[run.index_k_src], j * ihd, ihd);
                k::rmsnorm(
                    c,
                    k::base(&s.krow),
                    k::base(&kowner.k_norm),
                    krow_dst,
                    ihd,
                    eps,
                );
                k::rope(
                    c,
                    krow_dst,
                    k::base(&rope_cos[layer.freq]),
                    k::base(&rope_sin[layer.freq]),
                    1,
                    ihd,
                    rd,
                    pos + 1 - ratio,
                    false,
                );
                k::fp4_snap(c, krow_dst, ihd, 32, false);
            }
            // query side: wq_b(qr) at pos, fp4(32, e8m0) snap
            k::fp8_gemm(
                c,
                k::base(&s.act_qr),
                k::base(&s.act_qr_s),
                layer.indexer.as_ref().unwrap().wq_b.w.ptr(),
                layer.indexer.as_ref().unwrap().wq_b.s.ptr(),
                k::base(&s.qidx),
                1,
                inh * ihd,
                cfg.q_lora as usize,
            );
            k::rope_uniform_pos(
                c,
                k::base(&s.qidx),
                k::base(&rope_cos[layer.freq]),
                k::base(&rope_sin[layer.freq]),
                inh,
                ihd,
                rd,
                pos,
                false,
            );
            k::fp4_snap(c, k::base(&s.qidx), inh * ihd, 32, false);
            // per-head weights from weights_proj (scale folded at load)
            k::f32_gemv(
                c,
                k::base(&layer.indexer.as_ref().unwrap().weights_proj),
                k::base(&s.x),
                k::base(&s.widx),
                inh,
                dim,
            );

            // score + selection
            let mut idxs_host;
            if compress_len == 0 {
                idxs_host = window_topk_idxs(win, pos);
            } else {
                k::indexer_score(
                    c,
                    k::base(&s.qidx),
                    k::base(&caches.index_k[run.index_k_src]),
                    k::base(&s.widx),
                    k::base(&s.score),
                    inh,
                    ihd,
                    compress_len,
                );
                let mut score = vec![0.0f32; compress_len];
                {
                    let stream = c.stream.as_inner();
                    s.coll_stage.dtoh(
                        s.score.as_device_ptr().as_raw() as u64,
                        compress_len * 4,
                        stream,
                    )?;
                    s.coll_stage.sync(stream)?;
                    score.copy_from_slice(&s.coll_stage.as_f32()[..compress_len]);
                }
                // index_score all-reduce (model.py:559-560): each rank
                // scored only its 8 index heads
                if let Some(comm) = comm.as_deref_mut() {
                    score = comm.allreduce(&score)?;
                }
                if layer.cand_src {
                    run.candidates = select_candidate_blocks(
                        &score,
                        cfg.candidate_topk_blocks as usize,
                        cfg.candidate_block_size as usize,
                    );
                } else if layer.uses_cand {
                    for (t, v) in score.iter_mut().enumerate() {
                        if !run.candidates.get(t).copied().unwrap_or(false) {
                            *v = f32::NEG_INFINITY;
                        }
                    }
                }
                let topk = (cfg.index_topk as usize).min(compress_len);
                let cmp = select_topk(&score, topk, compress_len, win as i32);
                idxs_host = window_topk_idxs(win, pos);
                idxs_host.extend(cmp);
            }
            // partial upload: the buffer is sized for the worst case, the
            // kernel reads only [0, n_sel)
            s.shared_idxs
                .index(0..idxs_host.len())
                .copy_from(&idxs_host)
                .map_err(cuda_err)?;
            run.shared_n_sel = idxs_host.len();
        }

        // latent tail: rope at the group's first-token position, fp4(16,
        // e4m3) snap, fan the row out to every layer in the publish band
        // (each layer's kv_all carries its own compressed view)
        if latent_ready {
            if let Some(src) = layer.kv_src {
                let j = compress_len - 1;
                k::rope(
                    c,
                    k::base(&s.latent),
                    k::base(&rope_cos[layer.freq]),
                    k::base(&rope_sin[layer.freq]),
                    1,
                    hd,
                    rd,
                    pos + 1 - ratio,
                    false,
                );
                k::fp4_snap(c, k::base(&s.latent), hd, 16, true);
                for &cl in &bands[src] {
                    let dst = k::row(&caches.kv_all[cl], win * hd + j * hd, hd);
                    k::copy_f32(c, k::base(&s.latent), dst, hd);
                }
            }
        }
    }

    // sparse attention over [ring rows | compressed rows]. Before the
    // first index source publishes this step, the shared slot is empty and
    // the ref attends window-only (cat with an empty tensor).
    let (idxs, n_sel) = if layer.ratio > 0 && run.shared_n_sel > 0 {
        (k::base(&s.shared_idxs), run.shared_n_sel)
    } else {
        (k::base(&s.win_idxs), win)
    };
    dump_dev(
        c,
        &caches.kv_all[l],
        caches.kv_all[l].len(),
        format!("d_pos{pos}_kvall{l}.f32"),
    );
    k::sparse_attn(
        c,
        k::base(&s.q),
        k::base(&caches.kv_all[l]),
        idxs,
        k::base(&layer.attn_sink),
        k::base(&s.o),
        heads,
        hd,
        n_sel,
        (hd as f32).sqrt().recip(),
    );
    dump_dev(c, &s.o, heads * hd, format!("d_pos{pos}_attnraw{l}.f32"));

    // inverse rope on the output tail (the cache stays in the query-rotated
    // form), fp8 snap per group row, block-diagonal wo_a, wo_b
    k::rope_uniform_pos(
        c,
        k::base(&s.o),
        k::base(&rope_cos[layer.freq]),
        k::base(&rope_sin[layer.freq]),
        heads,
        hd,
        rd,
        pos,
        true,
    );
    let gdim = heads * hd / og;
    for g in 0..og {
        k::snap8(c, k::row(&s.o, g * gdim, gdim), gdim);
    }
    dump_dev(c, &s.o, heads * hd, format!("d_pos{pos}_osnap{l}.f32"));
    for g in 0..og {
        k::bf16_gemv(
            c,
            k::row16(&layer.wo_a, g * ol * gdim, ol * gdim),
            k::row(&s.o, g * gdim, gdim),
            k::row(&s.fl, g * ol, ol),
            ol,
            gdim,
        );
    }
    k::emit8(
        c,
        k::base(&s.fl),
        k::base(&s.act_fl),
        k::base(&s.act_fl_s),
        og * ol,
    );
    dump_dev(c, &s.fl, og * ol, format!("d_pos{pos}_fl{l}.f32"));
    k::fp8_gemm(
        c,
        k::base(&s.act_fl),
        k::base(&s.act_fl_s),
        layer.wo_b.w.ptr(),
        layer.wo_b.s.ptr(),
        k::base(&s.attn_out),
        1,
        dim,
        og * ol,
    );
    // wo_b is row-parallel: fp32 partial-sum all-reduce (spec §3)
    if let Some(comm) = comm.as_deref_mut() {
        let stream = c.stream.as_inner();
        s.coll_stage
            .dtoh(s.attn_out.as_device_ptr().as_raw() as u64, dim * 4, stream)?;
        s.coll_stage.sync(stream)?;
        let red = comm.allreduce(&s.coll_stage.as_f32()[..dim])?;
        s.attn_out.copy_from(&red).map_err(cuda_err)?;
    }
    dump_dev(c, &s.attn_out, dim, format!("d_pos{pos}_attnout{l}.f32"));
    Ok(())
}

/// Env-gated device->host dump of `n` u8 from `buf` (gate bisection).
fn dump_u8(c: &Cuda, buf: &D<u8>, n: usize, name: String) {
    if std::env::var_os("FERROSEEK_DUMP_GPU").is_none() {
        return;
    }
    let _ = c.stream.synchronize();
    let mut v = vec![0u8; n];
    buf.index(0..n).copy_to(&mut v).ok();
    crate::dump::u8s(name, &v);
}

/// Env-gated device->host dump of `n` f32 from `buf` (gate bisection).
fn dump_dev(c: &Cuda, buf: &D<f32>, n: usize, name: String) {
    if std::env::var_os("FERROSEEK_DUMP_GPU").is_none() {
        return;
    }
    let _ = c.stream.synchronize();
    let mut v = vec![0.0f32; n];
    buf.index(0..n).copy_to(&mut v).ok();
    crate::dump::f32s(name, &v);
}

/// The MoE sublayer (spec §6, model.py:794-906): device-side routing, then
/// selected experts in ascending expert-id order (matching the CPU
/// accumulation order), then the shared expert.
fn moe_step(
    c: &Cuda,
    cfg: &RefConfig,
    layer: &GpuLayer,
    s: &mut Scratch,
    arena: Option<&crate::stream::HostArena>,
    sh: Sharding,
    mut comm: Option<&mut (dyn Collectives + 'static)>,
    prof: &mut Prof,
) -> io::Result<()> {
    let l = layer.layer;
    let dim = cfg.dim as usize;
    let inter = cfg.inter_dim as usize;
    let n_experts = cfg.n_routed as usize;
    let top_k = cfg.n_activated as usize;
    // expert-index shard: rank r owns experts [96r, 96r+96) of every
    // layer (spec §3). The gate is replicated, so every rank routes the
    // identical top-6 and computes only its intersection.
    let world = sh.world_size as usize;
    let e_start = sh.rank as usize * n_experts / world;
    let e_len = n_experts / world;

    let t_gate = Instant::now();
    if layer.w1_all.is_some() {
        // parallel gate (moe.cu): one block per expert; the one-block form
        // did 384 serial 5120-dots (1.1 ms/call, 20% of GPU time in nsys)
        k::moe_gate(
            c,
            k::base(&s.x),
            k::base(&layer.gate_w),
            k::base(&layer.gate_bias),
            k::base(&s.gate_scores),
            dim,
            n_experts,
        );
        k::moe_select(
            c,
            k::base(&s.gate_scores),
            k::base(&layer.gate_bias),
            k::base(&s.gate_w_out),
            k::base(&s.gate_i_out),
            n_experts,
            top_k,
            cfg.route_scale,
        );
    } else {
        k::moe_route(
            c,
            k::base(&s.x),
            k::base(&layer.gate_w),
            k::base(&layer.gate_bias),
            k::base(&s.gate_w_out),
            k::base(&s.gate_i_out),
            dim,
            n_experts,
            top_k,
            cfg.route_scale,
        );
    }
    let _ = s.y_moe.set_zero();
    // the activation is quantized once and reused by every expert and the
    // shared expert (identical to the ref's per-gemv quantization of x)
    k::emit8(
        c,
        k::base(&s.x),
        k::base(&s.act_x),
        k::base(&s.act_x_s),
        dim,
    );
    prof.add_us(2, t_gate);

    if layer.w1_all.is_some() {
        // Fused grouped-expert path (moe_grouped.cu): the gate's top-6 is
        // consumed on the device — no D2H readback, no host expert loop
        // (those cost ~107 ms + ~43 ms of a 300 ms token at 40 layers).
        // Bit-exact by construction: moe_local_slots sorts the rank's local
        // picks ascending by global expert id (model.py:895 order),
        // expert_glu/expert_quant reuse the fp4_gemm/emit8/swiglu math
        // per slot, and expert_reduce adds per-expert block-reduced partials
        // in the same ascending order the host loop used.
        let w1 = layer.w1_all.as_ref().unwrap();
        let w1s = layer.w1s_all.as_ref().unwrap();
        let w2 = layer.w2_all.as_ref().unwrap();
        let w2s = layer.w2s_all.as_ref().unwrap();
        let w3 = layer.w3_all.as_ref().unwrap();
        let w3s = layer.w3s_all.as_ref().unwrap();
        k::moe_local_slots(
            c,
            k::base(&s.gate_i_out),
            k::base(&s.local_slots),
            k::base(&s.local_experts),
            k::base(&s.local_count),
            top_k,
            e_start,
            e_len,
        );
        k::expert_glu(
            c,
            k::base(&s.act_x),
            k::base(&s.act_x_s),
            k::base(w1),
            k::base(w1s),
            k::base(w3),
            k::base(w3s),
            k::base(&s.gate_w_out),
            k::base(&s.local_slots),
            k::base(&s.local_experts),
            k::base(&s.local_count),
            k::base(&s.h_slots),
            inter,
            dim,
            top_k,
        );
        k::expert_quant(
            c,
            k::base(&s.h_slots),
            k::base(&s.local_slots),
            k::base(&s.local_count),
            k::base(&s.q_slots),
            k::base(&s.qs_slots),
            inter,
            top_k,
        );
        k::expert_reduce(
            c,
            k::base(&s.q_slots),
            k::base(&s.qs_slots),
            k::base(w2),
            k::base(w2s),
            k::base(&s.gate_i_out),
            k::base(&s.local_slots),
            k::base(&s.local_count),
            k::base(&s.y_moe),
            dim,
            inter,
            top_k,
            e_start,
        );
    } else {
        // Streaming mode (per-use staging): host dispatch, identical math.
        // Pinned async readback — the pageable copy_to drained the queue.
        let mut w_host = vec![0.0f32; top_k];
        let mut i_host = vec![0i32; top_k];
        {
            let stream = c.stream.as_inner();
            s.gate_stage.dtoh(
                s.gate_w_out.as_device_ptr().as_raw() as u64,
                top_k * 4,
                stream,
            )?;
            s.gate_stage.dtoh_at(
                s.gate_i_out.as_device_ptr().as_raw() as u64,
                top_k * 4,
                top_k * 4, // host offset: ids after weights
                stream,
            )?;
            s.gate_stage.sync(stream)?;
            let f32s = s.gate_stage.as_f32();
            w_host.copy_from_slice(&f32s[..top_k]);
            let i32s = s.gate_stage.as_i32_at(top_k * 4);
            i_host.copy_from_slice(&i32s[..top_k]);
        }
        crate::dump::i32s(
            format!("d_pos{}_gorder{l}.i32", crate::dump::pos()),
            &i_host,
        );
        crate::dump::f32s(
            format!("d_pos{}_gweights{l}.f32", crate::dump::pos()),
            &w_host,
        );
        let mut sel: Vec<(usize, f32)> = i_host
            .iter()
            .map(|&i| i as usize)
            .zip(w_host.iter().copied())
            .collect();
        sel.sort_by_key(|&(e, _)| e); // ascending expert id (model.py:895)
        for &(e, w) in &sel {
            if e < e_start || e >= e_start + e_len {
                continue; // another rank's expert
            }
            let [w1, w2, w3] = &layer.experts[e - e_start];
            let (w1d, w1sd) = (w1.w.stage(c, arena)?, w1.s.stage(c, arena)?);
            let (w3d, w3sd) = (w3.w.stage(c, arena)?, w3.s.stage(c, arena)?);
            k::fp4_gemm(
                c,
                k::base(&s.act_x),
                k::base(&s.act_x_s),
                w1.w.ptr_or(&w1d),
                w1.s.ptr_or(&w1sd),
                k::base(&s.gate_buf),
                1,
                inter,
                dim,
            );
            k::fp4_gemm(
                c,
                k::base(&s.act_x),
                k::base(&s.act_x_s),
                w3.w.ptr_or(&w3d),
                w3.s.ptr_or(&w3sd),
                k::base(&s.up_buf),
                1,
                inter,
                dim,
            );
            k::swiglu(
                c,
                k::base(&s.gate_buf),
                k::base(&s.up_buf),
                k::base(&s.h_e),
                inter,
                w,
            );
            k::emit8(
                c,
                k::base(&s.h_e),
                k::base(&s.act_h),
                k::base(&s.act_h_s),
                inter,
            );
            drop((w1d, w1sd, w3d, w3sd));
            let (w2d, w2sd) = (w2.w.stage(c, arena)?, w2.s.stage(c, arena)?);
            k::fp4_gemm(
                c,
                k::base(&s.act_h),
                k::base(&s.act_h_s),
                w2.w.ptr_or(&w2d),
                w2.s.ptr_or(&w2sd),
                k::base(&s.expert_y),
                1,
                dim,
                inter,
            );
            drop((w2d, w2sd));
            k::acc_f32(c, k::base(&s.expert_y), k::base(&s.y_moe), dim);
        }
    }

    // routed-expert partial all-reduce (spec §3 TP4 note): fp32, rank-
    // ordered. The shared expert is replicated and added AFTER the reduce
    // (recomputed, not reduced), so every rank lands on identical bits.
    if let Some(comm) = comm.as_deref_mut() {
        let stream = c.stream.as_inner();
        s.coll_stage
            .dtoh(s.y_moe.as_device_ptr().as_raw() as u64, dim * 4, stream)?;
        s.coll_stage.sync(stream)?;
        let red = comm.allreduce(&s.coll_stage.as_f32()[..dim])?;
        s.y_moe.copy_from(&red).map_err(cuda_err)?;
    }

    // shared expert on every token, added last (model.py:905)
    let [w1, w2, w3] = &layer.shared_experts;
    k::fp8_gemm(
        c,
        k::base(&s.act_x),
        k::base(&s.act_x_s),
        w1.w.ptr(),
        w1.s.ptr(),
        k::base(&s.gate_buf),
        1,
        inter,
        dim,
    );
    k::fp8_gemm(
        c,
        k::base(&s.act_x),
        k::base(&s.act_x_s),
        w3.w.ptr(),
        w3.s.ptr(),
        k::base(&s.up_buf),
        1,
        inter,
        dim,
    );
    k::swiglu(
        c,
        k::base(&s.gate_buf),
        k::base(&s.up_buf),
        k::base(&s.h_e),
        inter,
        1.0,
    );
    k::emit8(
        c,
        k::base(&s.h_e),
        k::base(&s.act_h),
        k::base(&s.act_h_s),
        inter,
    );
    k::fp8_gemm(
        c,
        k::base(&s.act_h),
        k::base(&s.act_h_s),
        w2.w.ptr(),
        w2.s.ptr(),
        k::base(&s.expert_y),
        1,
        dim,
        inter,
    );
    k::acc_f32(c, k::base(&s.expert_y), k::base(&s.y_moe), dim);
    dump_dev(
        c,
        &s.y_moe,
        dim,
        format!("d_pos{}_y{l}.f32", crate::dump::pos()),
    );
    Ok(())
}
