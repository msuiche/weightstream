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
use ferroseek_core::ModelConfig;
use ferroseek_core::loader::Checkpoint;
use ferroseek_cuda::gpu::Cuda;
use ferroseek_ref::attn::{select_candidate_blocks, select_topk, window_topk_idxs};
use ferroseek_ref::config::RefConfig;
use ferroseek_ref::engram::{EngramLayout, hash_ids};
use ferroseek_ref::model::{validate_dims, validate_layout};
use ferroseek_ref::store::{CkptStore, DType, SyntheticWeights, WeightStore, bytes_to_f32};
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

/// TileLang's sparse kernel treats `SwaLengths` as a valid prefix length.
/// The reference keeps physical ring order with leading holes, so compact
/// only the GPU handoff; valid entries retain their oldest-first order.
fn packed_window_idxs(window: usize, pos: u64) -> Vec<i32> {
    let mut out: Vec<i32> = window_topk_idxs(window, pos)
        .into_iter()
        .filter(|&i| i >= 0)
        .collect();
    out.resize(window, -1);
    out
}

/// Row width baked into the generated sparse-MLA kernel's indexed-list
/// layout. DS4.1F uses 128 SWA entries plus 512 compressed entries.
const SPARSE_MLA_IDX_WIDTH: usize = 2048;

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
pub enum Sharded<'b> {
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

impl W8 {
    /// Test seam: the owned device buffer (None = streamed/host-backed).
    pub(crate) fn own_for_test(&self) -> Option<&D<u8>> {
        self._own.as_ref()
    }
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
    packed_s: Option<D<u32>>,
}

/// Packed fp4 expert weight + per-1x32-K ue8m0 scales.
pub struct Q4 {
    pub w: W8,
    pub s: W8,
}

struct GpuCompressor {
    /// compressor wkv dequantized to f32 (bf16 for ratio 1, f32/bf16 for
    /// ratio 2 — the ref's dense path decodes the same way)
    wkv: D<u16>,
    wgate: Option<D<u16>>,
    wkv_f32: Option<D<f32>>,
    wgate_f32: Option<D<f32>>,
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

pub struct GpuLayer {
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
    wo_a_fp8: Option<WoAFP8>,
    wo_b: Q8,
    gate_w: D<u16>,
    gate_bias: D<f32>,
    pub experts: Vec<[Q4; 3]>,
    /// Grouped-expert slabs (upload mode): contiguous [e_len] tables per
    /// weight kind for device-side dispatch (moe_grouped.cu). None in
    /// streaming mode (per-use staging keeps per-expert buffers).
    w1_all: Option<D<u8>>,
    w1s_all: Option<D<u8>>,
    w2_all: Option<D<u8>>,
    w2s_all: Option<D<u8>>,
    w3_all: Option<D<u8>>,
    w3s_all: Option<D<u8>>,
    w1_tl: Option<D<u8>>,
    w1s_tl: Option<D<u32>>,
    w2_tl: Option<D<u8>>,
    w2s_tl: Option<D<u32>>,
    w3_tl: Option<D<u8>>,
    w3s_tl: Option<D<u32>>,
    w1_map: Option<k::TensorMap>,
    w2_map: Option<k::TensorMap>,
    w3_map: Option<k::TensorMap>,
    shared_experts: [Q8; 3],
    shared_w13: Option<Q8>,
    compressor: Option<GpuCompressor>,
    indexer: Option<GpuIndexer>,
    k_owner: Option<GpuKOwner>,
    engram_w: Option<GpuEngram>,
}

struct GpuDrafter {
    layers: Vec<GpuLayer>,
    caches: GpuCaches,
    run: RunState,
    eng_cache: Vec<i64>,
    pos: u64,
    main_proj: Q8,
    main_norm: D<f32>,
    markov_embed: D<f32>,
    markov_head: D<f32>,
    norm: D<f32>,
}

struct EngramHostGather {
    ids: Vec<i64>,
    flat: Vec<f32>,
    weights: usize,
    weights_len: usize,
    scales: usize,
    scales_len: usize,
}

/// Owns every prefetch thread until its rows are consumed. Early returns
/// join the remaining workers before the weight store's mmap can be dropped.
struct EngramGatherWorkers(Vec<Option<std::thread::JoinHandle<EngramHostGather>>>);

impl EngramGatherWorkers {
    fn take(&mut self, index: usize) -> io::Result<EngramHostGather> {
        self.0[index]
            .take()
            .expect("engram gather consumed twice")
            .join()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "engram gather worker panicked"))
    }
}

impl Drop for EngramGatherWorkers {
    fn drop(&mut self) {
        for worker in self.0.iter_mut().filter_map(Option::take) {
            let _ = worker.join();
        }
    }
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
    attn_part: D<f32>,
    attn_stats: D<f32>,
    q_bf16: D<u16>,
    pos_i64: D<i64>,
    swa_len: D<i32>,
    idx_len: D<i32>,
    page_table: D<i32>,
    y_moe: D<f32>,
    gate_buf: D<f32>,
    up_buf: D<f32>,
    shared_out: D<f32>,
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
    dispatch_idx: D<i64>,
    order: D<i32>,
    layout_weights: D<f32>,
    route_pos: D<i32>,
    tile_expert: D<i32>,
    expert_count: D<i32>,
    expert_offset: D<i32>,
    expert_psum: D<i32>,
    num_tiles: D<i32>,
    route_h: D<f32>,
    route_q: D<u8>,
    route_q_s: D<u8>,
    route_y: D<f32>,
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
    kv_swa: Vec<D<u8>>,
    idx_records: Vec<D<u8>>,
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
    /// Debug: capture every MoE sublayer's output y (gate bisection).
    capture_y: Option<Vec<(u32, Vec<f32>)>>,
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
    kv_swa: Vec<D<u8>>,
    idx_records: Vec<D<u8>>,
    comp_score: Vec<D<f32>>,
    run: RunState,
    eng_cache: Vec<i64>,
    drafter_kv_all: Vec<D<f32>>,
}

struct SpecSnapshot {
    pos: u64,
    run: RunState,
    eng_cache: Vec<i64>,
    main_hiddens: Vec<D<f32>>,
    kv_all: Vec<(usize, usize, Vec<f32>)>,
    kv_swa: Vec<(usize, usize, Vec<u8>)>,
    idx_records: Vec<(usize, usize, Vec<u8>)>,
    index_k: Vec<D<f32>>,
    comp_kv: Vec<D<f32>>,
    comp_score: Vec<D<f32>>,
}

/// The GPU decode engine.
///
/// `'a` is the weight-store lifetime in streaming mode: the model borrows
/// the `Checkpoint` whose shard mmaps back the streamed weights. Upload
/// mode is `'static` (every weight is a device copy).
pub struct GpuModel<'a> {
    pub cfg: RefConfig,
    pub(crate) layers: Vec<GpuLayer>,
    caches: GpuCaches,
    engram: EngramLayout,
    eng_cache: Vec<i64>,
    token_map: Option<Vec<i64>>,
    /// [plain, compress] rope tables: [max_pos, rope_dim/2].
    rope_cos: [D<f32>; 2],
    rope_sin: [D<f32>; 2],
    emb: Option<D<f32>>,
    head: D<u16>,
    final_norm: D<f32>,
    glp_dirs: Vec<Option<D<f32>>>,
    glp_alpha: f32,
    scratch: Scratch,
    /// M-row scratch for `decode_m`, grown on first use (None until then).
    scratch_m: Option<ScratchM>,
    drafter_scratch_m: Option<ScratchM>,
    /// Publisher fan-out: for kv-source ordinal s, the layers attending its
    /// compressed rows (source included).
    bands: Vec<Vec<usize>>,
    run: RunState,
    pos: u64,
    spec_accept: usize,
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
    /// Drafter modules and captured target-layer hidden means.
    drafter: Option<GpuDrafter>,
    main_hiddens: Vec<D<f32>>,
    /// Prefix cache (serving): snapshots of per-request state keyed by
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

fn view_bf16(store: &mut dyn WeightStore, name: &str) -> io::Result<Vec<u16>> {
    let v = store.get(name)?;
    if v.dtype != DType::Bf16 {
        return ioerr(format!("{name}: expected bf16, got {:?}", v.dtype));
    }
    Ok(v.bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
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

pub(crate) fn pack_weight_scales(bytes: &[u8], n: usize, k: usize) -> Vec<u32> {
    let kb = k / 32;
    let words = k / 128;
    let mut out = vec![0u32; n * words];
    for row in 0..n {
        for w in 0..words {
            for j in 0..4 {
                out[row * words + w] |= (bytes[(row / 32) * kb + w * 4 + j] as u32) << (8 * j);
            }
        }
    }
    out
}
pub(crate) fn pack_expert_scales(bytes: &[u8], n: usize, k: usize) -> Vec<u32> {
    let kb = k / 32;
    let words = k / 128;
    let mut out = vec![0u32; n * words];
    for row in 0..n {
        for w in 0..words {
            for j in 0..4 {
                out[row * words + w] |= (bytes[row * kb + w * 4 + j] as u32) << (8 * j);
            }
        }
    }
    out
}

pub(crate) fn tile_major_expert(bytes: &[u8], experts: usize, n: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 128, 0);
    let k_tiles = k / 128;
    let mut out = vec![0u8; bytes.len()];
    for e in 0..experts {
        for nb in 0..n / 128 {
            for kb in 0..k_tiles {
                let dst = ((e * (n / 128) + nb) * k_tiles + kb) * 128 * 64;
                for row in 0..128 {
                    let src = (e * n + nb * 128 + row) * (k / 2) + kb * 64;
                    out[dst + row * 64..dst + row * 64 + 64].copy_from_slice(&bytes[src..src + 64]);
                }
            }
        }
    }
    out
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
            packed_s: None,
        })
    } else {
        // upload mode (synthetic gate): snapshot both slices to owned bytes
        // first — the store lends one view at a time
        let (wb, n, k) = {
            let v = store.get(wname)?;
            let wb = shard_weight(v.bytes, &v.shape, shard)?.as_slice().to_vec();
            let k = match shard {
                WShard::Cols { len, .. } => len,
                _ => v.shape[1],
            };
            let n = wb.len() / k;
            (wb, n, k)
        };
        let sb: Vec<u8> = {
            let v = store.get(sname)?;
            shard_scale(v.bytes, &v.shape, shard)?.as_slice().to_vec()
        };
        let packed = (k % 128 == 0).then(|| pack_weight_scales(&sb, n, k));
        Ok(Q8 {
            w: W8::device(up_u8(c, &wb)?),
            s: W8::device(up_u8(c, &sb)?),
            packed_s: packed
                .map(|packed| DeviceBuffer::from_slice(&packed).map_err(cuda_err))
                .transpose()?,
        })
    }
}

fn load_q8_combined_rows(
    store: &mut dyn WeightStore,
    c: &Cuda,
    wa: &str,
    sa: &str,
    wb: &str,
    sb: &str,
    shard: WShard,
) -> io::Result<Q8> {
    let load = |store: &mut dyn WeightStore, wname: &str, sname: &str| {
        let (w, n, k) = {
            let v = store.get(wname)?;
            let bytes = shard_weight(v.bytes, &v.shape, shard)?.as_slice().to_vec();
            let k = match shard {
                WShard::Cols { len, .. } => len,
                _ => v.shape[1],
            };
            let n = bytes.len() / k;
            (bytes, n, k)
        };
        let s = {
            let v = store.get(sname)?;
            shard_scale(v.bytes, &v.shape, shard)?.as_slice().to_vec()
        };
        Ok::<_, io::Error>((w, s, n, k))
    };
    let (w1, s1, n1, k1) = load(store, wa, sa)?;
    let (w3, s3, n3, k3) = load(store, wb, sb)?;
    if k1 != k3 {
        return ioerr("combined Q8 projections have different K dimensions".into());
    }
    let (n, k) = (n1 + n3, k1);
    let w = [w1, w3].concat();
    let s = [s1, s3].concat();
    let packed = pack_weight_scales(&s, n, k);
    Ok(Q8 {
        w: W8::device(up_u8(c, &w)?),
        s: W8::device(up_u8(c, &s)?),
        packed_s: Some(DeviceBuffer::from_slice(&packed).map_err(cuda_err)?),
    })
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

pub(crate) struct WoAFP8 {
    pub w: D<u8>,
    pub s: D<u8>,
    pub block: usize,
    pub scale_cols: usize,
}

/// wo_a: bf16 dense (converted export, spec §11) or fp8+scale (raw HF
/// export) — dequantize host-side to f32 either way, like the ref's dense
/// path. TP4 loads the rank's contiguous o-group block: rows
pub(crate) fn load_wo_a(
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

pub(crate) fn load_wo_a_fp8(
    store: &mut dyn WeightStore,
    c: &Cuda,
    name: &str,
    rows: usize,
    cols: usize,
    row_start: usize,
    row_len: usize,
) -> io::Result<Option<WoAFP8>> {
    let v = store.get(name)?;
    if v.dtype != DType::Fp8E4m3 {
        return Ok(None);
    }
    let full_rows = rows;
    let bytes = v.bytes[row_start * cols..(row_start + row_len) * cols].to_vec();
    drop(v);
    let sname = format!("{}.scale", name.trim_end_matches(".weight"));
    let sview = store.get(&sname)?;
    let sshape = sview.shape.clone();
    let sbytes_full = sview.bytes.to_vec();
    drop(sview);
    if sshape.len() != 2 {
        return ioerr(format!("{sname}: bad scale shape {sshape:?}"));
    }
    let block = full_rows / sshape[0];
    if block == 0 || block != cols / sshape[1] {
        return ioerr(format!(
            "{sname}: scale shape {sshape:?} does not tile [{full_rows}, {cols}]"
        ));
    }
    if row_start % block != 0 || row_len % block != 0 {
        return ioerr(format!(
            "{sname}: wo_a shard [{row_start}, +{row_len}) not {block}-aligned"
        ));
    }
    let scale_cols = sshape[1];
    let s0 = row_start / block;
    let s1 = s0 + row_len / block;
    let scales = sbytes_full[s0 * scale_cols..s1 * scale_cols].to_vec();
    Ok(Some(WoAFP8 {
        w: up_u8(c, &bytes)?,
        s: up_u8(c, &scales)?,
        block,
        scale_cols,
    }))
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
    let (p, n_routed) = if l >= cfg.n_layers as usize {
        (
            format!("mtp.{}", l - cfg.n_layers as usize),
            cfg.dspark_n_routed as usize,
        )
    } else {
        (format!("layers.{l}"), cfg.n_routed as usize)
    };
    let dim = cfg.dim as usize;
    let hd = cfg.head_dim as usize;
    let inh = cfg.index_n_heads as usize;
    let ihd = cfg.index_head_dim as usize;
    let ratio = if l >= cfg.n_layers as usize {
        0
    } else {
        cfg.compress_ratios[l]
    };
    // TP4 locals (spec §3): 16 heads, 2 o-groups, 8 index heads, 96 experts
    let world = sh.world_size as usize;
    let rank = sh.rank as usize;
    let heads_l = cfg.n_heads as usize / world;
    let og_l = cfg.o_groups as usize / world;
    let inh_l = inh / world;
    let e_start = rank * n_routed / world;
    let e_len = n_routed / world;
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
            let wkv_name = format!("{p}.attn.compressor.wkv.weight");
            let (wkv, wkv_f32) = if store.get(&wkv_name)?.dtype == DType::Bf16 {
                (up_u16(c, &view_bf16(store, &wkv_name)?)?, None)
            } else {
                (
                    DeviceBuffer::zeroed(0).map_err(cuda_err)?,
                    Some(up_f32(c, &view_f32(store, &wkv_name)?)?),
                )
            };
            let wgate_name = format!("{p}.attn.compressor.wgate.weight");
            let (wgate, wgate_f32) = if ratio > 1 {
                if store.get(&wgate_name)?.dtype == DType::Bf16 {
                    (Some(up_u16(c, &view_bf16(store, &wgate_name)?)?), None)
                } else {
                    (None, Some(up_f32(c, &view_f32(store, &wgate_name)?)?))
                }
            } else {
                (None, None)
            };
            Some(GpuCompressor {
                wkv,
                wgate,
                wkv_f32,
                wgate_f32,
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
                // rows [rank*inh_l, +inh_l) of [inh*ihd, ql]; at TP1 the
                // slice is the whole tensor and Rows-vs-All produced a
                // synthetic-gate divergence despite byte-identity (the
                // probe test documents the open question) — pass All when unsharded
                if world == 1 {
                    WShard::All
                } else {
                    WShard::Rows {
                        start: rank * inh_l * ihd,
                        len: inh_l * ihd,
                    }
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
    let mut w1_tl: Option<D<u8>> = None;
    let mut w1s_tl: Option<D<u32>> = None;
    let mut w2_tl: Option<D<u8>> = None;
    let mut w2s_tl: Option<D<u32>> = None;
    let mut w3_tl: Option<D<u8>> = None;
    let mut w3s_tl: Option<D<u32>> = None;
    let mut w1_map = None;
    let mut w2_map = None;
    let mut w3_map = None;
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
        w1_tl = Some(up_u8(c, &tile_major_expert(&b1, e_len, inter, dim))?);
        w1s_tl = Some(
            DeviceBuffer::from_slice(&pack_expert_scales(&b1s, e_len * inter, dim))
                .map_err(cuda_err)?,
        );
        w2_tl = Some(up_u8(c, &tile_major_expert(&b2, e_len, dim, inter))?);
        w2s_tl = Some(
            DeviceBuffer::from_slice(&pack_expert_scales(&b2s, e_len * dim, inter))
                .map_err(cuda_err)?,
        );
        w3_tl = Some(up_u8(c, &tile_major_expert(&b3, e_len, inter, dim))?);
        w3s_tl = Some(
            DeviceBuffer::from_slice(&pack_expert_scales(&b3s, e_len * inter, dim))
                .map_err(cuda_err)?,
        );
        w1_map = Some(k::tensor_map_fp4(k::base(w1_tl.as_ref().unwrap()), 69120));
        w2_map = Some(k::tensor_map_fp4(k::base(w2_tl.as_ref().unwrap()), 69120));
        w3_map = Some(k::tensor_map_fp4(k::base(w3_tl.as_ref().unwrap()), 69120));
        eprintln!(
            "build: layer {l} experts fused into grouped slabs ({} x 3, {:.1} GB)",
            e_len,
            (b1.len() + b2.len() + b3.len()) as f64 / (1u64 << 30) as f64
        );
    }
    if w1_all.is_none() && w1_tl.is_none() {
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
    let wo_a_fp8 = load_wo_a_fp8(
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
        let n = n_routed * dim;
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
    let shared_w13 = if arena.is_none() {
        Some(load_q8_combined_rows(
            store,
            c,
            &format!("{p}.ffn.shared_experts.w1.weight"),
            &format!("{p}.ffn.shared_experts.w1.scale"),
            &format!("{p}.ffn.shared_experts.w3.weight"),
            &format!("{p}.ffn.shared_experts.w3.scale"),
            WShard::All,
        )?)
    } else {
        None
    };

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
        hc_ffn_base,
        hc_attn_base,
        hc_attn_scale,
        hc_ffn_scale,
        wq_a,
        q_norm_w,
        wq_b,
        wkv,
        kv_norm_w,
        attn_sink,
        wo_a,
        wo_a_fp8,
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
        w1_tl,
        w1s_tl,
        w2_tl,
        w2s_tl,
        w3_tl,
        w3s_tl,
        w1_map,
        w2_map,
        w3_map,
        shared_experts,
        shared_w13,
        compressor,
        indexer,
        k_owner,
        engram_w,
    };
    Ok(layer)
}

// ---------------------------------------------------------------------------
// decode

impl<'a> GpuModel<'a> {
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
        for buf in self.caches.kv_swa.iter_mut() {
            let _ = buf.set_zero();
        }
        for buf in self.caches.idx_records.iter_mut() {
            let _ = buf.set_zero();
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
        if let Some(drafter) = self.drafter.as_mut() {
            for buf in drafter.caches.kv_all.iter_mut() {
                zero(buf)?;
            }
            drafter.eng_cache.clear();
            drafter.pos = 0;
            drafter.run.shared_n_sel = 0;
        }
        self.main_hiddens.clear();
        self.eng_cache.clear();
        let n_sources = self.run.compress_len.len();
        self.run = RunState {
            compress_len: vec![0; n_sources],
            index_k_src: 0,
            candidates: Vec::new(),
            shared_n_sel: 0,
            capture_y: None,
            capture_x: None,
        };
        self.pos = 0;
        self.spec_accept = 0;
        Ok(())
    }

    fn spec_snapshot(&self, m: usize) -> io::Result<SpecSnapshot> {
        let clone_buf = |b: &D<f32>| -> io::Result<D<f32>> {
            let mut out = unsafe { DeviceBuffer::uninitialized(b.len()) }.map_err(cuda_err)?;
            out.copy_from(b).map_err(cuda_err)?;
            Ok(out)
        };
        let win = self.cfg.window_size as usize;
        let hd = self.cfg.head_dim as usize;
        let mut kv_all = Vec::new();
        let mut kv_swa = Vec::new();
        let mut idx_records = Vec::new();
        for (l, layer) in self.layers.iter().enumerate() {
            for i in 0..m {
                let slot = ((self.pos + i as u64) % win as u64) as usize;
                let row = slot * hd;
                let mut values = vec![0.0f32; hd];
                self.caches.kv_all[l]
                    .index(row..row + hd)
                    .copy_to(&mut values)
                    .map_err(cuda_err)?;
                kv_all.push((l, row, values));
                let byte_row = slot * 528;
                let mut packed = vec![0u8; 528];
                self.caches.kv_swa[l]
                    .index(byte_row..byte_row + 528)
                    .copy_to(&mut packed)
                    .map_err(cuda_err)?;
                kv_swa.push((l, byte_row, packed));
            }
            let ratio = layer.ratio as u64;
            if ratio > 0 {
                let first = ((self.pos + 1) / ratio).saturating_sub(1) as usize;
                let last = ((self.pos + m as u64) / ratio).saturating_sub(1) as usize;
                for j in first..=last {
                    let row = (win + j) * hd;
                    let mut values = vec![0.0f32; hd];
                    self.caches.kv_all[l]
                        .index(row..row + hd)
                        .copy_to(&mut values)
                        .map_err(cuda_err)?;
                    kv_all.push((l, row, values));
                    let byte_row = j * 288;
                    let mut packed = vec![0u8; 288];
                    self.caches.idx_records[l]
                        .index(byte_row..byte_row + 288)
                        .copy_to(&mut packed)
                        .map_err(cuda_err)?;
                    idx_records.push((l, byte_row, packed));
                }
            }
        }
        let mut main_hiddens = Vec::with_capacity(self.main_hiddens.len());
        for b in &self.main_hiddens {
            main_hiddens.push(clone_buf(b)?);
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
        Ok(SpecSnapshot {
            pos: self.pos,
            run: self.run.clone(),
            eng_cache: self.eng_cache.clone(),
            main_hiddens,
            kv_all,
            kv_swa,
            idx_records,
            index_k,
            comp_kv,
            comp_score,
        })
    }

    fn spec_restore(&mut self, snap: &SpecSnapshot) -> io::Result<()> {
        for (l, row, values) in &snap.kv_all {
            self.caches.kv_all[*l]
                .index(*row..*row + values.len())
                .copy_from(values)
                .map_err(cuda_err)?;
        }
        for (l, row, values) in &snap.kv_swa {
            self.caches.kv_swa[*l]
                .index(*row..*row + values.len())
                .copy_from(values)
                .map_err(cuda_err)?;
        }
        for (l, row, values) in &snap.idx_records {
            self.caches.idx_records[*l]
                .index(*row..*row + values.len())
                .copy_from(values)
                .map_err(cuda_err)?;
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
        self.main_hiddens.clear();
        for src in &snap.main_hiddens {
            let mut dst = unsafe { DeviceBuffer::uninitialized(src.len()) }.map_err(cuda_err)?;
            dst.copy_from(src).map_err(cuda_err)?;
            self.main_hiddens.push(dst);
        }
        self.run = snap.run.clone();
        self.eng_cache.clone_from(&snap.eng_cache);
        self.pos = snap.pos;
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
        let mut kv_swa = Vec::with_capacity(self.caches.kv_swa.len());
        let mut idx_records = Vec::with_capacity(self.caches.idx_records.len());
        let clone_u8 = |b: &D<u8>| -> io::Result<D<u8>> {
            let mut out = unsafe { DeviceBuffer::uninitialized(b.len()) }.map_err(cuda_err)?;
            out.copy_from(b).map_err(cuda_err)?;
            Ok(out)
        };
        for b in &self.caches.kv_swa {
            kv_swa.push(clone_u8(b)?);
        }
        for b in &self.caches.idx_records {
            idx_records.push(clone_u8(b)?);
        }
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
        let mut drafter_kv_all = Vec::new();
        if let Some(drafter) = self.drafter.as_ref() {
            for b in &drafter.caches.kv_all {
                drafter_kv_all.push(clone_buf(b)?);
            }
        }
        Ok(CacheSnapshot {
            pos: self.pos,
            kv_all,
            index_k,
            comp_kv,
            kv_swa,
            idx_records,
            comp_score,
            run: self.run.clone(),
            eng_cache: self.eng_cache.clone(),
            drafter_kv_all,
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
        for (dst, src) in self.caches.kv_swa.iter_mut().zip(&snap.kv_swa) {
            dst.copy_from(src).map_err(cuda_err)?;
        }
        for (dst, src) in self.caches.idx_records.iter_mut().zip(&snap.idx_records) {
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
        if let Some(drafter) = self.drafter.as_mut() {
            for (dst, src) in drafter.caches.kv_all.iter_mut().zip(&snap.drafter_kv_all) {
                dst.copy_from(src).map_err(cuda_err)?;
            }
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
        // Sequential prefill is the correctness default: real-checkpoint
        // greedy streams diverge from decode_m, so M-batch is opt-in while
        // that path lacks an end-to-end identity proof.
        let mut n = start as usize;
        while n < ids.len() {
            let win = self.cfg.window_size as usize;
            let room = win - (self.pos as usize % win);
            let chunk = (ids.len() - n).min(room);
            let r = if self.drafter.is_some()
                || std::env::var_os("FERROSEEK_MBATCH_PREFILL").is_none()
            {
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "sequential prefill",
                ))
            } else {
                self.decode_m(&ids[n..n + chunk])
            };
            match r {
                Ok(rows) => {
                    logits = rows.into_iter().last().unwrap_or_default();
                    n += chunk;
                }
                Err(e) if e.kind() == io::ErrorKind::Unsupported => {
                    for (m, &t) in ids.iter().enumerate().skip(n) {
                        logits = self.decode(t, m as u64)?;
                    }
                    n = ids.len();
                }
                Err(e) => return Err(e),
            }
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

    /// Debug: capture every MoE sublayer's output y.
    pub fn debug_capture_y(&mut self, on: bool) {
        self.run.capture_y = if on { Some(Vec::new()) } else { None };
    }

    /// Debug: take the captured (layer, y) pairs.
    pub fn debug_take_y(&mut self) -> Vec<(u32, Vec<f32>)> {
        self.run.capture_y.take().unwrap_or_default()
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
        self.main_hiddens.clear();
        let capture_drafter = self.drafter.is_some();
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
            hash_ids(
                &self.cfg,
                &self.engram,
                &mut self.eng_cache,
                self.token_map.as_deref(),
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
            glp_dirs,
            glp_alpha,
            bands,
            run,
            engram,
            arena,
            sharding,
            comm,
            main_hiddens,
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
            let widxs = packed_window_idxs(win, position);
            s.win_idxs.copy_from(&widxs).map_err(cuda_err)?;
        }

        // Start both Engram tables' independent random-row reads together.
        // Layer 14's gather otherwise sits idle until layer 1's gather and
        // thirteen decoder layers have completed. Only host-mmapped tables
        // participate; device-resident synthetic tables keep the GPU gather.
        let ncols = engram.n_hash_cols;
        let ehd = engram.head_dim;
        let engram_gathers: Vec<Option<EngramHostGather>> = (0..hashes.len())
            .map(|ei| {
                let layer = layers.iter().find(|layer| layer.engram == Some(ei))?;
                let e = layer.engram_w.as_ref()?;
                match (e.embed_w.host(), e.embed_s.host()) {
                    (Some((wp, wlen)), Some((sp, slen))) => Some(EngramHostGather {
                        ids: hashes[ei].iter().map(|&h| h as i64).collect(),
                        flat: vec![0.0; ncols * ehd],
                        weights: wp as usize,
                        weights_len: wlen,
                        scales: sp as usize,
                        scales_len: slen,
                    }),
                    _ => None,
                }
            })
            .collect();
        let mut engram_gathers = EngramGatherWorkers(
            engram_gathers
                .into_iter()
                .map(|gather| {
                    gather.map(|mut gather| {
                        std::thread::spawn(move || {
                            // SAFETY: these alias the weight store's mmap.
                            // EngramGatherWorkers joins on every return path.
                            let (wt, st) = unsafe {
                                (
                                    std::slice::from_raw_parts(
                                        gather.weights as *const u8,
                                        gather.weights_len,
                                    ),
                                    std::slice::from_raw_parts(
                                        gather.scales as *const u8,
                                        gather.scales_len,
                                    ),
                                )
                            };
                            gather
                                .flat
                                .par_chunks_mut(ehd)
                                .zip(gather.ids.par_iter())
                                .for_each(|(chunk, &id)| {
                                    let row = id as usize;
                                    if row * ehd + ehd > wt.len() {
                                        chunk.fill(f32::NAN);
                                        return;
                                    }
                                    chunk.copy_from_slice(&ferroseek_ref::store::fp8_row_scaled(
                                        wt, st, row, ehd,
                                    ));
                                });
                            gather
                        })
                    })
                })
                .collect(),
        );

        let mut h_cur = 0usize;
        for li in 0..layers.len() {
            let layer = &layers[li];
            if let Some(d) = &glp_dirs[layer.layer as usize] {
                k::glp_project(
                    c,
                    k::base(&s.h[h_cur]),
                    k::base(d),
                    k::base(&s.h[h_cur]),
                    *glp_alpha,
                    hc,
                    dim,
                );
            }
            if capture_drafter {
                Self::capture_main_hidden(main_hiddens, c, cfg, &s.h[h_cur], 0, layer.layer)?;
            }

            // engram merge before the block (model.py:1264-1266)
            let t_eng = Instant::now();
            if let Some(ei) = layer.engram {
                let e = layer.engram_w.as_ref().unwrap();
                let ncols = engram.n_hash_cols;
                let ehd = engram.head_dim;
                // the real checkpoint's engram embed table is a 98 GB fp8
                // slab in the last shards — never registered, so gather its
                // n_hash_cols rows host-side with the ref engine's own
                // fp8_row_scaled (bit-identical dequant, model.py:328-365).
                // Registered tables (synthetic gate) keep the device gather.
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
                        let gather = engram_gathers.take(ei)?;
                        let ids = &gather.ids;
                        let flat = &gather.flat;
                        if flat.iter().any(|v| v.is_nan()) {
                            // distinguish: a row id beyond the table (fill
                            // branch) vs NaN inside a dequantized valid row
                            let bad_row = ids
                                .iter()
                                .take(ncols)
                                .map(|&id| id as usize)
                                .find(|&row| row * ehd + ehd > wt.len());
                            let nan_idx = flat.iter().position(|v| v.is_nan()).unwrap();
                            let nan_row = nan_idx / ehd;
                            let rid = ids.get(nan_row).copied().unwrap_or(i64::MAX) as usize;
                            let s_cols = ehd.div_ceil(32);
                            let sbyte = st.get(rid * s_cols).copied().unwrap_or(u8::MAX);
                            let wbytes: Vec<u8> = wt
                                .get(rid * ehd..rid * ehd + 8)
                                .map(|s| s.to_vec())
                                .unwrap_or_default();
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!(
                                    "engram gather NaN: bad_id={bad_row:?} nan_slot={nan_row} row_id={rid} table_rows={} wt={} ehd={ehd} scale_byte={sbyte:#04x} w={wbytes:02x?} ids={ids:?}",
                                    wt.len() / ehd,
                                    wt.len(),
                                ),
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
                        dump_dev(
                            c,
                            &s.eflat,
                            ncols * ehd,
                            format!("d_pos{}_eflat{}.f32", crate::dump::pos(), layer.layer),
                        );
                        if crate::dump::enabled() {
                            crate::dump::f32s(
                                format!("d_pos{}_hashids{}.f32", crate::dump::pos(), layer.layer),
                                &ids.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
                            );
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
                k::emit8_packed(
                    c,
                    k::base(&s.eflat),
                    k::base(&s.act_e),
                    k::base(&s.act_e_s),
                    ncols * ehd,
                );
                gemm_q8(
                    c,
                    &e.wkv,
                    k::base(&s.act_e),
                    k::base(&s.act_e_s),
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
            let attn_out = attention_step(
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
            if let Some(d) = glp_dirs
                .get(layer.layer as usize + 1)
                .and_then(|d| d.as_ref())
            {
                k::glp_project(c, attn_out, k::base(d), attn_out, *glp_alpha, 1, dim);
            }
            prof.add_us(0, t_att);

            {
                let h_next = 1 - h_cur;
                k::hc_post(
                    c,
                    attn_out,
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
                Some(run),
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
        if let Some(d) = &glp_dirs[layers.len() - 1] {
            k::glp_project(
                c,
                k::base(&s.h[h_cur]),
                k::base(d),
                k::base(&s.h[h_cur]),
                *glp_alpha,
                hc,
                dim,
            );
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
        k::bf16_gemv(
            c,
            k::base(head),
            k::base(&s.x),
            k::base(&s.logits),
            vocab_l,
            dim,
        );
        let out = if let Some(comm) = comm_opt.as_deref_mut() {
            let stream = c.stream.as_inner();
            s.coll_stage.dtoh(
                s.logits.as_device_ptr().as_raw() as u64,
                vocab_l * 4,
                stream,
            )?;
            s.coll_stage.sync(stream)?;
            let full = comm.allgather(&s.coll_stage.as_f32()[..vocab_l])?;
            // Keep device logits current for argmax_current, but return the
            // host vector we already own instead of a second full D2H copy.
            s.logits.copy_from(&full).map_err(cuda_err)?;
            full
        } else {
            let mut out = vec![0.0f32; cfg.vocab as usize];
            s.logits.copy_to(&mut out).map_err(cuda_err)?;
            out
        };
        prof.steps += 1;
        prof.add_us(4, t_head);
        self.seed_drafter(c, position)?;
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
        let mut model = Self::build(rcfg, store.as_mut(), 1024, None, &Sharding::default())?;
        // the engram embed tables are host-backed: raw pointers into the
        // store's tensors. The store MUST outlive the model — dropping it
        // here left those pointers dangling (use-after-free; read as
        // garbage only once the heap recycled, i.e. only in later tests
        // of a shared process).
        model.store_keepalive = Some(store);
        Ok(model)
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

        let glp = ferroseek_core::glp::GlpVector::from_env(n_layers, dim, None)?;
        let glp_alpha = glp.as_ref().map(|g| g.alpha).unwrap_or(0.0);
        let glp_dirs = (0..n_layers)
            .map(|layer| {
                glp.as_ref()
                    .and_then(|g| g.dirs.get(&(layer as u32)))
                    .map(|v| up_f32(c, v))
                    .transpose()
            })
            .collect::<io::Result<Vec<_>>>()?;

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

        let n_mtp = cfg.n_mtp_layers as usize;
        let drafter = if n_mtp > 0 {
            let mut drafter_layers = Vec::with_capacity(n_mtp);
            let mut draft_kv = Vec::with_capacity(n_layers + n_mtp);
            for l in 0..n_layers + n_mtp {
                if l < n_layers {
                    draft_kv.push(DeviceBuffer::zeroed(0).map_err(cuda_err)?);
                    continue;
                }
                drafter_layers.push(upload_layer(&cfg, store, c, l, None, arena, sh)?);
                // Five transient draft rows follow the persistent main-KV window.
                draft_kv.push(up_zero_f32(c, (win + 5) * hd)?);
            }
            Some(GpuDrafter {
                layers: drafter_layers,
                caches: GpuCaches {
                    kv_all: draft_kv,
                    kv_swa: Vec::new(),
                    idx_records: Vec::new(),
                    index_k: Vec::new(),
                    comp_kv: Vec::new(),
                    comp_score: Vec::new(),
                },
                run: RunState {
                    compress_len: Vec::new(),
                    index_k_src: 0,
                    candidates: Vec::new(),
                    shared_n_sel: 0,
                    capture_x: None,
                    capture_y: None,
                },
                eng_cache: Vec::new(),
                pos: 0,
                main_proj: load_q8(
                    store,
                    c,
                    arena,
                    "mtp.0.main_proj.weight",
                    "mtp.0.main_proj.scale",
                    WShard::All,
                )?,
                main_norm: up_f32(c, &view_f32(store, "mtp.0.main_norm.weight")?)?,
                markov_embed: up_f32(
                    c,
                    &view_f32(
                        store,
                        &format!("mtp.{}.markov_head.embed.weight", n_mtp - 1),
                    )?,
                )?,
                markov_head: {
                    let v = store.get(&format!("mtp.{}.markov_head.head.weight", n_mtp - 1))?;
                    let row_bytes = 256 * dtype_size(v.dtype);
                    let vl = cfg.vocab as usize / sh.world_size as usize;
                    let r0 = sh.rank as usize * vl;
                    if v.dtype != DType::Bf16 || (r0 + vl) * row_bytes > v.bytes.len() {
                        return ioerr("invalid DSpark Markov head shard".into());
                    }
                    let host: Vec<f32> = v.bytes[r0 * row_bytes..(r0 + vl) * row_bytes]
                        .chunks_exact(2)
                        .map(|b| {
                            ferroseek_core::quant::bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))
                        })
                        .collect();
                    drop(v);
                    up_f32(c, &host)?
                },
                norm: up_f32(
                    c,
                    &view_f32(store, &format!("mtp.{}.norm.weight", n_mtp - 1))?,
                )?,
            })
        } else {
            None
        };

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
            let host: Vec<u16> = v.bytes[r0 * row_bytes..(r0 + vl) * row_bytes]
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            drop(v);
            up_u16(c, &host)?
        };
        let final_norm = up_f32(c, &view_f32(store, "norm.weight")?)?;
        eprintln!("build: embed/head/norm in {:?}", t_head.elapsed());

        // caches: per layer kv_all = ring + compressed view; per source
        // index_k + compressor state
        let t_caches = std::time::Instant::now();
        let mut kv_all = Vec::with_capacity(n_layers);
        let mut kv_swa = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            kv_swa.push(DeviceBuffer::zeroed(win * 528).map_err(cuda_err)?);
        }
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
        let mut idx_records = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ratio = cfg.compress_ratios[l] as u64;
            idx_records.push(
                DeviceBuffer::zeroed(if ratio > 0 {
                    (max_seq / ratio + 1) as usize * 288
                } else {
                    288
                })
                .map_err(cuda_err)?,
            );
        }
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
        let page_table_host: Vec<i32> =
            (0..(cfg.original_seq_len as usize / 64 + 1) as i32).collect();
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
            let raw = n / 32;
            let packed_offset = raw.next_multiple_of(16);
            Ok((
                DeviceBuffer::zeroed(n).map_err(cuda_err)?,
                DeviceBuffer::zeroed(packed_offset + raw).map_err(cuda_err)?,
            ))
        };
        let (act_x, act_x_s) = act(dim)?;
        let (act_qr, act_qr_s) = act(cfg.q_lora as usize)?;
        let (act_fl, act_fl_s) = act(cfg.o_groups as usize * cfg.o_lora as usize)?;
        let (act_h, act_h_s) = act(cfg.inter_dim as usize)?;
        let (act_e, act_e_s) = act(eflat_len.max(32))?;
        let dispatch_tiles =
            16 * cfg.n_activated as usize + cfg.n_routed as usize / sh.world_size as usize;
        let scratch = Scratch {
            h: [up_zero_f32(c, hc * dim)?, up_zero_f32(c, hc * dim)?],
            x: up_zero_f32(c, dim)?,
            qr: up_zero_f32(c, cfg.q_lora as usize)?,
            q: up_zero_f32(c, cfg.n_heads as usize * hd)?,
            kvrow: up_zero_f32(c, hd)?,
            o: up_zero_f32(c, cfg.n_heads as usize * hd)?,
            fl: up_zero_f32(c, cfg.o_groups as usize * cfg.o_lora as usize)?,
            attn_out: up_zero_f32(c, dim)?,
            attn_part: up_zero_f32(
                c,
                ((win + cfg.index_topk as usize).div_ceil(128)) * cfg.n_heads as usize * hd,
            )?,
            attn_stats: up_zero_f32(
                c,
                ((win + cfg.index_topk as usize).div_ceil(128)) * cfg.n_heads as usize * 2,
            )?,
            q_bf16: DeviceBuffer::zeroed(cfg.n_heads as usize * hd).map_err(cuda_err)?,
            pos_i64: DeviceBuffer::zeroed(1).map_err(cuda_err)?,
            swa_len: DeviceBuffer::zeroed(1).map_err(cuda_err)?,
            idx_len: DeviceBuffer::zeroed(1).map_err(cuda_err)?,
            page_table: DeviceBuffer::from_slice(&page_table_host).map_err(cuda_err)?,
            y_moe: up_zero_f32(c, dim)?,
            gate_buf: up_zero_f32(c, cfg.inter_dim as usize)?,
            up_buf: up_zero_f32(c, cfg.inter_dim as usize)?,
            shared_out: up_zero_f32(c, 2 * cfg.inter_dim as usize)?,
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
            dispatch_idx: DeviceBuffer::zeroed(16 * cfg.n_activated as usize).map_err(cuda_err)?,
            order: DeviceBuffer::zeroed(dispatch_tiles * 32).map_err(cuda_err)?,
            layout_weights: up_zero_f32(c, dispatch_tiles * 32)?,
            route_pos: DeviceBuffer::zeroed(16 * cfg.n_activated as usize).map_err(cuda_err)?,
            tile_expert: DeviceBuffer::zeroed(dispatch_tiles).map_err(cuda_err)?,
            expert_count: DeviceBuffer::zeroed(cfg.n_routed as usize / sh.world_size as usize)
                .map_err(cuda_err)?,
            expert_offset: DeviceBuffer::zeroed(cfg.n_routed as usize / sh.world_size as usize)
                .map_err(cuda_err)?,
            expert_psum: DeviceBuffer::zeroed(cfg.n_routed as usize / sh.world_size as usize)
                .map_err(cuda_err)?,
            num_tiles: DeviceBuffer::zeroed(1).map_err(cuda_err)?,
            route_h: up_zero_f32(c, dispatch_tiles * 32 * cfg.inter_dim as usize)?,
            route_q: DeviceBuffer::zeroed(dispatch_tiles * 32 * cfg.inter_dim as usize)
                .map_err(cuda_err)?,
            route_q_s: DeviceBuffer::zeroed(
                (dispatch_tiles * cfg.inter_dim as usize).next_multiple_of(16)
                    + dispatch_tiles * cfg.inter_dim as usize,
            )
            .map_err(cuda_err)?,
            route_y: up_zero_f32(c, dispatch_tiles * 32 * dim)?,
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
            gate_w_out: up_zero_f32(c, 16 * cfg.n_activated as usize)?,
            gate_scores: up_zero_f32(c, cfg.n_routed as usize)?,
            hc_dots: up_zero_f32(c, cfg.mix_hc() as usize)?,
            gate_i_out: DeviceBuffer::zeroed(16 * cfg.n_activated as usize).map_err(cuda_err)?,
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
                score_cap
                    .max(cfg.vocab as usize / sh.world_size as usize)
                    .max(cfg.dim as usize)
                    * 4
                    + 4,
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
                kv_swa,
                idx_records,
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
            glp_dirs,
            glp_alpha,
            scratch,
            scratch_m: None,
            drafter_scratch_m: None,
            bands,
            run: RunState {
                compress_len: vec![0; n_sources],
                index_k_src: 0,
                candidates: Vec::new(),
                shared_n_sel: 0,
                capture_x: None,
                capture_y: None,
            },
            pos: 0,
            spec_accept: 0,
            max_seq,
            sharding: *sh,
            comm: None,
            prof: Prof::from_env(),
            drafter,
            main_hiddens: Vec::new(),
            prefix_cache: Vec::new(),
            arena: None,
            store_keepalive: None,
        })
    }
}
fn compressor_gemv(
    c: &Cuda,
    w: &D<u16>,
    wf: Option<&D<f32>>,
    x: k::F32,
    y: k::F32,
    rows: usize,
    cols: usize,
) {
    if let Some(wf) = wf {
        k::f32_gemv(c, k::base(wf), x, y, rows, cols);
    } else {
        k::bf16_gemv(c, k::base(w), x, y, rows, cols);
    }
}

fn gemm_q8(c: &Cuda, q: &Q8, a: k::U8, sa: k::U8, out: k::F32, m: usize, n: usize, k: usize) {
    gemm_q8_on(c, q, a, sa, out, m, n, k, &c.stream)
}

fn gemm_q8_on(
    c: &Cuda,
    q: &Q8,
    a: k::U8,
    sa: k::U8,
    out: k::F32,
    m: usize,
    n: usize,
    k: usize,
    stream: &Stream,
) {
    let kind = match (n, k) {
        (1280, 5120) => Some(k::TlGemm::Qa),
        (8192, 1280) => Some(k::TlGemm::Qb),
        (512, 5120) => Some(k::TlGemm::Wkv),
        (5120, 2048) => Some(k::TlGemm::WoB),
        (2304, 5120) => Some(k::TlGemm::SharedW13),
        (4608, 5120) => Some(k::TlGemm::SharedW13Fused),
        (5120, 2304) => Some(k::TlGemm::SharedW2),
        (25600, 5120) => Some(k::TlGemm::EngramWkv),
        (1280, 15360) => Some(k::TlGemm::DrafterMainProj),
        _ => None,
    };
    if m <= 16 {
        if let (Some(kind), Some(packed)) = (kind, q.packed_s.as_ref()) {
            let raw_scales = m * k / 32;
            let packed_offset = raw_scales.next_multiple_of(16);
            let packed_a = k::U32::from_raw(sa.as_raw() + packed_offset as u64);
            k::fp8_gemm_tl_on(
                c,
                kind,
                a,
                packed_a,
                q.w.ptr(),
                k::base(packed),
                out,
                m,
                stream,
            );
            return;
        }
    }
    k::fp8_gemm_on(c, a, sa, q.w.ptr(), q.s.ptr(), out, m, n, k, stream);
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
) -> io::Result<k::F32> {
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
    k::emit8_packed(
        c,
        k::base(&s.x),
        k::base(&s.act_x),
        k::base(&s.act_x_s),
        dim,
    );
    dump_u8(c, &s.act_x, dim, format!("d_pos{pos}_actx{l}.u8"));
    dump_u8(c, &s.act_x_s, dim / 32, format!("d_pos{pos}_actxs{l}.u8"));
    gemm_q8(
        c,
        &layer.wq_a,
        k::base(&s.act_x),
        k::base(&s.act_x_s),
        k::base(&s.qr),
        1,
        cfg.q_lora as usize,
        dim,
    );
    let activation_event = cust::event::Event::new(cust::event::EventFlags::DEFAULT).unwrap();
    activation_event.record(&c.stream).unwrap();
    c.aux_stream
        .wait_event(
            activation_event,
            cust::stream::StreamWaitEventFlags::DEFAULT,
        )
        .unwrap();
    let wkv_event = cust::event::Event::new(cust::event::EventFlags::DEFAULT).unwrap();
    gemm_q8_on(
        c,
        &layer.wkv,
        k::base(&s.act_x),
        k::base(&s.act_x_s),
        k::base(&s.kvrow),
        1,
        hd,
        dim,
        &c.aux_stream,
    );
    wkv_event.record(&c.aux_stream).unwrap();
    k::rmsnorm(
        c,
        k::base(&s.qr),
        k::base(&layer.q_norm_w),
        k::base(&s.qr),
        cfg.q_lora as usize,
        eps,
    );
    k::emit8_packed(
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
    gemm_q8(
        c,
        &layer.wq_b,
        k::base(&s.act_qr),
        k::base(&s.act_qr_s),
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
    c.stream
        .wait_event(wkv_event, cust::stream::StreamWaitEventFlags::DEFAULT)
        .unwrap();
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
    let packed = caches.kv_swa[l]
        .index(slot * 528..(slot + 1) * 528)
        .as_device_ptr();
    k::emit8(c, ring, packed, k::U8::from_raw(packed.as_raw() + 512), hd);

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
                compressor_gemv(
                    c,
                    &comp.wkv,
                    comp.wkv_f32.as_ref(),
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
                compressor_gemv(
                    c,
                    &comp.wkv,
                    comp.wkv_f32.as_ref(),
                    k::base(&s.x),
                    k::row(kv_state, slot_i * hd, hd),
                    hd,
                    dim,
                );
                compressor_gemv(
                    c,
                    comp.wgate.as_ref().unwrap_or(&comp.wkv),
                    comp.wgate_f32.as_ref(),
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
            if compress_len > cfg.index_topk as usize {
                // query side: wq_b(qr) at pos, fp4(32, e8m0) snap
                gemm_q8(
                    c,
                    &layer.indexer.as_ref().unwrap().wq_b,
                    k::base(&s.act_qr),
                    k::base(&s.act_qr_s),
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
            }

            // score + selection
            let mut idxs_host = packed_window_idxs(win, pos);
            if compress_len <= cfg.index_topk as usize {
                // topk >= every score: selection is always position order,
                idxs_host.extend((win as i32)..(win + compress_len) as i32);
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
                let cmp = select_topk(&score, cfg.index_topk as usize, compress_len, win as i32);
                idxs_host.extend(cmp);
            }
            // partial upload: the buffer is sized for the worst case, the
            // kernel reads only [0, n_sel)
            s.shared_idxs
                .index(0..idxs_host.len())
                .copy_from(&idxs_host)
                .map_err(cuda_err)?;
            run.shared_n_sel = idxs_host.len();
            crate::dump::i32s(format!("d_pos{pos}_idxsHost{l}.i32"), &idxs_host);
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
                let rec = caches.idx_records[l]
                    .index(j * 288..(j + 1) * 288)
                    .as_device_ptr();
                k::fp4_emit(
                    c,
                    k::base(&s.latent),
                    rec,
                    k::U8::from_raw(rec.as_raw() + 256),
                    hd,
                    16,
                );
                for &cl in &bands[src] {
                    if cl != src {
                        let dst = caches.idx_records[cl]
                            .index(j * 288..(j + 1) * 288)
                            .as_device_ptr();
                        k::copy_u8(c, rec, dst, 288);
                    }
                }
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
    if std::env::var_os("FERROSEEK_SPARSE_FP32").is_some() || heads != 16 || hd != 512 {
        if n_sel >= 256 {
            k::sparse_attn_split(
                c,
                k::base(&s.q),
                k::base(&caches.kv_all[l]),
                idxs,
                k::base(&layer.attn_sink),
                k::base(&s.attn_part),
                k::base(&s.attn_stats),
                k::base(&s.o),
                heads,
                hd,
                n_sel,
                (hd as f32).sqrt().recip(),
            );
        } else {
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
        }
        dump_dev(c, &s.o, heads * hd, format!("d_pos{pos}_attnraw{l}.f32"));
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
    } else {
        let filled = ((pos + 1).min(win as u64)) as i32;
        let compress_len = if layer.ratio > 0 && run.shared_n_sel > 0 {
            run.compress_len[run.index_k_src] as i32
        } else {
            0
        };
        let segments = (filled as usize + compress_len as usize)
            .div_ceil(128)
            .max(1);
        let idx_records = if layer.ratio > 0 {
            caches.idx_records[l].as_device_ptr()
        } else {
            caches.idx_records[0].as_device_ptr()
        };
        let ratio = layer.ratio as u64;
        let idx_pool = if ratio > 0 {
            ((cfg.original_seq_len as u64 / ratio + 63) / 64) as usize
        } else {
            1
        };
        s.pos_i64.copy_from(&[pos as i64]).map_err(cuda_err)?;
        s.swa_len.copy_from(&[filled]).map_err(cuda_err)?;
        s.idx_len.copy_from(&[compress_len]).map_err(cuda_err)?;
        k::f32_to_bf16(c, k::base(&s.q), k::base(&s.q_bf16), heads * hd);
        k::sparse_mla_decode(
            c,
            k::base(&s.q_bf16),
            k::base(&caches.kv_swa[l]),
            idxs,
            k::base(&s.swa_len),
            idx_records,
            k::I32::from_raw(idxs.as_raw() + (win * 4) as u64),
            k::base(&s.idx_len),
            k::base(&s.page_table),
            k::base(&layer.attn_sink),
            k::base(&s.pos_i64),
            k::base(&rope_cos[layer.freq]),
            k::base(&rope_sin[layer.freq]),
            k::base(&s.attn_part),
            k::base(&s.attn_stats),
            k::base(&s.o),
            heads,
            idx_pool,
            idx_pool,
            64 * 288,
            win,
            528,
            cfg.original_seq_len as usize + 1,
            segments,
            1,
        );
    }
    let gdim = heads * hd / og;
    dump_dev(c, &s.o, heads * hd, format!("d_pos{pos}_osnap{l}.f32"));
    for g in 0..og {
        if let Some(w8) = &layer.wo_a_fp8 {
            k::fp8_wo_a_gemv(
                c,
                k::row_u8(&w8.w, g * ol * gdim, ol * gdim),
                k::row_u8(
                    &w8.s,
                    g * (ol / w8.block) * w8.scale_cols,
                    (ol / w8.block) * w8.scale_cols,
                ),
                k::row(&s.o, g * gdim, gdim),
                k::row(&s.fl, g * ol, ol),
                ol,
                gdim,
                w8.block,
                w8.scale_cols,
            );
        } else {
            k::bf16_gemv(
                c,
                k::row16(&layer.wo_a, g * ol * gdim, ol * gdim),
                k::row(&s.o, g * gdim, gdim),
                k::row(&s.fl, g * ol, ol),
                ol,
                gdim,
            );
        }
    }
    k::emit8_packed(
        c,
        k::base(&s.fl),
        k::base(&s.act_fl),
        k::base(&s.act_fl_s),
        og * ol,
    );
    dump_dev(c, &s.fl, og * ol, format!("d_pos{pos}_fl{l}.f32"));
    dump_u8(c, &s.act_fl, og * ol, format!("d_pos{pos}_actfl{l}.u8"));
    dump_u8(
        c,
        &s.act_fl_s,
        (og * ol) / 32,
        format!("d_pos{pos}_actflS{l}.u8"),
    );
    let out = if world > 1 {
        k::F32::from_raw(s.coll_stage.device_ptr())
    } else {
        k::base(&s.attn_out)
    };
    gemm_q8(
        c,
        &layer.wo_b,
        k::base(&s.act_fl),
        k::base(&s.act_fl_s),
        out,
        1,
        dim,
        og * ol,
    );
    // wo_b is row-parallel: fp32 partial-sum all-reduce (spec §3)
    if let Some(comm) = comm.as_deref_mut() {
        let stream = c.stream.as_inner();
        s.coll_stage.sync(stream)?;
        let red = comm.allreduce(&s.coll_stage.as_f32()[..dim])?;
        s.coll_stage.as_f32()[..dim].copy_from_slice(&red);
    }
    let out = if world > 1 {
        let stream = c.stream.as_inner();
        s.coll_stage
            .htod(s.attn_out.as_device_ptr().as_raw() as u64, dim * 4, stream)?;
        k::base(&s.attn_out)
    } else {
        out
    };
    if world > 1 && std::env::var_os("FERROSEEK_DUMP_GPU").is_some() {
        let host = s.coll_stage.as_f32();
        s.attn_out.copy_from(&host[..dim]).map_err(cuda_err)?;
    }
    dump_dev(c, &s.attn_out, dim, format!("d_pos{pos}_attnout{l}.f32"));
    Ok(out)
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

fn moe_shared_step(c: &Cuda, cfg: &RefConfig, layer: &GpuLayer, s: &mut Scratch) {
    let dim = cfg.dim as usize;
    let inter = cfg.inter_dim as usize;
    let [w1, w2, w3] = &layer.shared_experts;
    let (gate, up) = if let Some(fused) = &layer.shared_w13 {
        gemm_q8(
            c,
            fused,
            k::base(&s.act_x),
            k::base(&s.act_x_s),
            k::base(&s.shared_out),
            1,
            2 * inter,
            dim,
        );
        (
            k::row(&s.shared_out, 0, inter),
            k::row(&s.shared_out, inter, inter),
        )
    } else {
        gemm_q8(
            c,
            w1,
            k::base(&s.act_x),
            k::base(&s.act_x_s),
            k::base(&s.gate_buf),
            1,
            inter,
            dim,
        );
        gemm_q8(
            c,
            w3,
            k::base(&s.act_x),
            k::base(&s.act_x_s),
            k::base(&s.up_buf),
            1,
            inter,
            dim,
        );
        (k::base(&s.gate_buf), k::base(&s.up_buf))
    };
    k::swiglu(c, gate, up, k::base(&s.h_e), inter, 1.0);
    k::emit8_packed(
        c,
        k::base(&s.h_e),
        k::base(&s.act_h),
        k::base(&s.act_h_s),
        inter,
    );
    gemm_q8(
        c,
        w2,
        k::base(&s.act_h),
        k::base(&s.act_h_s),
        k::base(&s.expert_y),
        1,
        dim,
        inter,
    );
}

fn moe_grouped_tl_step(c: &Cuda, cfg: &RefConfig, layer: &GpuLayer, s: &mut Scratch, sh: Sharding) {
    let dim = cfg.dim as usize;
    let inter = cfg.inter_dim as usize;
    let top_k = cfg.n_activated as usize;
    let e_start = sh.rank as usize * cfg.n_routed as usize / sh.world_size as usize;
    let e_len = cfg.n_routed as usize / sh.world_size as usize;
    let tiles = (top_k + 31) / 32 + top_k.min(e_len);
    let packed_a = k::U32::from_raw(
        s.act_x_s.as_device_ptr().as_raw() + (dim / 32).next_multiple_of(16) as u64,
    );
    k::moe_dispatch_ids(
        c,
        k::base(&s.gate_i_out),
        k::base(&s.dispatch_idx),
        top_k,
        e_start,
        e_len,
    );
    k::moe_dispatch_tl(
        c,
        k::base(&s.dispatch_idx),
        k::base(&s.gate_w_out),
        k::base(&s.order),
        k::base(&s.layout_weights),
        k::base(&s.route_pos),
        k::base(&s.tile_expert),
        k::base(&s.expert_count),
        k::base(&s.expert_offset),
        k::base(&s.expert_psum),
        k::base(&s.num_tiles),
        1,
        top_k,
        tiles,
    );
    let dispatch_event = cust::event::Event::new(cust::event::EventFlags::DEFAULT).unwrap();
    dispatch_event.record(&c.stream).unwrap();
    c.aux_stream
        .wait_event(dispatch_event, cust::stream::StreamWaitEventFlags::DEFAULT)
        .unwrap();
    k::moe_grouped_tl_cached(
        c,
        k::TlMoe::GateUp,
        k::base(&s.act_x),
        packed_a,
        layer.w1_map.unwrap(),
        k::base(layer.w1s_tl.as_ref().unwrap()),
        k::base(&s.order),
        k::base(&s.tile_expert),
        k::base(&s.route_h),
        tiles,
        1,
        top_k,
    );
    k::moe_grouped_tl_cached_on(
        c,
        k::TlMoe::GateUp,
        k::base(&s.act_x),
        packed_a,
        layer.w3_map.unwrap(),
        k::base(layer.w3s_tl.as_ref().unwrap()),
        k::base(&s.order),
        k::base(&s.tile_expert),
        k::base(&s.route_y),
        tiles,
        1,
        top_k,
        &c.aux_stream,
    );
    let event = cust::event::Event::new(cust::event::EventFlags::DEFAULT).unwrap();
    event.record(&c.aux_stream).unwrap();
    c.stream
        .wait_event(event, cust::stream::StreamWaitEventFlags::DEFAULT)
        .unwrap();
    k::moe_swiglu_route(
        c,
        k::base(&s.route_h),
        k::base(&s.route_y),
        k::base(&s.layout_weights),
        k::base(&s.route_h),
        inter,
        tiles * 32,
    );
    k::emit8_packed(
        c,
        k::base(&s.route_h),
        k::base(&s.route_q),
        k::base(&s.route_q_s),
        tiles * 32 * inter,
    );
    let packed_route = k::U32::from_raw(
        s.route_q_s.as_device_ptr().as_raw() + (tiles * inter).next_multiple_of(16) as u64,
    );
    k::moe_grouped_tl_cached(
        c,
        k::TlMoe::Down,
        k::base(&s.route_q),
        packed_route,
        layer.w2_map.unwrap(),
        k::base(layer.w2s_tl.as_ref().unwrap()),
        k::base(&s.order),
        k::base(&s.tile_expert),
        k::base(&s.route_y),
        tiles,
        1,
        top_k,
    );
    k::moe_combine_f32(
        c,
        k::base(&s.route_y),
        k::base(&s.route_pos),
        k::base(&s.y_moe),
        dim,
        top_k,
        1,
    );
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
    run: Option<&mut RunState>,
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
    dump_dev(
        c,
        &s.x,
        dim,
        format!("d_pos{}_gateX{l}.f32", crate::dump::pos()),
    );
    if layer.w1_all.is_some() || layer.w1_tl.is_some() {
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
    k::emit8_packed(
        c,
        k::base(&s.x),
        k::base(&s.act_x),
        k::base(&s.act_x_s),
        dim,
    );
    dump_u8(
        c,
        &s.act_x,
        dim,
        format!("d_pos{}_actX{}.u8", crate::dump::pos(), l),
    );
    dump_u8(
        c,
        &s.act_x_s,
        dim / 32,
        format!("d_pos{}_actXs{}.u8", crate::dump::pos(), l),
    );
    prof.add_us(2, t_gate);

    if layer.w1_tl.is_some() {
        moe_grouped_tl_step(c, cfg, layer, s, sh);
    } else if layer.w1_all.is_some() {
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
            dump_u8(
                c,
                &w1d,
                w1d.len(),
                format!("d_pos{}_expW1_{}_{l}.u8", crate::dump::pos(), e),
            );
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
            k::emit8_packed(
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

    // Copy routed partials, then launch the independent shared expert while
    // the host performs the ring reduction. The shared result is still added
    // after the reduction, preserving model.py:905 and rank-identical bits.
    if let Some(comm) = comm.as_deref_mut() {
        let stream = c.stream.as_inner();
        s.coll_stage
            .dtoh(s.y_moe.as_device_ptr().as_raw() as u64, dim * 4, stream)?;
        s.coll_stage.sync(stream)?;
        moe_shared_step(c, cfg, layer, s);
        let red = comm.allreduce(&s.coll_stage.as_f32()[..dim])?;
        s.coll_stage.as_f32()[..dim].copy_from_slice(&red);
        s.coll_stage
            .htod(s.y_moe.as_device_ptr().as_raw() as u64, dim * 4, stream)?;
    } else {
        moe_shared_step(c, cfg, layer, s);
    }
    dump_dev(
        c,
        &s.y_moe,
        dim,
        format!("d_pos{}_yrouted{l}.f32", crate::dump::pos()),
    );
    k::acc_f32(c, k::base(&s.expert_y), k::base(&s.y_moe), dim);
    if let Some(run) = run {
        if run.capture_y.is_some() {
            let mut host = vec![0.0f32; dim];
            s.y_moe.copy_to(&mut host).map_err(cuda_err)?;
            run.capture_y.as_mut().unwrap().push((l, host));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// M-batch decode (teacher-forced): one engine pass over m known tokens.
// Batched: embedding, hc, norms, rope, projections, sparse-MLA, gate.
// Sequential per position: ring writes, the CSA compressor state machine,
// indexer key append + selection, engram hashing — the position-dependent
// state, evolved in order so the post-pass state equals m sequential
// decode() calls. v1 bounds (explicit errors): upload-mode (TP1 or TP4),
// and the batch must not wrap the attention ring.

/// M-row scratch, grown on demand.
struct ScratchM {
    m_max: usize,
    n_routed: usize,
    h: [D<f32>; 2],
    x: D<f32>,
    qr: D<f32>,
    q: D<f32>,
    kvrow: D<f32>,
    o: D<f32>,
    fl: D<f32>,
    attn_out: D<f32>,
    y_moe: D<f32>,
    expert_y: D<f32>,
    gate_buf: D<f32>,
    up_buf: D<f32>,
    h_e: D<f32>,
    eflat: D<f32>,
    ekv: D<f32>,
    emb_row: D<f32>,
    pre_attn: D<f32>,
    pre_ffn: D<f32>,
    post: D<f32>,
    comb: D<f32>,
    zeros_h: D<f32>,
    ones_hc: D<f32>,
    comb_zero: D<f32>,
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
    gate_scores: D<f32>,
    qidx: D<f32>,
    widx: D<f32>,
    win_idxs: D<i32>,
    shared_idxs: D<i32>,
    pos_q: D<i32>,
    q_bf16: D<u16>,
    attn_part: D<f32>,
    attn_stats: D<f32>,
    pos_i64: D<i64>,
    swa_len: D<i32>,
    idx_len: D<i32>,
    page_table: D<i32>,
    logits: D<f32>,
    /// Per-position candidate blocks (host): the cand_src layer computes
    /// them for all m positions in its sequential loop; uses_cand layers
    /// consume position i's set at position i. `run.candidates` alone
    /// would leak the LAST position's set into every position.
    candidates_m: Vec<Vec<bool>>,
}

impl ScratchM {
    fn new(cfg: &RefConfig, engram: &EngramLayout, m: usize) -> io::Result<Self> {
        let dim = cfg.dim as usize;
        let hc = cfg.hc_mult as usize;
        let hd = cfg.head_dim as usize;
        let heads = cfg.n_heads as usize;
        let og = cfg.o_groups as usize;
        let ol = cfg.o_lora as usize;
        let inter = cfg.inter_dim as usize;
        let win = cfg.window_size as usize;
        let ql = cfg.q_lora as usize;
        let inh = cfg.index_n_heads as usize;
        let ihd = cfg.index_head_dim as usize;
        let vocab = cfg.vocab as usize;
        let ncols = engram.n_hash_cols;
        let ehd = engram.head_dim;
        let alloc = |n: usize| -> io::Result<D<f32>> {
            DeviceBuffer::from_slice(&vec![0.0f32; n.max(1)]).map_err(cuda_err)
        };
        let alloc_i = |n: usize| -> io::Result<D<i32>> {
            DeviceBuffer::from_slice(&vec![0i32; n.max(1)]).map_err(cuda_err)
        };
        let alloc_u = |n: usize| -> io::Result<D<u8>> {
            DeviceBuffer::from_slice(&vec![0u8; n.max(1)]).map_err(cuda_err)
        };
        let scale = |n: usize| (n / 32).next_multiple_of(16) + n / 32;
        Ok(Self {
            m_max: m,
            n_routed: cfg.n_routed as usize,
            h: [alloc(m * hc * dim)?, alloc(m * hc * dim)?],
            x: alloc(m * dim)?,
            qr: alloc(m * ql)?,
            q: alloc(m * heads * hd)?,
            kvrow: alloc(m * hd)?,
            o: alloc(m * heads * hd)?,
            fl: alloc(m * og * ol)?,
            attn_out: alloc(m * dim)?,
            y_moe: alloc(m * dim)?,
            expert_y: alloc(m * dim)?,
            gate_buf: alloc(m * inter)?,
            up_buf: alloc(m * inter)?,
            h_e: alloc(m * inter)?,
            eflat: alloc(m * ncols * ehd)?,
            ekv: alloc(m * (hc + 1) * dim)?,
            emb_row: alloc(m * dim)?,
            pre_attn: alloc(m * hc)?,
            pre_ffn: alloc(m * hc)?,
            post: alloc(m * hc)?,
            comb: alloc(m * hc * hc)?,
            zeros_h: alloc(m * hc * dim)?,
            ones_hc: alloc(m * hc)?,
            comb_zero: alloc(m * hc * hc)?,
            act_x: alloc_u(m * dim)?,
            act_x_s: alloc_u(scale(m * dim))?,
            act_qr: alloc_u(m * ql)?,
            act_qr_s: alloc_u(scale(m * ql))?,
            act_fl: alloc_u(m * og * ol)?,
            act_fl_s: alloc_u(scale(m * og * ol))?,
            act_h: alloc_u(m * inter)?,
            act_h_s: alloc_u(scale(m * inter))?,
            act_e: alloc_u(m * ncols * ehd)?,
            act_e_s: alloc_u(scale(m * ncols * ehd))?,
            gate_scores: alloc(m * cfg.n_routed as usize)?,
            qidx: alloc(m * inh * ihd)?,
            widx: alloc(m * inh)?,
            win_idxs: alloc_i(m * win)?,
            shared_idxs: alloc_i(m * SPARSE_MLA_IDX_WIDTH)?,
            pos_q: alloc_i(m * heads)?,
            q_bf16: DeviceBuffer::zeroed(m * heads * hd).map_err(cuda_err)?,
            // sparse_mla_decode.cu uses fixed 17-segment row strides
            // (17 * 8192 partials and 17 * 32 stats), independent of
            // the runtime segment count passed to the merge launch.
            attn_part: alloc(17 * m * heads * hd)?,
            attn_stats: alloc(17 * m * heads * 2)?,
            pos_i64: DeviceBuffer::from_slice(&vec![0i64; m]).map_err(cuda_err)?,
            swa_len: alloc_i(m)?,
            idx_len: alloc_i(m)?,
            page_table: {
                let pages = cfg.original_seq_len as usize / 64 + 1;
                let host: Vec<i32> = (0..m)
                    .flat_map(|_| (0..pages as i32).collect::<Vec<_>>())
                    .collect();
                DeviceBuffer::from_slice(&host).map_err(cuda_err)?
            },
            logits: alloc(m * vocab)?,
            candidates_m: vec![Vec::new(); m],
        })
    }
}

impl<'a> GpuModel<'a> {
    /// Teacher-forced M-batch decode: one engine pass over `tokens`,
    /// positions [self.pos, self.pos + m). Returns m logit rows. Batched
    /// where the math allows (embedding, hc, norms, rope, every GEMM,
    /// sparse-MLA, gate); position-dependent state (ring writes, CSA
    /// compressor, indexer, engram hashes) evolves sequentially per
    /// position, so the post-pass model state equals m sequential
    /// `decode` calls. v1 bounds: upload mode (TP1 or TP4), and the batch
    /// must not wrap the attention ring (chunk at the window boundary).

    fn capture_main_hidden(
        main_hiddens: &mut Vec<D<f32>>,
        c: &Cuda,
        cfg: &RefConfig,
        h: &D<f32>,
        row: usize,
        layer: u32,
    ) -> io::Result<()> {
        if !matches!(layer, 37 | 38 | 39) {
            return Ok(());
        }
        let dim = cfg.dim as usize;
        let hc = cfg.hc_mult as usize;
        let mut hv = vec![0.0f32; hc * dim];
        h.index(row * hc * dim..(row + 1) * hc * dim)
            .copy_to(&mut hv)
            .map_err(cuda_err)?;
        let mean: Vec<f32> = (0..dim)
            .map(|i| (0..hc).map(|j| hv[j * dim + i]).sum::<f32>() / hc as f32)
            .collect();
        main_hiddens.push(up_f32(c, &mean)?);
        Ok(())
    }

    pub fn decode_m(&mut self, tokens: &[u32]) -> io::Result<Vec<Vec<f32>>> {
        let guard = Cuda::global();
        let c = guard
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "cuda init"))?;
        self.decode_layers(c, tokens, None, None, None, None)
    }
    fn decode_layers(
        &mut self,
        c: &Cuda,
        tokens: &[u32],
        cfg_override: Option<&RefConfig>,
        layers_override: Option<&[GpuLayer]>,
        kv_x: Option<&D<f32>>,
        final_norm_override: Option<&D<f32>>,
    ) -> io::Result<Vec<Vec<f32>>> {
        let m = tokens.len();
        let cfg = cfg_override.unwrap_or(&self.cfg);
        let capture_drafter = self.drafter.is_some();
        if capture_drafter && layers_override.is_none() {
            self.main_hiddens.clear();
        }
        if m == 0 {
            return Ok(Vec::new());
        }
        let world = self.sharding.world_size as usize;
        if world > 1 && self.arena.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "decode_m: TP4 streaming is not supported",
            ));
        }

        let win = cfg.window_size as usize;
        let pos0 = self.pos;
        if pos0 + m as u64 > self.max_seq {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "batch beyond max_seq_len",
            ));
        }
        if (pos0 as usize % win) + m > win {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "decode_m batch wraps the attention ring: chunk at the window boundary",
            ));
        }
        if tokens.iter().any(|&t| t >= cfg.vocab) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "token id out of vocab",
            ));
        }

        // embedding rows (host bf16 dequant like decode(), or device gather)
        let dim0 = cfg.dim as usize;
        let mut emb_host: Option<Vec<f32>> = None;
        if self.emb.is_none() {
            if let Some(store) = self.store_keepalive.as_mut() {
                let v = store.get("embed.weight")?;
                let mut rows = vec![0.0f32; m * dim0];
                for (i, &t) in tokens.iter().enumerate() {
                    let start = t as usize * dim0 * 2;
                    if start + dim0 * 2 > v.bytes.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "token out of vocab",
                        ));
                    }
                    for (j, b) in v.bytes[start..start + dim0 * 2].chunks_exact(2).enumerate() {
                        rows[i * dim0 + j] =
                            ferroseek_core::quant::bf16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                    }
                }
                emb_host = Some(rows);
            }
        }

        // engram hashes per position (sequential, ordered — eng_cache state)
        let mut hashes_m: Vec<Vec<Vec<u64>>> = Vec::with_capacity(m);
        for (i, &t) in tokens.iter().enumerate() {
            if self.engram.layer_ids.is_empty() {
                hashes_m.push(Vec::new());
            } else {
                let token_map = self.token_map.clone();
                hashes_m.push(hash_ids(
                    cfg,
                    &self.engram,
                    &mut self.eng_cache,
                    token_map.as_deref(),
                    t,
                    pos0 + i as u64,
                ));
            }
        }

        let GpuModel {
            cfg: cfg_self,
            layers: layers_self,
            caches,
            scratch: s1,
            scratch_m,
            drafter_scratch_m,
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
            main_hiddens,
            ..
        } = self;
        let cfg = cfg_override.unwrap_or(cfg_self);
        let scratch_slot = if layers_override.is_some() {
            drafter_scratch_m
        } else {
            scratch_m
        };
        if scratch_slot
            .as_ref()
            .map(|s| (s.m_max, s.n_routed))
            .unwrap_or((0, 0))
            != (m, cfg.n_routed as usize)
        {
            *scratch_slot = Some(ScratchM::new(cfg, engram, m)?);
        }
        let layers = layers_override.unwrap_or(layers_self);
        let sm = scratch_slot.as_mut().unwrap();
        let arena_opt: Option<&crate::stream::HostArena> = arena.as_ref();
        let sh = *sharding;
        let mut comm_opt: Option<&mut (dyn Collectives + 'static)> = comm.as_deref_mut();

        let dim = cfg.dim as usize;
        let hc = cfg.hc_mult as usize;
        let heads = cfg.n_heads as usize / sh.world_size as usize;
        let eps = cfg.norm_eps;

        // h init: embedding rows -> hc copies (one-hot pre per row)
        if let Some(rows) = &emb_host {
            sm.emb_row
                .index(0..m * dim)
                .copy_from(rows)
                .map_err(cuda_err)?;
        } else {
            for (i, &t) in tokens.iter().enumerate() {
                let idx = [t as i64];
                let d_idx = DeviceBuffer::from_slice(&idx).map_err(cuda_err)?;
                k::gather_f32(
                    c,
                    k::base(emb.as_ref().unwrap()),
                    k::base(&d_idx),
                    k::row(&sm.emb_row, i * dim, dim),
                    1,
                    dim,
                );
            }
        }
        {
            // one-hot pre per row; ones_hc all ones — MUST upload before
            // the h-init hc_post below (post=ones expands the embedding
            // into hc identical copies; zeros would zero the stream)
            let mut pre = vec![0.0f32; m * hc];
            let mut ones = vec![0.0f32; m * hc];
            for i in 0..m {
                pre[i * hc] = 1.0;
                for j in 0..hc {
                    ones[i * hc + j] = 1.0;
                }
            }
            sm.pre_ffn
                .index(0..m * hc)
                .copy_from(&pre)
                .map_err(cuda_err)?;
            sm.ones_hc
                .index(0..m * hc)
                .copy_from(&ones)
                .map_err(cuda_err)?;
        }
        k::hc_post(
            c,
            k::base(&sm.emb_row),
            k::base(&sm.zeros_h),
            k::base(&sm.ones_hc),
            k::base(&sm.comb_zero),
            k::base(&sm.h[0]),
            m,
            dim,
        );
        {
            // per-position window idx lists and rope position rows
            let mut w = vec![-1i32; m * win];
            for (i, wi) in (0..m).map(|i| (i, packed_window_idxs(win, pos0 + i as u64))) {
                w[i * win..(i + 1) * win].copy_from_slice(&wi);
            }
            sm.win_idxs
                .index(0..m * win)
                .copy_from(&w)
                .map_err(cuda_err)?;
            let pos_q: Vec<i32> = (0..m * heads)
                .map(|r| (pos0 + u64::from(layers_override.is_some()) + (r / heads) as u64) as i32)
                .collect();
            sm.pos_q
                .index(0..m * heads)
                .copy_from(&pos_q)
                .map_err(cuda_err)?;
        }

        let mut h_cur = 0usize;
        for li in 0..layers.len() {
            let layer = &layers[li];
            if layers_override.is_none() {
                if let Some(d) = &self.glp_dirs[layer.layer as usize] {
                    for i in 0..m {
                        let h = k::row(&sm.h[h_cur], i * hc * dim, hc * dim);
                        k::glp_project(c, h, k::base(d), h, self.glp_alpha, hc, dim);
                    }
                }
            }
            if capture_drafter && layers_override.is_none() {
                Self::capture_main_hidden(main_hiddens, c, cfg, &sm.h[h_cur], m - 1, layer.layer)?;
            }

            // engram merge (per position: gather -> batched wkv -> merge)
            if let Some(ei) = layer.engram {
                let e = layer.engram_w.as_ref().unwrap();
                let ncols = engram.n_hash_cols;
                let ehd = engram.head_dim;
                for (i, hash_p) in hashes_m.iter().enumerate() {
                    let hash = &hash_p[ei];
                    let ids: Vec<i64> = hash.iter().map(|&h| h as i64).collect();
                    match (e.embed_w.host(), e.embed_s.host()) {
                        (Some((wp, wlen)), Some((sp, slen))) => {
                            let (wt, st) = unsafe {
                                (
                                    std::slice::from_raw_parts(wp, wlen),
                                    std::slice::from_raw_parts(sp, slen),
                                )
                            };
                            let rows: Vec<usize> =
                                ids.iter().take(ncols).map(|&id| id as usize).collect();
                            let mut flat = vec![0.0f32; rows.len() * ehd];
                            flat.par_chunks_mut(ehd).zip(rows.par_iter()).for_each(
                                |(chunk, &r)| {
                                    if r * ehd + ehd > wt.len() {
                                        chunk.fill(f32::NAN);
                                        return;
                                    }
                                    let row = ferroseek_ref::store::fp8_row_scaled(wt, st, r, ehd);
                                    chunk.copy_from_slice(&row);
                                },
                            );
                            if flat.iter().any(|v| v.is_nan()) {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidInput,
                                    "engram gather NaN (decode_m)",
                                ));
                            }
                            sm.eflat
                                .index(i * ncols * ehd..(i + 1) * ncols * ehd)
                                .copy_from(&flat)
                                .map_err(cuda_err)?;
                        }
                        _ => {
                            s1.hash_ids.copy_from(&ids).map_err(cuda_err)?;
                            k::gather_fp8(
                                c,
                                e.embed_w.ptr(),
                                e.embed_s.ptr(),
                                k::base(&s1.hash_ids),
                                k::row(&sm.eflat, i * ncols * ehd, ncols * ehd),
                                ncols,
                                ehd,
                            );
                        }
                    }
                }
                k::emit8_packed(
                    c,
                    k::base(&sm.eflat),
                    k::base(&sm.act_e),
                    k::base(&sm.act_e_s),
                    m * ncols * ehd,
                );
                gemm_q8(
                    c,
                    &e.wkv,
                    k::base(&sm.act_e),
                    k::base(&sm.act_e_s),
                    k::base(&sm.ekv),
                    m,
                    dim * (hc + 1),
                    ncols * ehd,
                );
                let inv = (dim as f32).sqrt().recip();
                for i in 0..m {
                    for j in 0..hc {
                        k::engram_merge(
                            c,
                            k::row(&sm.h[h_cur], i * hc * dim + j * dim, dim),
                            k::row(&sm.ekv, i * (hc + 1) * dim + j * dim, dim),
                            k::row(&sm.ekv, i * (hc + 1) * dim + hc * dim, dim),
                            k::row(&e.w_prod, j * dim, dim),
                            dim,
                            inv,
                        );
                    }
                }
            }

            // attention wiring (hc mix + collapse + norm)
            let hc_dim = hc * dim;
            let mix_hc = cfg.mix_hc();
            k::hc_block(
                c,
                k::base(&sm.h[h_cur]),
                k::base(&layer.hc_attn_fn),
                k::base(&layer.hc_attn_base),
                k::base(&layer.hc_attn_scale),
                k::base(&sm.pre_attn),
                k::base(&sm.post),
                k::base(&sm.comb),
                m,
                hc_dim,
                mix_hc,
                cfg.hc_eps,
            );
            k::hc_pre(
                c,
                k::base(&sm.h[h_cur]),
                k::base(&sm.pre_ffn),
                k::base(&sm.x),
                m,
                dim,
            );
            k::rmsnorm_m(
                c,
                k::base(&sm.x),
                k::base(&layer.attn_norm_w),
                k::base(&sm.x),
                m,
                dim,
                eps,
            );

            attention_step_m(
                c,
                cfg,
                layer,
                sm,
                s1,
                caches,
                run,
                bands,
                rope_cos,
                rope_sin,
                pos0,
                m,
                kv_x,
                sh,
                comm_opt.as_deref_mut(),
            )?;
            if layers_override.is_none() {
                if let Some(d) = self
                    .glp_dirs
                    .get(layer.layer as usize + 1)
                    .and_then(|d| d.as_ref())
                {
                    for i in 0..m {
                        let out = k::row(&sm.attn_out, i * dim, dim);
                        k::glp_project(c, out, k::base(d), out, self.glp_alpha, 1, dim);
                    }
                }
            }

            {
                let h_next = 1 - h_cur;
                k::hc_post(
                    c,
                    k::base(&sm.attn_out),
                    k::base(&sm.h[h_cur]),
                    k::base(&sm.post),
                    k::base(&sm.comb),
                    k::base(&sm.h[h_next]),
                    m,
                    dim,
                );
                h_cur = h_next;
            }

            // ffn wiring
            k::hc_block(
                c,
                k::base(&sm.h[h_cur]),
                k::base(&layer.hc_ffn_fn),
                k::base(&layer.hc_ffn_base),
                k::base(&layer.hc_ffn_scale),
                k::base(&sm.pre_ffn),
                k::base(&sm.post),
                k::base(&sm.comb),
                m,
                hc_dim,
                mix_hc,
                cfg.hc_eps,
            );
            k::hc_pre(
                c,
                k::base(&sm.h[h_cur]),
                k::base(&sm.pre_attn),
                k::base(&sm.x),
                m,
                dim,
            );
            k::rmsnorm_m(
                c,
                k::base(&sm.x),
                k::base(&layer.ffn_norm_w),
                k::base(&sm.x),
                m,
                dim,
                eps,
            );

            moe_step_m(
                c,
                cfg,
                layer,
                sm,
                s1,
                arena_opt,
                m,
                sh,
                comm_opt.as_deref_mut(),
            )?;

            {
                let h_next = 1 - h_cur;
                k::hc_post(
                    c,
                    k::base(&sm.y_moe),
                    k::base(&sm.h[h_cur]),
                    k::base(&sm.post),
                    k::base(&sm.comb),
                    k::base(&sm.h[h_next]),
                    m,
                    dim,
                );
                h_cur = h_next;
            }
        }
        if layers_override.is_none() {
            if let Some(d) = &self.glp_dirs[layers.len() - 1] {
                for i in 0..m {
                    let h = k::row(&sm.h[h_cur], i * hc * dim, hc * dim);
                    k::glp_project(c, h, k::base(d), h, self.glp_alpha, hc, dim);
                }
            }
        }

        let final_norm = final_norm_override.unwrap_or(final_norm);
        // tail: collapse, final norm, local vocab head + TP all-gather per row
        let vocab_l = cfg.vocab as usize / world;
        k::hc_pre(
            c,
            k::base(&sm.h[h_cur]),
            k::base(&sm.pre_ffn),
            k::base(&sm.x),
            m,
            dim,
        );
        k::rmsnorm_m(
            c,
            k::base(&sm.x),
            k::base(final_norm),
            k::base(&sm.x),
            m,
            dim,
            eps,
        );
        for i in 0..m {
            k::bf16_gemv(
                c,
                k::base(head),
                k::row(&sm.x, i * dim, dim),
                k::row(&sm.logits, i * cfg.vocab as usize, cfg.vocab as usize),
                vocab_l,
                dim,
            );
            if let Some(comm) = comm_opt.as_deref_mut() {
                let stream = c.stream.as_inner();
                s1.coll_stage.dtoh(
                    k::row(&sm.logits, i * cfg.vocab as usize, vocab_l).as_raw() as u64,
                    vocab_l * 4,
                    stream,
                )?;
                s1.coll_stage.sync(stream)?;

                let full = comm.allgather(&s1.coll_stage.as_f32()[..vocab_l])?;
                sm.logits
                    .index(i * cfg.vocab as usize..(i + 1) * cfg.vocab as usize)
                    .copy_from(&full)
                    .map_err(cuda_err)?;
            }
        }
        let mut out = vec![0.0f32; m * cfg.vocab as usize];
        sm.logits
            .index(0..m * cfg.vocab as usize)
            .copy_to(&mut out)
            .map_err(cuda_err)?;
        self.pos += m as u64;
        Ok(out.chunks(cfg.vocab as usize).map(|c| c.to_vec()).collect())
    }

    fn drafter_main_x(
        c: &Cuda,
        drafter: &GpuDrafter,
        main_hiddens: &[D<f32>],
        cfg: &RefConfig,
    ) -> io::Result<D<f32>> {
        if main_hiddens.len() != 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "DSpark requires main_hiddens from layers 37, 38 and 39",
            ));
        }
        let dim = cfg.dim as usize;
        let mut main_hidden = up_zero_f32(c, 3 * dim)?;
        for i in 0..3 {
            main_hidden
                .index(i * dim..(i + 1) * dim)
                .copy_from(&main_hiddens[i])
                .map_err(cuda_err)?;
        }
        let act = DeviceBuffer::<u8>::zeroed(3 * dim).map_err(cuda_err)?;
        let act_s = DeviceBuffer::<u8>::zeroed(3 * dim / 16).map_err(cuda_err)?;
        k::emit8_packed(
            c,
            k::base(&main_hidden),
            k::base(&act),
            k::base(&act_s),
            3 * dim,
        );
        let main_x = up_zero_f32(c, dim)?;
        gemm_q8(
            c,
            &drafter.main_proj,
            k::base(&act),
            k::base(&act_s),
            k::base(&main_x),
            1,
            dim,
            3 * dim,
        );
        k::rmsnorm(
            c,
            k::base(&main_x),
            k::base(&drafter.main_norm),
            k::base(&main_x),
            dim,
            cfg.norm_eps,
        );
        Ok(main_x)
    }

    fn seed_drafter(&mut self, c: &Cuda, position: u64) -> io::Result<()> {
        let Some(drafter) = self.drafter.take() else {
            return Ok(());
        };
        let result = (|| {
            let main_x = Self::drafter_main_x(c, &drafter, &self.main_hiddens, &self.cfg)?;
            let dim = self.cfg.dim as usize;
            let hd = self.cfg.head_dim as usize;
            let rd = self.cfg.rope_dim as usize;
            let win = self.cfg.window_size as usize;
            let act = DeviceBuffer::<u8>::zeroed(dim).map_err(cuda_err)?;
            let act_s = DeviceBuffer::<u8>::zeroed(dim / 16).map_err(cuda_err)?;
            let kv = up_zero_f32(c, hd)?;
            k::emit8_packed(c, k::base(&main_x), k::base(&act), k::base(&act_s), dim);
            for layer in &drafter.layers {
                gemm_q8(
                    c,
                    &layer.wkv,
                    k::base(&act),
                    k::base(&act_s),
                    k::base(&kv),
                    1,
                    hd,
                    dim,
                );
                let ring = k::row(
                    &drafter.caches.kv_all[layer.layer as usize],
                    (position as usize % win) * hd,
                    hd,
                );
                k::rmsnorm(
                    c,
                    k::base(&kv),
                    k::base(&layer.kv_norm_w),
                    ring,
                    hd,
                    self.cfg.norm_eps,
                );
                k::rope(
                    c,
                    ring,
                    k::base(&self.rope_cos[layer.freq]),
                    k::base(&self.rope_sin[layer.freq]),
                    1,
                    hd,
                    rd,
                    position,
                    false,
                );
                k::snap8(c, ring, hd);
            }
            Ok(())
        })();
        self.drafter = Some(drafter);
        result
    }

    fn drafter_step(&mut self, anchor: u32, position: u64) -> io::Result<Vec<u32>> {
        let mut drafter = self.drafter.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Unsupported, "DSpark drafter is not loaded")
        })?;
        let result = (|| {
            let guard = Cuda::global();
            let c = guard
                .as_ref()
                .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "cuda init"))?;
            let main_x = Self::drafter_main_x(c, &drafter, &self.main_hiddens, &self.cfg)?;
            let mut dcfg = self.cfg.clone();
            dcfg.n_routed = self.cfg.dspark_n_routed;
            dcfg.n_activated = self.cfg.dspark_activated;
            let noise = dcfg.dspark_noise_token_id;
            let drafts = [anchor, noise, noise, noise, noise];
            drafter.pos = position;
            drafter.eng_cache = self.eng_cache[..position as usize].to_vec();
            std::mem::swap(&mut self.caches, &mut drafter.caches);
            std::mem::swap(&mut self.run, &mut drafter.run);
            std::mem::swap(&mut self.eng_cache, &mut drafter.eng_cache);
            std::mem::swap(&mut self.pos, &mut drafter.pos);
            let result = self.decode_layers(
                c,
                &drafts,
                Some(&dcfg),
                Some(&drafter.layers),
                Some(&main_x),
                Some(&drafter.norm),
            );
            std::mem::swap(&mut self.caches, &mut drafter.caches);
            std::mem::swap(&mut self.run, &mut drafter.run);
            std::mem::swap(&mut self.eng_cache, &mut drafter.eng_cache);
            std::mem::swap(&mut self.pos, &mut drafter.pos);
            let rows = result?;
            let vocab_l = self.cfg.vocab as usize / self.sharding.world_size as usize;
            let emb = up_zero_f32(c, 256)?;
            let markov = up_zero_f32(c, vocab_l)?;
            let mut previous = anchor;
            let mut ids = Vec::with_capacity(rows.len());
            for mut row in rows {
                let idx = DeviceBuffer::from_slice(&[previous as i64]).map_err(cuda_err)?;
                k::gather_f32(
                    c,
                    k::base(&drafter.markov_embed),
                    k::base(&idx),
                    k::base(&emb),
                    1,
                    256,
                );
                k::f32_gemv(
                    c,
                    k::base(&drafter.markov_head),
                    k::base(&emb),
                    k::base(&markov),
                    vocab_l,
                    256,
                );
                let mut local = vec![0.0f32; vocab_l];
                markov.copy_to(&mut local).map_err(cuda_err)?;
                let full = match self.comm.as_deref_mut() {
                    Some(comm) => comm.allgather(&local)?,
                    None => local,
                };
                for (logit, bias) in row.iter_mut().zip(full) {
                    *logit += bias;
                }
                previous = ferroseek_ref::sample::argmax(&row);
                ids.push(previous);
            }
            Ok(ids)
        })();
        self.drafter = Some(drafter);
        result
    }

    pub fn can_speculate(&self) -> bool {
        self.drafter.is_some() && !(self.sharding.world_size > 1 && self.arena.is_some())
    }
    pub fn disable_speculation(&mut self) {
        self.drafter = None;
        self.main_hiddens.clear();
    }

    pub fn speculate(
        &mut self,
        position: u64,
        anchor: u32,
    ) -> io::Result<(Vec<u32>, u64, Vec<f32>)> {
        let initial = self.decode(anchor, position)?;
        let drafts = self.drafter_step(anchor, position)?;
        let win = self.cfg.window_size as usize;
        let batch_len = if std::env::var_os("FERROSEEK_SEQ_SPEC").is_some() {
            0
        } else if self.spec_accept >= 5 {
            5
        } else {
            0
        };
        let batch_len = batch_len.min(drafts.len());
        let batch_ok = batch_len >= 2 && (self.pos as usize % win) + batch_len <= win;
        if batch_ok {
            let batch = &drafts[..batch_len];
            let snapshot = self.spec_snapshot(batch_len)?;
            let rows = self.decode_m(batch)?;
            let mut accepted = vec![anchor];
            let mut last = initial.clone();
            let mut n = 0usize;
            for (i, &draft) in batch.iter().enumerate() {
                if ferroseek_ref::sample::argmax(&last) != draft {
                    break;
                }
                accepted.push(draft);
                last = rows[i].clone();
                n += 1;
            }
            if n < batch.len() {
                self.spec_restore(&snapshot)?;
                last = initial;
                for &draft in &batch[..n] {
                    last = self.decode(draft, self.pos)?;
                }
            }
            self.spec_accept = accepted.len();
            return Ok((accepted, self.pos, last));
        }
        let mut logits = initial;
        let mut accepted = vec![anchor];
        for draft in drafts {
            if ferroseek_ref::sample::argmax(&logits) != draft {
                break;
            }
            logits = self.decode(draft, self.pos)?;
            accepted.push(draft);
        }
        self.spec_accept = accepted.len();
        Ok((accepted, self.pos, logits))
    }
}

/// The M-batch attention sublayer: batched q/kv projections and rope,
/// sequential per-position cache writes + CSA compressor + indexer,
/// one fused sparse-MLA launch per main layer on the real shape (scalar fallback otherwise).
#[allow(clippy::too_many_arguments)]
fn attention_step_m(
    c: &Cuda,
    cfg: &RefConfig,
    layer: &GpuLayer,
    sm: &mut ScratchM,
    s1: &mut Scratch,
    caches: &mut GpuCaches,
    run: &mut RunState,
    bands: &[Vec<usize>],
    rope_cos: &[D<f32>; 2],
    rope_sin: &[D<f32>; 2],
    pos0: u64,
    m: usize,
    kv_x: Option<&D<f32>>,
    sh: Sharding,
    mut comm: Option<&mut (dyn Collectives + 'static)>,
) -> io::Result<()> {
    let l = layer.layer as usize;
    let dim = cfg.dim as usize;
    let hd = cfg.head_dim as usize;
    let rd = cfg.rope_dim as usize;
    let world = sh.world_size as usize;
    let heads = cfg.n_heads as usize / world;
    let win = cfg.window_size as usize;
    let eps = cfg.norm_eps;
    let og = cfg.o_groups as usize / world;
    let ol = cfg.o_lora as usize;
    let inh = cfg.index_n_heads as usize / world;
    let ihd = cfg.index_head_dim as usize;
    let ratio = layer.ratio as u64;
    let use_sparse_mla = std::env::var_os("FERROSEEK_SPARSE_FP32").is_none()
        && heads == 16
        && hd == 512
        && layer.layer < cfg.n_layers;
    let mut filled_rows = Vec::with_capacity(m);
    let mut idx_len_rows = Vec::with_capacity(m);
    let mut segments_max = 1usize;
    let kv_x = kv_x.unwrap_or(&sm.x);

    // q path: emit8 -> wq_a -> q_norm -> emit8 -> wq_b -> rope(buf)
    k::emit8_packed(
        c,
        k::base(&sm.x),
        k::base(&sm.act_x),
        k::base(&sm.act_x_s),
        m * dim,
    );
    gemm_q8(
        c,
        &layer.wq_a,
        k::base(&sm.act_x),
        k::base(&sm.act_x_s),
        k::base(&sm.qr),
        m,
        cfg.q_lora as usize,
        dim,
    );
    k::rmsnorm_m(
        c,
        k::base(&sm.qr),
        k::base(&layer.q_norm_w),
        k::base(&sm.qr),
        m,
        cfg.q_lora as usize,
        eps,
    );
    k::emit8_packed(
        c,
        k::base(&sm.qr),
        k::base(&sm.act_qr),
        k::base(&sm.act_qr_s),
        m * cfg.q_lora as usize,
    );
    gemm_q8(
        c,
        &layer.wq_b,
        k::base(&sm.act_qr),
        k::base(&sm.act_qr_s),
        k::base(&sm.q),
        m,
        heads * hd,
        cfg.q_lora as usize,
    );
    k::rope_uniform_pos_buf(
        c,
        k::base(&sm.q),
        k::base(&rope_cos[layer.freq]),
        k::base(&rope_sin[layer.freq]),
        k::base(&sm.pos_q),
        m * heads,
        hd,
        rd,
        false,
    );

    if layer.layer >= cfg.n_layers {
        if m != 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "DSpark expects five draft rows",
            ));
        }
        // Target-token KV at p. The ring contains only target tokens; the
        // current slot was already seeded by decode(), so this is the same
        // value rewritten through the shared DSpark forward path.
        k::emit8_packed(
            c,
            k::base(kv_x),
            k::base(&sm.act_x),
            k::base(&sm.act_x_s),
            dim,
        );
        gemm_q8(
            c,
            &layer.wkv,
            k::base(&sm.act_x),
            k::base(&sm.act_x_s),
            k::base(&sm.kvrow),
            1,
            hd,
            dim,
        );
        let ring = k::row(&caches.kv_all[l], (pos0 as usize % win) * hd, hd);
        k::rmsnorm(
            c,
            k::base(&sm.kvrow),
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
            pos0,
            false,
        );
        k::snap8(c, ring, hd);

        // Draft KVs are transient rows after the window. Every draft query
        // attends every draft row (non-causal block attention).
        k::emit8_packed(
            c,
            k::base(&sm.x),
            k::base(&sm.act_x),
            k::base(&sm.act_x_s),
            m * dim,
        );
        gemm_q8(
            c,
            &layer.wkv,
            k::base(&sm.act_x),
            k::base(&sm.act_x_s),
            k::base(&sm.kvrow),
            m,
            hd,
            dim,
        );
        for i in 0..m {
            let draft = k::row(&caches.kv_all[l], (win + i) * hd, hd);
            k::rmsnorm(
                c,
                k::row(&sm.kvrow, i * hd, hd),
                k::base(&layer.kv_norm_w),
                draft,
                hd,
                eps,
            );
            k::rope(
                c,
                draft,
                k::base(&rope_cos[layer.freq]),
                k::base(&rope_sin[layer.freq]),
                1,
                hd,
                rd,
                pos0 + 1 + i as u64,
                false,
            );
            k::snap8(c, draft, hd);
        }
        let filled = (pos0 as usize + 1).min(win);
        let list: Vec<i32> = (0..filled as i32)
            .chain((win as i32)..(win + m) as i32)
            .collect();
        let stride = SPARSE_MLA_IDX_WIDTH;
        for i in 0..m {
            sm.shared_idxs
                .index(i * stride..i * stride + list.len())
                .copy_from(&list)
                .map_err(cuda_err)?;
            k::sparse_attn(
                c,
                k::row(&sm.q, i * heads * hd, heads * hd),
                k::base(&caches.kv_all[l]),
                sm.shared_idxs
                    .index(i * stride..i * stride + list.len())
                    .as_device_ptr(),
                k::base(&layer.attn_sink),
                k::row(&sm.o, i * heads * hd, heads * hd),
                heads,
                hd,
                list.len(),
                (hd as f32).sqrt().recip(),
            );
        }
    } else {
        // window KV projection (batched), ring writes (per position)
        k::emit8_packed(
            c,
            k::base(kv_x),
            k::base(&sm.act_x),
            k::base(&sm.act_x_s),
            m * dim,
        );
        gemm_q8(
            c,
            &layer.wkv,
            k::base(&sm.act_x),
            k::base(&sm.act_x_s),
            k::base(&sm.kvrow),
            m,
            hd,
            dim,
        );
        for i in 0..m {
            let pos = pos0 + i as u64;
            let slot = (pos % win as u64) as usize;
            filled_rows.push(((pos + 1).min(win as u64)) as i32);
            let mut row_idx_len = 0i32;
            let ring = k::row(&caches.kv_all[l], slot * hd, hd);
            k::rmsnorm(
                c,
                k::row(&sm.kvrow, i * hd, hd),
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
            let packed = caches.kv_swa[l]
                .index(slot * 528..(slot + 1) * 528)
                .as_device_ptr();
            k::emit8(c, ring, packed, k::U8::from_raw(packed.as_raw() + 512), hd);
            let mut selected_len = win;
            if layer.ratio > 0 {
                let pos = pos0 + i as u64;
                let compress_len = ((pos + 1) / ratio) as usize;
                let mut latent_ready = false;
                if let Some(src) = layer.kv_src {
                    run.index_k_src = src;
                    let comp = layer.compressor.as_ref().unwrap();
                    if ratio == 1 {
                        compressor_gemv(
                            c,
                            &comp.wkv,
                            comp.wkv_f32.as_ref(),
                            k::row(&sm.x, i * dim, dim),
                            k::base(&s1.latent),
                            hd,
                            dim,
                        );
                        k::rmsnorm(
                            c,
                            k::base(&s1.latent),
                            k::base(&comp.norm_w),
                            k::base(&s1.latent),
                            hd,
                            eps,
                        );
                        latent_ready = true;
                    } else {
                        let slot_i = (pos % ratio) as usize;
                        let kv_state = caches.comp_kv[src].as_ref().unwrap();
                        let score_state = caches.comp_score[src].as_mut().unwrap();
                        compressor_gemv(
                            c,
                            &comp.wkv,
                            comp.wkv_f32.as_ref(),
                            k::row(&sm.x, i * dim, dim),
                            k::row(kv_state, slot_i * hd, hd),
                            hd,
                            dim,
                        );
                        compressor_gemv(
                            c,
                            comp.wgate.as_ref().unwrap_or(&comp.wkv),
                            comp.wgate_f32.as_ref(),
                            k::row(&sm.x, i * dim, dim),
                            k::row(score_state, slot_i * hd, hd),
                            hd,
                            dim,
                        );
                        if (pos + 1) % ratio == 0 {
                            k::compressor_finish(
                                c,
                                k::base(kv_state),
                                k::base(score_state),
                                k::base(&s1.latent),
                                hd,
                            );
                            k::rmsnorm(
                                c,
                                k::base(&s1.latent),
                                k::base(&comp.norm_w),
                                k::base(&s1.latent),
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

                if layer.idx_src {
                    if layer.owns_k && latent_ready {
                        let kowner = layer.k_owner.as_ref().unwrap();
                        let j = compress_len - 1;
                        k::bf16_gemv(
                            c,
                            k::base(&kowner.wk),
                            k::base(&s1.latent),
                            k::base(&s1.krow),
                            ihd,
                            hd,
                        );
                        let krow_dst = k::row(&caches.index_k[run.index_k_src], j * ihd, ihd);
                        k::rmsnorm(
                            c,
                            k::base(&s1.krow),
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
                    // query side for this position
                    k::fp8_gemm(
                        c,
                        k::row_u8(&sm.act_qr, i * cfg.q_lora as usize, cfg.q_lora as usize),
                        k::row_u8(
                            &sm.act_qr_s,
                            i * cfg.q_lora as usize / 32,
                            cfg.q_lora as usize / 32,
                        ),
                        layer.indexer.as_ref().unwrap().wq_b.w.ptr(),
                        layer.indexer.as_ref().unwrap().wq_b.s.ptr(),
                        k::row(&sm.qidx, i * inh * ihd, inh * ihd),
                        1,
                        inh * ihd,
                        cfg.q_lora as usize,
                    );
                    k::rope_uniform_pos(
                        c,
                        k::row(&sm.qidx, i * inh * ihd, inh * ihd),
                        k::base(&rope_cos[layer.freq]),
                        k::base(&rope_sin[layer.freq]),
                        inh,
                        ihd,
                        rd,
                        pos,
                        false,
                    );
                    k::fp4_snap(
                        c,
                        k::row(&sm.qidx, i * inh * ihd, inh * ihd),
                        inh * ihd,
                        32,
                        false,
                    );
                    k::f32_gemv(
                        c,
                        k::base(&layer.indexer.as_ref().unwrap().weights_proj),
                        k::row(&sm.x, i * dim, dim),
                        k::row(&sm.widx, i * inh, inh),
                        inh,
                        dim,
                    );

                    // score + selection for this position
                    let topk_max = cfg.index_topk as usize;
                    let mut list = vec![-1i32; win + topk_max];
                    if compress_len == 0 {
                        let wi = packed_window_idxs(win, pos);
                        list[..win].copy_from_slice(&wi);
                        if i + 1 == m {
                            run.shared_n_sel = win;
                            s1.shared_idxs
                                .index(0..win)
                                .copy_from(&list[..win])
                                .map_err(cuda_err)?;
                        }
                    } else {
                        k::indexer_score(
                            c,
                            k::row(&sm.qidx, i * inh * ihd, inh * ihd),
                            k::base(&caches.index_k[run.index_k_src]),
                            k::row(&sm.widx, i * inh, inh),
                            k::base(&s1.score),
                            inh,
                            ihd,
                            compress_len,
                        );
                        let mut score = vec![0.0f32; compress_len];
                        s1.score
                            .index(0..compress_len)
                            .copy_to(&mut score)
                            .map_err(cuda_err)?;
                        if let Some(comm) = comm.as_deref_mut() {
                            score = comm.allreduce(&score)?;
                        }
                        if layer.cand_src {
                            let cand = select_candidate_blocks(
                                &score,
                                cfg.candidate_topk_blocks as usize,
                                cfg.candidate_block_size as usize,
                            );
                            run.candidates = cand.clone();
                            if i < sm.candidates_m.len() {
                                sm.candidates_m[i] = cand;
                            }
                        } else if layer.uses_cand {
                            let cand = sm
                                .candidates_m
                                .get(i)
                                .filter(|c| !c.is_empty())
                                .unwrap_or(&run.candidates);
                            for (t, v) in score.iter_mut().enumerate() {
                                if !cand.get(t).copied().unwrap_or(false) {
                                    *v = f32::NEG_INFINITY;
                                }
                            }
                        }
                        let topk = topk_max.min(compress_len);
                        let cmp = select_topk(&score, topk, compress_len, win as i32);
                        selected_len = win + cmp.len();
                        let wi = packed_window_idxs(win, pos);
                        list[..win].copy_from_slice(&wi);
                        for (j, &v) in cmp.iter().enumerate() {
                            list[win + j] = v;
                        }
                        if i + 1 == m {
                            run.shared_n_sel = selected_len;
                            s1.shared_idxs
                                .index(0..selected_len)
                                .copy_from(&list[..selected_len])
                                .map_err(cuda_err)?;
                        }
                    }
                    sm.shared_idxs
                        .index(i * SPARSE_MLA_IDX_WIDTH..i * SPARSE_MLA_IDX_WIDTH + list.len())
                        .copy_from(&list)
                        .map_err(cuda_err)?;
                }

                if selected_len > win {
                    row_idx_len = compress_len as i32;
                }
                // latent tail: rope + fp4 snap + fan-out
                if latent_ready {
                    if let Some(src) = layer.kv_src {
                        let j = compress_len - 1;
                        k::rope(
                            c,
                            k::base(&s1.latent),
                            k::base(&rope_cos[layer.freq]),
                            k::base(&rope_sin[layer.freq]),
                            1,
                            hd,
                            rd,
                            pos + 1 - ratio,
                            false,
                        );
                        k::fp4_snap(c, k::base(&s1.latent), hd, 16, true);
                        let rec = caches.idx_records[l]
                            .index(j * 288..(j + 1) * 288)
                            .as_device_ptr();
                        k::fp4_emit(
                            c,
                            k::base(&s1.latent),
                            rec,
                            k::U8::from_raw(rec.as_raw() + 256),
                            hd,
                            16,
                        );
                        for &cl in &bands[src] {
                            if cl != src {
                                let dst = caches.idx_records[cl]
                                    .index(j * 288..(j + 1) * 288)
                                    .as_device_ptr();
                                k::copy_u8(c, rec, dst, 288);
                            }
                        }
                        for &cl in &bands[src] {
                            let dst = k::row(&caches.kv_all[cl], win * hd + j * hd, hd);
                            k::copy_f32(c, k::base(&s1.latent), dst, hd);
                        }
                    }
                }
            }
            // The indexed list has a 2048-wide row stride; only its valid
            // prefix participates in attention. This matters for short M-batch verification.
            let topk_max = cfg.index_topk as usize;
            let (idxs, n_sel) = if layer.ratio > 0 {
                (
                    sm.shared_idxs
                        .index(i * SPARSE_MLA_IDX_WIDTH..(i + 1) * SPARSE_MLA_IDX_WIDTH)
                        .as_device_ptr(),
                    selected_len,
                )
            } else {
                (
                    sm.win_idxs.index(i * win..(i + 1) * win).as_device_ptr(),
                    win,
                )
            };
            if !use_sparse_mla {
                k::sparse_attn(
                    c,
                    k::row(&sm.q, i * heads * hd, heads * hd),
                    k::base(&caches.kv_all[l]),
                    idxs,
                    k::base(&layer.attn_sink),
                    k::row(&sm.o, i * heads * hd, heads * hd),
                    heads,
                    hd,
                    n_sel,
                    (hd as f32).sqrt().recip(),
                );
            }
            idx_len_rows.push(row_idx_len);
            let row_segments =
                ((filled_rows[i] as usize + row_idx_len as usize).div_ceil(128)).max(1);
            segments_max = segments_max.max(row_segments);
        }
        if use_sparse_mla {
            let positions: Vec<i64> = (0..m).map(|i| (pos0 + i as u64) as i64).collect();
            sm.pos_i64.copy_from(&positions).map_err(cuda_err)?;
            sm.swa_len.copy_from(&filled_rows).map_err(cuda_err)?;
            sm.idx_len.copy_from(&idx_len_rows).map_err(cuda_err)?;
            k::f32_to_bf16(c, k::base(&sm.q), k::base(&sm.q_bf16), m * heads * hd);
            let idx_pool = if ratio > 0 {
                ((cfg.original_seq_len as u64 / ratio + 63) / 64) as usize
            } else {
                1
            };
            let (swa_idxs, idx_idxs) = if ratio > 0 {
                (
                    k::base(&sm.shared_idxs),
                    k::I32::from_raw(sm.shared_idxs.as_device_ptr().as_raw() + (win * 4) as u64),
                )
            } else {
                (
                    k::base(&sm.win_idxs),
                    k::I32::from_raw(sm.win_idxs.as_device_ptr().as_raw() + (win * 4) as u64),
                )
            };
            let idx_records = if ratio > 0 {
                caches.idx_records[l].as_device_ptr()
            } else {
                caches.idx_records[0].as_device_ptr()
            };
            k::sparse_mla_decode(
                c,
                k::base(&sm.q_bf16),
                k::base(&caches.kv_swa[l]),
                swa_idxs,
                k::base(&sm.swa_len),
                idx_records,
                idx_idxs,
                k::base(&sm.idx_len),
                k::base(&sm.page_table),
                k::base(&layer.attn_sink),
                k::base(&sm.pos_i64),
                k::base(&rope_cos[layer.freq]),
                k::base(&rope_sin[layer.freq]),
                k::base(&sm.attn_part),
                k::base(&sm.attn_stats),
                k::base(&sm.o),
                heads,
                idx_pool,
                idx_pool,
                64 * 288,
                win,
                528,
                cfg.original_seq_len as usize + 1,
                segments_max,
                m,
            );
        }
    }

    // inverse rope, snap, wo_a per (pos, group), wo_b batched
    k::rope_uniform_pos_buf(
        c,
        k::base(&sm.o),
        k::base(&rope_cos[layer.freq]),
        k::base(&rope_sin[layer.freq]),
        k::base(&sm.pos_q),
        m * heads,
        hd,
        rd,
        true,
    );
    k::snap8(c, k::base(&sm.o), m * heads * hd);
    let gdim = heads * hd / og;
    for i in 0..m {
        for g in 0..og {
            k::bf16_gemv(
                c,
                k::row16(&layer.wo_a, g * ol * gdim, ol * gdim),
                k::row(&sm.o, i * heads * hd + g * gdim, gdim),
                k::row(&sm.fl, i * og * ol + g * ol, ol),
                ol,
                gdim,
            );
        }
    }
    k::emit8_packed(
        c,
        k::base(&sm.fl),
        k::base(&sm.act_fl),
        k::base(&sm.act_fl_s),
        m * og * ol,
    );
    gemm_q8(
        c,
        &layer.wo_b,
        k::base(&sm.act_fl),
        k::base(&sm.act_fl_s),
        k::base(&sm.attn_out),
        m,
        dim,
        og * ol,
    );
    if let Some(comm) = comm.as_deref_mut() {
        for i in 0..m {
            let mut partial = vec![0.0f32; dim];
            sm.attn_out
                .index(i * dim..(i + 1) * dim)
                .copy_to(&mut partial)
                .map_err(cuda_err)?;
            let reduced = comm.allreduce(&partial)?;
            sm.attn_out
                .index(i * dim..(i + 1) * dim)
                .copy_from(&reduced)
                .map_err(cuda_err)?;
        }
    }
    Ok(())
}

/// The M-batch MoE sublayer: batched gate + shared expert, sequential
/// per-position routing + grouped-expert path on row pointers.
fn moe_step_m(
    c: &Cuda,
    cfg: &RefConfig,
    layer: &GpuLayer,
    sm: &mut ScratchM,
    s1: &mut Scratch,
    arena: Option<&crate::stream::HostArena>,
    m: usize,
    sh: Sharding,
    mut comm: Option<&mut (dyn Collectives + 'static)>,
) -> io::Result<()> {
    let dim = cfg.dim as usize;
    let inter = cfg.inter_dim as usize;
    let n_experts = cfg.n_routed as usize;
    let top_k = cfg.n_activated as usize;
    let world = sh.world_size as usize;
    let e_start = sh.rank as usize * n_experts / world;
    let e_len = n_experts / world;

    k::moe_gate_m(
        c,
        k::base(&sm.x),
        k::base(&layer.gate_w),
        k::base(&layer.gate_bias),
        k::base(&sm.gate_scores),
        m,
        dim,
        n_experts,
    );
    let _ = sm.y_moe.set_zero();
    k::emit8_packed(
        c,
        k::base(&sm.x),
        k::base(&sm.act_x),
        k::base(&sm.act_x_s),
        m * dim,
    );

    if world > 1 && layer.w1_all.is_none() && layer.w1_tl.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "decode_m: TP4 requires grouped expert slabs",
        ));
    }
    if layer.w1_tl.is_some() {
        let routes = m * top_k;
        let tiles = (routes + 31) / 32 + routes.min(e_len);
        for i in 0..m {
            k::moe_select(
                c,
                k::row(&sm.gate_scores, i * n_experts, n_experts),
                k::base(&layer.gate_bias),
                k::row(&s1.gate_w_out, i * top_k, top_k),
                k::row_i32(&s1.gate_i_out, i * top_k, top_k),
                n_experts,
                top_k,
                cfg.route_scale,
            );
        }
        let packed_a = k::U32::from_raw(
            sm.act_x_s.as_device_ptr().as_raw() + (m * dim / 32).next_multiple_of(16) as u64,
        );
        k::moe_dispatch_ids(
            c,
            k::base(&s1.gate_i_out),
            k::base(&s1.dispatch_idx),
            routes,
            e_start,
            e_len,
        );
        k::moe_dispatch_tl(
            c,
            k::base(&s1.dispatch_idx),
            k::base(&s1.gate_w_out),
            k::base(&s1.order),
            k::base(&s1.layout_weights),
            k::base(&s1.route_pos),
            k::base(&s1.tile_expert),
            k::base(&s1.expert_count),
            k::base(&s1.expert_offset),
            k::base(&s1.expert_psum),
            k::base(&s1.num_tiles),
            m,
            top_k,
            tiles,
        );
        k::moe_grouped_tl_cached(
            c,
            k::TlMoe::GateUp,
            k::base(&sm.act_x),
            packed_a,
            layer.w1_map.unwrap(),
            k::base(layer.w1s_tl.as_ref().unwrap()),
            k::base(&s1.order),
            k::base(&s1.tile_expert),
            k::base(&s1.route_h),
            tiles,
            m,
            top_k,
        );
        k::moe_grouped_tl_cached(
            c,
            k::TlMoe::GateUp,
            k::base(&sm.act_x),
            packed_a,
            layer.w3_map.unwrap(),
            k::base(layer.w3s_tl.as_ref().unwrap()),
            k::base(&s1.order),
            k::base(&s1.tile_expert),
            k::base(&s1.route_y),
            tiles,
            m,
            top_k,
        );
        k::moe_swiglu_route(
            c,
            k::base(&s1.route_h),
            k::base(&s1.route_y),
            k::base(&s1.layout_weights),
            k::base(&s1.route_h),
            inter,
            tiles * 32,
        );
        k::emit8_packed(
            c,
            k::base(&s1.route_h),
            k::base(&s1.route_q),
            k::base(&s1.route_q_s),
            tiles * 32 * inter,
        );
        let packed_route = k::U32::from_raw(
            s1.route_q_s.as_device_ptr().as_raw() + (tiles * inter).next_multiple_of(16) as u64,
        );
        k::moe_grouped_tl_cached(
            c,
            k::TlMoe::Down,
            k::base(&s1.route_q),
            packed_route,
            layer.w2_map.unwrap(),
            k::base(layer.w2s_tl.as_ref().unwrap()),
            k::base(&s1.order),
            k::base(&s1.tile_expert),
            k::base(&s1.route_y),
            tiles,
            m,
            top_k,
        );
        k::moe_combine_f32(
            c,
            k::base(&s1.route_y),
            k::base(&s1.route_pos),
            k::base(&sm.y_moe),
            dim,
            top_k,
            m,
        );
    } else if layer.w1_all.is_some() {
        // grouped-expert path (real checkpoint, upload mode)
        let w1 = layer.w1_all.as_ref().unwrap();
        let w1s = layer.w1s_all.as_ref().unwrap();
        let w2 = layer.w2_all.as_ref().unwrap();
        let w2s = layer.w2s_all.as_ref().unwrap();
        let w3 = layer.w3_all.as_ref().unwrap();
        let w3s = layer.w3s_all.as_ref().unwrap();
        for i in 0..m {
            k::moe_select(
                c,
                k::row(&sm.gate_scores, i * n_experts, n_experts),
                k::base(&layer.gate_bias),
                k::base(&s1.gate_w_out),
                k::base(&s1.gate_i_out),
                n_experts,
                top_k,
                cfg.route_scale,
            );
            k::moe_local_slots(
                c,
                k::base(&s1.gate_i_out),
                k::base(&s1.local_slots),
                k::base(&s1.local_experts),
                k::base(&s1.local_count),
                top_k,
                e_start,
                e_len,
            );
            k::expert_glu(
                c,
                k::row_u8(&sm.act_x, i * dim, dim),
                k::row_u8(&sm.act_x_s, i * dim / 32, dim / 32),
                k::base(w1),
                k::base(w1s),
                k::base(w3),
                k::base(w3s),
                k::base(&s1.gate_w_out),
                k::base(&s1.local_slots),
                k::base(&s1.local_experts),
                k::base(&s1.local_count),
                k::base(&s1.h_slots),
                inter,
                dim,
                top_k,
            );
            k::expert_quant(
                c,
                k::base(&s1.h_slots),
                k::base(&s1.local_slots),
                k::base(&s1.local_count),
                k::base(&s1.q_slots),
                k::base(&s1.qs_slots),
                inter,
                top_k,
            );
            k::expert_reduce(
                c,
                k::base(&s1.q_slots),
                k::base(&s1.qs_slots),
                k::base(w2),
                k::base(w2s),
                k::base(&s1.gate_i_out),
                k::base(&s1.local_slots),
                k::base(&s1.local_count),
                k::row(&sm.y_moe, i * dim, dim),
                dim,
                inter,
                top_k,
                e_start,
            );
        }
    } else {
        // per-expert path (synthetic gate and small uploads): read the
        // routing back, run each selected expert in ascending id order
        for i in 0..m {
            k::moe_select(
                c,
                k::row(&sm.gate_scores, i * n_experts, n_experts),
                k::base(&layer.gate_bias),
                k::base(&s1.gate_w_out),
                k::base(&s1.gate_i_out),
                n_experts,
                top_k,
                cfg.route_scale,
            );
            let mut w_host = vec![0.0f32; top_k];
            let mut i_host = vec![0i32; top_k];
            {
                let stream = c.stream.as_inner();
                s1.gate_stage.dtoh(
                    s1.gate_w_out.as_device_ptr().as_raw() as u64,
                    top_k * 4,
                    stream,
                )?;
                s1.gate_stage.dtoh_at(
                    s1.gate_i_out.as_device_ptr().as_raw() as u64,
                    top_k * 4,
                    top_k * 4,
                    stream,
                )?;
                s1.gate_stage.sync(stream)?;
                let f32s = s1.gate_stage.as_f32();
                w_host.copy_from_slice(&f32s[..top_k]);
                let i32s = s1.gate_stage.as_i32_at(top_k * 4);
                i_host.copy_from_slice(&i32s[..top_k]);
            }
            let mut sel: Vec<(usize, f32)> = i_host
                .iter()
                .map(|&i| i as usize)
                .zip(w_host.iter().copied())
                .collect();
            sel.sort_by_key(|&(e, _)| e); // ascending expert id (model.py:895)
            for &(e, w) in &sel {
                let [w1, w2, w3] = &layer.experts[e];
                let (w1d, w1sd) = (w1.w.stage(c, arena)?, w1.s.stage(c, arena)?);
                let (w3d, w3sd) = (w3.w.stage(c, arena)?, w3.s.stage(c, arena)?);
                k::fp4_gemm(
                    c,
                    k::row_u8(&sm.act_x, i * dim, dim),
                    k::row_u8(&sm.act_x_s, i * dim / 32, dim / 32),
                    w1.w.ptr_or(&w1d),
                    w1.s.ptr_or(&w1sd),
                    k::base(&s1.gate_buf),
                    1,
                    inter,
                    dim,
                );
                k::fp4_gemm(
                    c,
                    k::row_u8(&sm.act_x, i * dim, dim),
                    k::row_u8(&sm.act_x_s, i * dim / 32, dim / 32),
                    w3.w.ptr_or(&w3d),
                    w3.s.ptr_or(&w3sd),
                    k::base(&s1.up_buf),
                    1,
                    inter,
                    dim,
                );
                k::swiglu(
                    c,
                    k::base(&s1.gate_buf),
                    k::base(&s1.up_buf),
                    k::base(&s1.h_e),
                    inter,
                    w,
                );
                k::emit8_packed(
                    c,
                    k::base(&s1.h_e),
                    k::base(&s1.act_h),
                    k::base(&s1.act_h_s),
                    inter,
                );
                drop((w1d, w1sd, w3d, w3sd));
                let (w2d, w2sd) = (w2.w.stage(c, arena)?, w2.s.stage(c, arena)?);
                k::fp4_gemm(
                    c,
                    k::base(&s1.act_h),
                    k::base(&s1.act_h_s),
                    w2.w.ptr_or(&w2d),
                    w2.s.ptr_or(&w2sd),
                    k::base(&s1.expert_y),
                    1,
                    dim,
                    inter,
                );
                drop((w2d, w2sd));
                k::acc_f32(
                    c,
                    k::base(&s1.expert_y),
                    k::row(&sm.y_moe, i * dim, dim),
                    dim,
                );
            }
        }
    }

    if let Some(comm) = comm.as_deref_mut() {
        for i in 0..m {
            let mut partial = vec![0.0f32; dim];
            sm.y_moe
                .index(i * dim..(i + 1) * dim)
                .copy_to(&mut partial)
                .map_err(cuda_err)?;
            let reduced = comm.allreduce(&partial)?;
            sm.y_moe
                .index(i * dim..(i + 1) * dim)
                .copy_from(&reduced)
                .map_err(cuda_err)?;
        }
    }

    // shared expert, batched, added last per row
    let [w1, w2, w3] = &layer.shared_experts;
    let shared_inter = inter;
    let (gate, up) = {
        gemm_q8(
            c,
            w1,
            k::base(&sm.act_x),
            k::base(&sm.act_x_s),
            k::base(&sm.gate_buf),
            m,
            shared_inter,
            dim,
        );
        gemm_q8(
            c,
            w3,
            k::base(&sm.act_x),
            k::base(&sm.act_x_s),
            k::base(&sm.up_buf),
            m,
            shared_inter,
            dim,
        );
        (k::base(&sm.gate_buf), k::base(&sm.up_buf))
    };
    k::swiglu(c, gate, up, k::base(&sm.h_e), m * shared_inter, 1.0);
    k::emit8_packed(
        c,
        k::base(&sm.h_e),
        k::base(&sm.act_h),
        k::base(&sm.act_h_s),
        m * shared_inter,
    );
    gemm_q8(
        c,
        w2,
        k::base(&sm.act_h),
        k::base(&sm.act_h_s),
        k::base(&sm.expert_y),
        m,
        dim,
        shared_inter,
    );
    k::acc_f32(c, k::base(&sm.expert_y), k::base(&sm.y_moe), m * dim);

    Ok(())
}
