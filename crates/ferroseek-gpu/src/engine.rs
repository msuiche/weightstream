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
use cust::memory::DeviceBuffer;
use cust::prelude::*;
use ferroseek_core::loader::Checkpoint;
use ferroseek_core::ModelConfig;
use ferroseek_cuda::gpu::Cuda;
use ferroseek_ref::attn::{select_candidate_blocks, select_topk, window_topk_idxs};
use ferroseek_ref::config::RefConfig;
use ferroseek_ref::engram::{hash_ids, EngramLayout};
use ferroseek_ref::model::{validate_dims, validate_layout};
use ferroseek_ref::store::{bytes_to_f32, CkptStore, DType, SyntheticWeights, WeightStore};
use std::io;

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
        Sharding { world_size: 1, rank: 0 }
    }
}

type D<T> = DeviceBuffer<T>;

/// fp8 weight + ue8m0 scales, exactly the on-disk bytes.
struct Q8 {
    w: D<u8>,
    s: D<u8>,
}

/// Packed fp4 expert weight + per-1x32-K ue8m0 scales.
struct Q4 {
    w: D<u8>,
    s: D<u8>,
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
    embed_w: D<u8>,
    embed_s: D<u8>,
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
    wo_a: D<f32>,
    wo_b: Q8,
    gate_w: D<u16>,
    gate_bias: D<f32>,
    experts: Vec<[Q4; 3]>,
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
    h_e: D<f32>,
    expert_y: D<f32>,
    latent: D<f32>,
    krow: D<f32>,
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
struct RunState {
    compress_len: Vec<usize>,
    index_k_src: usize,
    candidates: Vec<bool>,
    shared_n_sel: usize,
    /// Debug: when Some, every attention sublayer records (layer, x).
    capture_x: Option<Vec<(u32, Vec<f32>)>>,
}

/// The GPU decode engine.
pub struct GpuModel {
    pub cfg: RefConfig,
    layers: Vec<GpuLayer>,
    caches: GpuCaches,
    engram: EngramLayout,
    eng_cache: Vec<i64>,
    token_map: Option<Vec<i64>>,
    /// [plain, compress] rope tables: [max_pos, rope_dim/2].
    rope_cos: [D<f32>; 2],
    rope_sin: [D<f32>; 2],
    emb: D<f32>,
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
}

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

fn up_zero_f32(c: &Cuda, n: usize) -> io::Result<D<f32>> {
    DeviceBuffer::zeroed(n.max(1)).map_err(cuda_err)
}

fn load_q8(store: &mut dyn WeightStore, c: &Cuda, wname: &str, sname: &str) -> io::Result<Q8> {
    Ok(Q8 { w: up_u8(c, &view_bytes(store, wname)?)?, s: up_u8(c, &view_bytes(store, sname)?)? })
}

fn load_q4(store: &mut dyn WeightStore, c: &Cuda, wname: &str, sname: &str) -> io::Result<Q4> {
    Ok(Q4 { w: up_u8(c, &view_bytes(store, wname)?)?, s: up_u8(c, &view_bytes(store, sname)?)? })
}

/// wo_a: bf16 dense (converted export, spec §11) or fp8+scale (raw HF
/// export) — dequantize host-side to f32 either way, like the ref's dense
/// path.
fn load_wo_a(store: &mut dyn WeightStore, c: &Cuda, name: &str, rows: usize, cols: usize) -> io::Result<D<f32>> {
    let v = store.get(name)?;
    let dtype = v.dtype;
    let bytes = v.bytes.to_vec();
    drop(v);
    if bytes.len() != rows * cols * dtype_size(dtype) {
        return ioerr(format!("{name}: wo_a byte length {} does not match [{rows}, {cols}]", bytes.len()));
    }
    match dtype {
        DType::Bf16 => {
            let host: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|b| ferroseek_core::quant::bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            up_f32(c, &host)
        }
        DType::F32 => {
            let host: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            up_f32(c, &host)
        }
        DType::Fp8E4m3 => {
            let sname = format!("{}.scale", name.trim_end_matches(".weight"));
            let sbytes = view_bytes(store, &sname)?;
            // scale shape is [rows/B, cols/B] with B in {32, 128} (spec §12.9)
            let host = ferroseek_core::quant::dequant_fp8_blockwise(&bytes, &sbytes, rows, cols);
            up_f32(c, &host)
        }
        d => ioerr(format!("{name}: unexpected wo_a dtype {d:?}")),
    }
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
) -> io::Result<GpuLayer> {
    let p = format!("layers.{l}");
    let dim = cfg.dim as usize;
    let hd = cfg.head_dim as usize;
    let inh = cfg.index_n_heads as usize;
    let ihd = cfg.index_head_dim as usize;
    let ratio = cfg.compress_ratios[l];
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
            embed_w: up_u8(c, &view_bytes(store, &format!("{p}.engram.embed.weight"))?)?,
            embed_s: up_u8(c, &view_bytes(store, &format!("{p}.engram.embed.scale"))?)?,
            wkv: load_q8(store, c, &format!("{p}.engram.wkv.weight"), &format!("{p}.engram.wkv.scale"))?,
            w_prod: up_f32(c, &prod)?,
        })
    } else {
        None
    };

    let compressor = match kv_src {
        Some(_) => {
            let wkv = up_f32(c, &view_f32(store, &format!("{p}.attn.compressor.wkv.weight"))?)?;
            let wgate = if ratio > 1 {
                Some(up_f32(c, &view_f32(store, &format!("{p}.attn.compressor.wgate.weight"))?)?)
            } else {
                None
            };
            Some(GpuCompressor {
                wkv,
                wgate,
                norm_w: up_f32(c, &view_f32(store, &format!("{p}.attn.compressor.norm.weight"))?)?,
            })
        }
        None => None,
    };

    let indexer = if idx_src {
        Some(GpuIndexer {
            wq_b: load_q8(store, c, &format!("{p}.attn.indexer.wq_b.weight"), &format!("{p}.attn.indexer.wq_b.scale"))?,
            weights_proj: {
                let w = view_f32(store, &format!("{p}.attn.indexer.weights_proj.weight"))?;
                let f = (ihd as f32) * (inh as f32);
                let scaled: Vec<f32> = w.iter().map(|&v| v / f.sqrt()).collect();
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
                v.bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect::<Vec<u16>>()
            })?,
            k_norm: up_f32(c, &view_f32(store, &format!("{p}.attn.indexer.k_norm.weight"))?)?,
        }),
        None => None,
    };

    let mut experts = Vec::with_capacity(cfg.n_routed as usize);
    for e in 0..cfg.n_routed as usize {
        let w1 = load_q4(store, c, &format!("{p}.ffn.experts.{e}.w1.weight"), &format!("{p}.ffn.experts.{e}.w1.scale"))?;
        let w2 = load_q4(store, c, &format!("{p}.ffn.experts.{e}.w2.weight"), &format!("{p}.ffn.experts.{e}.w2.scale"))?;
        let w3 = load_q4(store, c, &format!("{p}.ffn.experts.{e}.w3.weight"), &format!("{p}.ffn.experts.{e}.w3.scale"))?;
        experts.push([w1, w2, w3]);
    }

    Ok(GpuLayer {
        layer,
        ratio,
        kv_src,
        idx_src,
        owns_k: idx_src && kv_src.is_some(),
        cand_src: cfg.candidate_source >= 0 && (layer as i64) == cfg.candidate_source && idx_src,
        uses_cand: cfg.candidate_source >= 0 && (layer as i64) > cfg.candidate_source && idx_src,
        engram: cfg.engram_layers.iter().position(|&e| e == layer),
        freq: if ratio > 0 { 1 } else { 0 },
        attn_norm_w: up_f32(c, &view_f32(store, &format!("{p}.attn_norm.weight"))?)?,
        ffn_norm_w: up_f32(c, &view_f32(store, &format!("{p}.ffn_norm.weight"))?)?,
        hc_attn_fn: up_f32(c, &view_f32(store, &format!("{p}.hc_attn_fn"))?)?,
        hc_ffn_fn: up_f32(c, &view_f32(store, &format!("{p}.hc_ffn_fn"))?)?,
        hc_attn_base: up_f32(c, &view_f32(store, &format!("{p}.hc_attn_base"))?)?,
        hc_ffn_base: up_f32(c, &view_f32(store, &format!("{p}.hc_ffn_base"))?)?,
        hc_attn_scale: up_f32(c, &view_f32(store, &format!("{p}.hc_attn_scale"))?)?,
        hc_ffn_scale: up_f32(c, &view_f32(store, &format!("{p}.hc_ffn_scale"))?)?,
        wq_a: load_q8(store, c, &format!("{p}.attn.wq_a.weight"), &format!("{p}.attn.wq_a.scale"))?,
        q_norm_w: up_f32(c, &view_f32(store, &format!("{p}.attn.q_norm.weight"))?)?,
        wq_b: load_q8(store, c, &format!("{p}.attn.wq_b.weight"), &format!("{p}.attn.wq_b.scale"))?,
        wkv: load_q8(store, c, &format!("{p}.attn.wkv.weight"), &format!("{p}.attn.wkv.scale"))?,
        kv_norm_w: up_f32(c, &view_f32(store, &format!("{p}.attn.kv_norm.weight"))?)?,
        attn_sink: up_f32(c, &view_f32(store, &format!("{p}.attn.attn_sink"))?)?,
        wo_a: load_wo_a(store, c, &format!("{p}.attn.wo_a.weight"), cfg.o_groups as usize * cfg.o_lora as usize, cfg.n_heads as usize * hd / cfg.o_groups as usize)?,
        wo_b: load_q8(store, c, &format!("{p}.attn.wo_b.weight"), &format!("{p}.attn.wo_b.scale"))?,
        gate_w: up_u16(c, &{
            let v = store.get(&format!("{p}.ffn.gate.weight"))?;
            let n = cfg.n_routed as usize * dim;
            if v.bytes.len() != n * 2 {
                return ioerr("gate weight size mismatch".into());
            }
            v.bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect::<Vec<u16>>()
        })?,
        gate_bias: up_f32(c, &view_f32(store, &format!("{p}.ffn.gate.bias"))?)?,
        experts,
        shared_experts: [
            load_q8(store, c, &format!("{p}.ffn.shared_experts.w1.weight"), &format!("{p}.ffn.shared_experts.w1.scale"))?,
            load_q8(store, c, &format!("{p}.ffn.shared_experts.w2.weight"), &format!("{p}.ffn.shared_experts.w2.scale"))?,
            load_q8(store, c, &format!("{p}.ffn.shared_experts.w3.weight"), &format!("{p}.ffn.shared_experts.w3.scale"))?,
        ],
        compressor,
        indexer,
        k_owner,
        engram_w,
    })
}

// ---------------------------------------------------------------------------
// build

impl GpuModel {
    /// Load from a lazily-mmapped checkpoint. Weights upload in their
    /// on-disk quantized formats; wo_a dequantizes on the host; bf16/f32
    /// params upload as f32 (embedding/head) or stay bf16 where the kernel
    /// takes bf16 (gate).
    pub fn from_checkpoint(ckpt: &mut Checkpoint) -> io::Result<Self> {
        let cfg = RefConfig::from_model_config(&ckpt.config)?;
        let mut store: Box<dyn WeightStore> = Box::new(CkptStore::new(ckpt));
        validate_dims(&cfg)?;
        validate_layout(&cfg, store.as_mut())?;
        Self::build(cfg, store.as_mut(), DEFAULT_MAX_SEQ_LEN)
    }

    /// Load from deterministically generated synthetic weights — the gate:
    /// the same weights go through `RefModel::from_synthetic` on the CPU.
    pub fn from_synthetic(cfg: ModelConfig, weights: SyntheticWeights) -> io::Result<Self> {
        let rcfg = RefConfig::from_model_config(&cfg)?;
        let mut store: Box<dyn WeightStore> = Box::new(weights);
        validate_dims(&rcfg)?;
        Self::build(rcfg, store.as_mut(), 1024)
    }

    /// Raise the decode-position bound (must be set before the first step).
    pub fn with_max_seq_len(mut self, n: u64) -> Self {
        assert_eq!(self.pos, 0, "caches are preallocated at build time");
        self.max_seq = n;
        self
    }

    fn build(cfg: RefConfig, store: &mut dyn WeightStore, max_seq: u64) -> io::Result<Self> {
        let guard = Cuda::global();
        let c = guard.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "cuda init"))?;

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
                    if cfg.compress_ratios[l] != cfg.compress_ratios[cfg.kv_sources[pord] as usize] {
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

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            layers.push(upload_layer(&cfg, store, c, l, kv_src_of[l])?);
        }

        // embedding + head as f32 (spec §11: head held fp32; embed is
        // row-gathered)
        let emb = up_f32(c, &view_f32(store, "embed.weight")?)?;
        let head = up_f32(c, &view_f32(store, "head.weight")?)?;
        let final_norm = up_f32(c, &view_f32(store, "norm.weight")?)?;

        // caches: per layer kv_all = ring + compressed view; per source
        // index_k + compressor state
        let mut kv_all = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ratio = cfg.compress_ratios[l] as u64;
            let rows = if ratio > 0 { win + (max_seq / ratio + 1) as usize } else { win };
            kv_all.push(up_zero_f32(c, rows * hd)?);
        }
        let mut index_k = Vec::new();
        let mut comp_kv = Vec::new();
        let mut comp_score = Vec::new();
        for &s in &cfg.kv_sources {
            let ratio = cfg.compress_ratios[s as usize] as u64;
            index_k.push(up_zero_f32(c, (max_seq / ratio + 1) as usize * cfg.index_head_dim as usize)?);
            if ratio > 1 {
                comp_kv.push(Some(up_zero_f32(c, ratio as usize * hd)?));
                let score = vec![f32::NEG_INFINITY; ratio as usize * hd];
                comp_score.push(Some(up_f32(c, &score)?));
            } else {
                comp_kv.push(None);
                comp_score.push(None);
            }
        }

        // scratch
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
        };

        let n_sources = cfg.kv_sources.len();
        Ok(GpuModel {
            cfg,
            layers,
            caches: GpuCaches { kv_all, index_k, comp_kv, comp_score },
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
            sharding: Sharding::default(),
        })
    }

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
        let row = self.caches.kv_all[layer as usize].index(win * hd + j * hd..win * hd + (j + 1) * hd);
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
                format!("decode expects sequential positions: got {position}, next is {}", self.pos),
            ));
        }
        if position >= self.max_seq {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "position beyond max_seq_len"));
        }
        self.pos += 1;
        if token >= self.cfg.vocab {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "token id out of vocab"));
        }
        let guard = Cuda::global();
        let c = guard.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "cuda init"))?;
        crate::dump::set_pos(position);

        // engram hashes, host-side (the same ref code path)
        let hashes = if self.engram.layer_ids.is_empty() {
            Vec::new()
        } else {
            let token_map = self.token_map.clone();
            hash_ids(&self.cfg, &self.engram, &mut self.eng_cache, token_map.as_deref(), token, position)
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
            ..
        } = self;

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
            let idx = [token as i64];
            let d_idx = DeviceBuffer::from_slice(&idx).map_err(cuda_err)?;
            k::gather_f32(c, k::base(emb), k::base(&d_idx), k::base(&s.emb_row), 1, dim);
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
            if let Some(ei) = layer.engram {
                let e = layer.engram_w.as_ref().unwrap();
                let hash = &hashes[ei];
                let ids: Vec<i64> = hash.iter().map(|&h| h as i64).collect();
                s.hash_ids.copy_from(&ids).map_err(cuda_err)?;
                let ncols = engram.n_hash_cols;
                let ehd = engram.head_dim;
                k::gather_fp8(c, k::base(&e.embed_w), k::base(&e.embed_s), k::base(&s.hash_ids), k::base(&s.eflat), ncols, ehd);
                k::emit8(c, k::base(&s.eflat), k::base(&s.act_e), k::base(&s.act_e_s), ncols * ehd);
                k::fp8_gemm(
                    c,
                    k::base(&s.act_e),
                    k::base(&s.act_e_s),
                    k::base(&e.wkv.w),
                    k::base(&e.wkv.s),
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

            // attention sublayer wiring (spec §2, model.py:970-996)
            let hc_dim = hc * dim;
            let mix_hc = cfg.mix_hc();
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
            k::hc_pre(c, k::base(&s.h[h_cur]), k::base(&s.pre_ffn), k::base(&s.x), 1, dim);
            k::rmsnorm(c, k::base(&s.x), k::base(&layer.attn_norm_w), k::base(&s.x), dim, eps);

            attention_step(c, cfg, layer, s, caches, run, bands, &*rope_cos, &*rope_sin, position)?;

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
            k::hc_pre(c, k::base(&s.h[h_cur]), k::base(&s.pre_attn), k::base(&s.x), 1, dim);
            k::rmsnorm(c, k::base(&s.x), k::base(&layer.ffn_norm_w), k::base(&s.x), dim, eps);

            moe_step(c, cfg, layer, s)?;

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
        k::hc_pre(c, k::base(&s.h[h_cur]), k::base(&s.pre_ffn), k::base(&s.x), 1, dim);
        k::rmsnorm(c, k::base(&s.x), k::base(final_norm), k::base(&s.x), dim, eps);
        k::f32_gemv(c, k::base(head), k::base(&s.x), k::base(&s.logits), cfg.vocab as usize, dim);
        let mut out = vec![0.0f32; cfg.vocab as usize];
        s.logits.copy_to(&mut out).map_err(cuda_err)?;
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
        let c = guard.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "cuda init"))?;
        let s = &self.scratch;
        k::argmax(c, k::base(&s.logits), k::base(&s.argmax_i), k::base(&s.argmax_v), self.cfg.vocab as usize);
        let mut idx = [0i32; 1];
        s.argmax_i.copy_to(&mut idx).map_err(cuda_err)?;
        Ok(idx[0] as u32)
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
) -> io::Result<()> {
    let dim = cfg.dim as usize;
    let hd = cfg.head_dim as usize;
    let rd = cfg.rope_dim as usize;
    let heads = cfg.n_heads as usize;
    let og = cfg.o_groups as usize;
    let ol = cfg.o_lora as usize;
    let win = cfg.window_size as usize;
    let eps = cfg.norm_eps;
    let l = layer.layer as usize;

    if run.capture_x.is_some() {
        let mut v = vec![0.0f32; dim];
        s.x.index(0..dim).copy_to(&mut v).map_err(cuda_err)?;
        run.capture_x.as_mut().unwrap().push((layer.layer, v));
    }

    // q path: wq_a -> q_norm -> wq_b -> rope (model.py:771-773)
    k::emit8(c, k::base(&s.x), k::base(&s.act_x), k::base(&s.act_x_s), dim);
    dump_u8(c, &s.act_x, dim, format!("d_pos{pos}_actx{l}.u8"));
    dump_u8(c, &s.act_x_s, dim / 32, format!("d_pos{pos}_actxs{l}.u8"));
    k::fp8_gemm(c, k::base(&s.act_x), k::base(&s.act_x_s), k::base(&layer.wq_a.w), k::base(&layer.wq_a.s), k::base(&s.qr), 1, cfg.q_lora as usize, dim);
    k::rmsnorm(c, k::base(&s.qr), k::base(&layer.q_norm_w), k::base(&s.qr), cfg.q_lora as usize, eps);
    k::emit8(c, k::base(&s.qr), k::base(&s.act_qr), k::base(&s.act_qr_s), cfg.q_lora as usize);
    dump_dev(c, &s.qr, cfg.q_lora as usize, format!("d_pos{pos}_qr{l}.f32"));
    dump_u8(c, &s.act_qr, cfg.q_lora as usize, format!("d_pos{pos}_actqr{l}.u8"));
    dump_u8(c, &s.act_qr_s, cfg.q_lora as usize / 32, format!("d_pos{pos}_actqrs{l}.u8"));
    k::fp8_gemm(c, k::base(&s.act_qr), k::base(&s.act_qr_s), k::base(&layer.wq_b.w), k::base(&layer.wq_b.s), k::base(&s.q), 1, heads * hd, cfg.q_lora as usize);
    k::rope_uniform_pos(c, k::base(&s.q), k::base(&rope_cos[layer.freq]), k::base(&rope_sin[layer.freq]), heads, hd, rd, pos, false);
    dump_dev(c, &s.q, heads * hd, format!("d_pos{pos}_q{l}.f32"));

    // window KV: one shared latent, rope tail, fp8 snap, ring write
    k::fp8_gemm(c, k::base(&s.act_x), k::base(&s.act_x_s), k::base(&layer.wkv.w), k::base(&layer.wkv.s), k::base(&s.kvrow), 1, hd, dim);
    let slot = (pos % win as u64) as usize;
    let ring = k::row(&caches.kv_all[l], slot * hd, hd);
    k::rmsnorm(c, k::base(&s.kvrow), k::base(&layer.kv_norm_w), ring, hd, eps);
    k::rope(c, ring, k::base(&rope_cos[layer.freq]), k::base(&rope_sin[layer.freq]), 1, hd, rd, pos, false);
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
                k::f32_gemv(c, k::base(&comp.wkv), k::base(&s.x), k::base(&s.latent), hd, dim);
                k::rmsnorm(c, k::base(&s.latent), k::base(&comp.norm_w), k::base(&s.latent), hd, eps);
                latent_ready = true;
            } else {
                let slot_i = (pos % ratio) as usize;
                let kv_state = caches.comp_kv[src].as_ref().unwrap();
                let score_state = caches.comp_score[src].as_mut().unwrap();
                k::f32_gemv(c, k::base(&comp.wkv), k::base(&s.x), k::row(kv_state, slot_i * hd, hd), hd, dim);
                k::f32_gemv(c, k::base(comp.wgate.as_ref().unwrap()), k::base(&s.x), k::row(score_state, slot_i * hd, hd), hd, dim);
                if (pos + 1) % ratio == 0 {
                    k::compressor_finish(c, k::base(kv_state), k::base(score_state), k::base(&s.latent), hd);
                    // RMS norm of the pooled latent (model.py:485; the
                    // ratio-1 branch norms right after its gemv)
                    k::rmsnorm(c, k::base(&s.latent), k::base(&comp.norm_w), k::base(&s.latent), hd, eps);
                    latent_ready = true;
                }
            }
            if latent_ready {
                run.compress_len[src] = compress_len;
            }
        }

        // DSA indexer (index sources run their own; consumers reuse the
        // published idxs — model.py:723-738)
        if layer.idx_src {
            let inh = cfg.index_n_heads as usize;
            let ihd = cfg.index_head_dim as usize;
            // key side: append on group completion, pre-RoPE latent
            if layer.owns_k && latent_ready {
                let kowner = layer.k_owner.as_ref().unwrap();
                let j = compress_len - 1;
                k::bf16_gemv(c, k::base(&kowner.wk), k::base(&s.latent), k::base(&s.krow), ihd, hd);
                let krow_dst = k::row(&caches.index_k[run.index_k_src], j * ihd, ihd);
                k::rmsnorm(c, k::base(&s.krow), k::base(&kowner.k_norm), krow_dst, ihd, eps);
                k::rope(c, krow_dst, k::base(&rope_cos[layer.freq]), k::base(&rope_sin[layer.freq]), 1, ihd, rd, pos + 1 - ratio, false);
                k::fp4_snap(c, krow_dst, ihd, 32, false);
            }
            // query side: wq_b(qr) at pos, fp4(32, e8m0) snap
            k::fp8_gemm(c, k::base(&s.act_qr), k::base(&s.act_qr_s), k::base(&layer.indexer.as_ref().unwrap().wq_b.w), k::base(&layer.indexer.as_ref().unwrap().wq_b.s), k::base(&s.qidx), 1, inh * ihd, cfg.q_lora as usize);
            k::rope_uniform_pos(c, k::base(&s.qidx), k::base(&rope_cos[layer.freq]), k::base(&rope_sin[layer.freq]), inh, ihd, rd, pos, false);
            k::fp4_snap(c, k::base(&s.qidx), inh * ihd, 32, false);
            // per-head weights from weights_proj (scale folded at load)
            k::f32_gemv(c, k::base(&layer.indexer.as_ref().unwrap().weights_proj), k::base(&s.x), k::base(&s.widx), inh, dim);

            // score + selection
            let mut idxs_host;
            if compress_len == 0 {
                idxs_host = window_topk_idxs(win, pos);
            } else {
                k::indexer_score(c, k::base(&s.qidx), k::base(&caches.index_k[run.index_k_src]), k::base(&s.widx), k::base(&s.score), inh, ihd, compress_len);
                let mut score = vec![0.0f32; compress_len];
                s.score.index(0..compress_len).copy_to(&mut score).map_err(cuda_err)?;
                if layer.cand_src {
                    run.candidates = select_candidate_blocks(&score, cfg.candidate_topk_blocks as usize, cfg.candidate_block_size as usize);
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
            s.shared_idxs.index(0..idxs_host.len()).copy_from(&idxs_host).map_err(cuda_err)?;
            run.shared_n_sel = idxs_host.len();
        }

        // latent tail: rope at the group's first-token position, fp4(16,
        // e4m3) snap, fan the row out to every layer in the publish band
        // (each layer's kv_all carries its own compressed view)
        if latent_ready {
            if let Some(src) = layer.kv_src {
                let j = compress_len - 1;
                k::rope(c, k::base(&s.latent), k::base(&rope_cos[layer.freq]), k::base(&rope_sin[layer.freq]), 1, hd, rd, pos + 1 - ratio, false);
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
    dump_dev(c, &caches.kv_all[l], caches.kv_all[l].len(), format!("d_pos{pos}_kvall{l}.f32"));
    k::sparse_attn(c, k::base(&s.q), k::base(&caches.kv_all[l]), idxs, k::base(&layer.attn_sink), k::base(&s.o), heads, hd, n_sel, (hd as f32).sqrt().recip());
    dump_dev(c, &s.o, heads * hd, format!("d_pos{pos}_attnraw{l}.f32"));

    // inverse rope on the output tail (the cache stays in the query-rotated
    // form), fp8 snap per group row, block-diagonal wo_a, wo_b
    k::rope_uniform_pos(c, k::base(&s.o), k::base(&rope_cos[layer.freq]), k::base(&rope_sin[layer.freq]), heads, hd, rd, pos, true);
    let gdim = heads * hd / og;
    for g in 0..og {
        k::snap8(c, k::row(&s.o, g * gdim, gdim), gdim);
    }
    dump_dev(c, &s.o, heads * hd, format!("d_pos{pos}_osnap{l}.f32"));
    for g in 0..og {
        k::f32_gemv(
            c,
            k::row(&layer.wo_a, g * ol * gdim, ol * gdim),
            k::row(&s.o, g * gdim, gdim),
            k::row(&s.fl, g * ol, ol),
            ol,
            gdim,
        );
    }
    k::emit8(c, k::base(&s.fl), k::base(&s.act_fl), k::base(&s.act_fl_s), og * ol);
    dump_dev(c, &s.fl, og * ol, format!("d_pos{pos}_fl{l}.f32"));
    k::fp8_gemm(c, k::base(&s.act_fl), k::base(&s.act_fl_s), k::base(&layer.wo_b.w), k::base(&layer.wo_b.s), k::base(&s.attn_out), 1, dim, og * ol);
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
fn moe_step(c: &Cuda, cfg: &RefConfig, layer: &GpuLayer, s: &mut Scratch) -> io::Result<()> {
    let l = layer.layer;
    let dim = cfg.dim as usize;
    let inter = cfg.inter_dim as usize;
    let n_experts = cfg.n_routed as usize;
    let top_k = cfg.n_activated as usize;

    k::moe_route(c, k::base(&s.x), k::base(&layer.gate_w), k::base(&layer.gate_bias), k::base(&s.gate_w_out), k::base(&s.gate_i_out), dim, n_experts, top_k, cfg.route_scale);
    // TODO(gpu): batched expert GEMMs without host readback
    let mut w_host = vec![0.0f32; top_k];
    let mut i_host = vec![0i32; top_k];
    s.gate_w_out.copy_to(&mut w_host).map_err(cuda_err)?;
    s.gate_i_out.copy_to(&mut i_host).map_err(cuda_err)?;

    crate::dump::i32s(
        format!("d_pos{}_gorder{l}.i32", crate::dump::pos()),
        &i_host,
    );
    crate::dump::f32s(
        format!("d_pos{}_gweights{l}.f32", crate::dump::pos()),
        &w_host,
    );
    let mut sel: Vec<(usize, f32)> = i_host.iter().map(|&i| i as usize).zip(w_host.iter().copied()).collect();
    sel.sort_by_key(|&(e, _)| e); // ascending expert id (model.py:895 loop order)

    let _ = s.y_moe.set_zero();
    // the activation is quantized once and reused by every expert and the
    // shared expert (identical to the ref's per-gemv quantization of x)
    k::emit8(c, k::base(&s.x), k::base(&s.act_x), k::base(&s.act_x_s), dim);

    for &(e, w) in &sel {
        let [w1, w2, w3] = &layer.experts[e];
        k::fp4_gemm(c, k::base(&s.act_x), k::base(&s.act_x_s), k::base(&w1.w), k::base(&w1.s), k::base(&s.gate_buf), 1, inter, dim);
        k::fp4_gemm(c, k::base(&s.act_x), k::base(&s.act_x_s), k::base(&w3.w), k::base(&w3.s), k::base(&s.up_buf), 1, inter, dim);
        k::swiglu(c, k::base(&s.gate_buf), k::base(&s.up_buf), k::base(&s.h_e), inter, w);
        k::emit8(c, k::base(&s.h_e), k::base(&s.act_h), k::base(&s.act_h_s), inter);
        k::fp4_gemm(c, k::base(&s.act_h), k::base(&s.act_h_s), k::base(&w2.w), k::base(&w2.s), k::base(&s.expert_y), 1, dim, inter);
        k::acc_f32(c, k::base(&s.expert_y), k::base(&s.y_moe), dim);
    }

    // shared expert on every token, added last (model.py:905)
    let [w1, w2, w3] = &layer.shared_experts;
    k::fp8_gemm(c, k::base(&s.act_x), k::base(&s.act_x_s), k::base(&w1.w), k::base(&w1.s), k::base(&s.gate_buf), 1, inter, dim);
    k::fp8_gemm(c, k::base(&s.act_x), k::base(&s.act_x_s), k::base(&w3.w), k::base(&w3.s), k::base(&s.up_buf), 1, inter, dim);
    k::swiglu(c, k::base(&s.gate_buf), k::base(&s.up_buf), k::base(&s.h_e), inter, 1.0);
    k::emit8(c, k::base(&s.h_e), k::base(&s.act_h), k::base(&s.act_h_s), inter);
    k::fp8_gemm(c, k::base(&s.act_h), k::base(&s.act_h_s), k::base(&w2.w), k::base(&w2.s), k::base(&s.expert_y), 1, dim, inter);
    k::acc_f32(c, k::base(&s.expert_y), k::base(&s.y_moe), dim);
    dump_dev(c, &s.y_moe, dim, format!("d_pos{}_y{l}.f32", crate::dump::pos()));
    Ok(())
}
