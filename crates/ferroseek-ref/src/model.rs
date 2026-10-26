//! [`RefModel`]: the CPU reference decode engine. Single sequence (B=1),
//! one token per call, f32 throughout. The weight strategy is a struct
//! field (a `Box<dyn WeightStore>`), so tests inject synthetic weights
//! without any checkpoint on disk.

use crate::config::RefConfig;
use crate::engram::{self, EngramLayout};
use crate::hc;
use crate::moe;
use crate::num;
use crate::rope::Freqs;
use crate::store::{bf16_row, CkptStore, DenseLru, SyntheticWeights, WeightStore};
use ferroseek_core::loader::Checkpoint;
use ferroseek_core::ModelConfig;
use std::io;

/// Per-layer static wiring derived from the config (spec §0/§3-§5).
pub struct LayerInfo {
    pub layer: u32,
    /// compress_ratios[layer]: 0 = window only, 1 = full, 2 = CSA 2:1.
    pub ratio: u32,
    /// Ordinal into `Runtime::sources` when this layer owns a compressor.
    pub kv_src: Option<usize>,
    /// This layer runs its own DSA indexer.
    pub idx_src: bool,
    /// Index keys owned (kv-source layer that is an index source; spec §4).
    pub owns_k: bool,
    /// Publishes the candidate block mask (candidate_source_layer).
    pub cand_src: bool,
    /// Masks its own indexer scores with the published candidates.
    pub uses_cand: bool,
    /// Index into the engram layout when the layer carries an engram.
    pub engram_idx: Option<usize>,
    /// The layer's rope variant (theta + YaRN by ratio; spec §3).
    pub freqs: Freqs,
}

/// Per kv-source runtime state (spec §5): the compressed KV cache, the
/// indexer key cache, and the CSA open-pair buffers.
pub struct SourceState {
    /// Flat `[rows][head_dim]` post-RoPE fp4(16, e4m3)-snapped latents.
    pub compress: Vec<f32>,
    /// Flat `[rows][index_head_dim]` post-RoPE fp4(32, e8m0)-snapped keys.
    pub index_k: Vec<f32>,
    /// ratio > 1 only: `[ratio][head_dim]` fp32 open-pair buffers.
    pub kv_state: Option<Vec<f32>>,
    pub score_state: Option<Vec<f32>>,
}

/// Shared attention runtime: one slot per published artifact, exactly as
/// model.py:1169-1183 — a source writes before its consumers read and
/// nothing resets between steps.
pub struct Runtime {
    /// Per layer: `[window_size][head_dim]` ring of fp8-snapped latents.
    pub window: Vec<Vec<f32>>,
    pub sources: Vec<SourceState>,
    /// Which source's compressed KV `shared_attn.compress_kv` points at.
    pub compress_slot: usize,
    pub index_k_slot: usize,
    /// Published topk indices (consumers reuse verbatim; spec §4).
    pub topk_idxs: Vec<i32>,
    /// Published candidate block mask (positions the later indexers may keep).
    pub candidates: Vec<bool>,
    /// Compressed token ids per position (engram hash cache).
    pub engram_ids: Vec<i64>,
    /// Tokenizer-derived compressed map (spec §11); identity until set.
    pub token_map: Option<Vec<i64>>,
    /// Next expected decode position.
    pub pos: u64,
    /// Debug: when Some, every attention sublayer records (layer, x).
    pub capture_x: Option<Vec<(u32, Vec<f32>)>>,
}

/// Borrow bundle passed through the layer pipeline.
pub struct Ctx<'a> {
    pub store: &'a mut dyn WeightStore,
    pub lru: &'a mut DenseLru,
}

impl Ctx<'_> {
    /// Dense f32 view of a bf16/f32 tensor (LRU-cached conversion).
    pub fn dense(&mut self, name: &str) -> io::Result<&[f32]> {
        self.lru.get(self.store, name)
    }

    pub fn get(&mut self, name: &str) -> io::Result<crate::store::TensorView<'_>> {
        self.store.get(name)
    }

    pub fn get2(
        &mut self,
        a: &str,
        b: &str,
    ) -> io::Result<(crate::store::TensorView<'_>, crate::store::TensorView<'_>)> {
        self.store.get2(a, b)
    }
}

/// The CPU reference decode engine (numerics gate).
pub struct RefModel<'a> {
    /// Flattened decode config (spec §0).
    pub cfg: RefConfig,
    store: Box<dyn WeightStore + 'a>,
    lru: DenseLru,
    layers: Vec<LayerInfo>,
    engram: EngramLayout,
    rt: Runtime,
}

/// Default budget for dequantized dense (bf16/f32) tensors. The full model is
/// ~475 GB quantized and is NEVER densified; only the bf16/f32 subset is,
/// and it is capped here.
pub const DEFAULT_DENSE_BUDGET: usize = 8 << 30;

impl<'a> RefModel<'a> {
    /// Build from a lazily-mmapped checkpoint. The checkpoint stays alive
    /// outside the model (mmap is kept); tensors are dequantized on demand.
    pub fn from_checkpoint(ckpt: &'a mut Checkpoint) -> io::Result<Self> {
        let cfg = RefConfig::from_model_config(&ckpt.config)?;
        let mut model = Self::build(cfg, Box::new(CkptStore::new(ckpt)))?;
        model.validate_layout()?;
        Ok(model)
    }

    /// Build from deterministically generated synthetic weights (tests).
    pub fn from_synthetic(
        cfg: ModelConfig,
        weights: SyntheticWeights,
    ) -> io::Result<RefModel<'static>> {
        let rcfg = RefConfig::from_model_config(&cfg)?;
        let model = RefModel::build(rcfg, Box::new(weights))?;
        Ok(model)
    }

    fn build(cfg: RefConfig, store: Box<dyn WeightStore + 'a>) -> io::Result<Self> {
        Self::validate_dims(&cfg)?;
        let engram = EngramLayout::build(&cfg)?;
        // table rows must cover every (offset + prime) range (spec §12.5)
        for (i, need) in engram.required_rows().iter().enumerate() {
            let have = cfg.engram_num_embeddings[i];
            if have < *need {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "engram layer {} needs {need} rows, config says {have}",
                        cfg.engram_layers[i]
                    ),
                ));
            }
        }
        let hd = cfg.head_dim as usize;
        let win = cfg.window_size as usize;
        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        let mut sources = Vec::new();
        for l in 0..cfg.n_layers {
            let ratio = cfg.compress_ratios[l as usize];
            let kv_src = cfg
                .kv_sources
                .iter()
                .position(|&s| s == l)
                .filter(|_| ratio > 0);
            if kv_src.is_some() {
                sources.push(SourceState {
                    compress: Vec::new(),
                    index_k: Vec::new(),
                    kv_state: if ratio > 1 {
                        Some(vec![0.0; ratio as usize * hd])
                    } else {
                        None
                    },
                    score_state: if ratio > 1 {
                        Some(vec![f32::NEG_INFINITY; ratio as usize * hd])
                    } else {
                        None
                    },
                });
            }
            let idx_src = cfg.index_sources.contains(&l) && ratio > 0;
            let owns_k = idx_src && kv_src.is_some();
            let cand_src = cfg.candidate_source >= 0 && l as i64 == cfg.candidate_source && idx_src;
            let uses_cand =
                cfg.candidate_source >= 0 && (l as i64) > cfg.candidate_source && idx_src;
            let engram_idx = cfg.engram_layers.iter().position(|&e| e == l);
            layers.push(LayerInfo {
                layer: l,
                ratio,
                kv_src,
                idx_src,
                owns_k,
                cand_src,
                uses_cand,
                engram_idx,
                freqs: Freqs::for_layer(&cfg, ratio),
            });
        }
        let rt = Runtime {
            window: vec![vec![0.0; win * hd]; cfg.n_layers as usize],
            sources,
            compress_slot: 0,
            index_k_slot: 0,
            topk_idxs: Vec::new(),
            candidates: Vec::new(),
            engram_ids: Vec::new(),
            token_map: None,
            pos: 0,
            capture_x: None,
        };
        Ok(Self {
            cfg,
            store,
            lru: DenseLru::new(DEFAULT_DENSE_BUDGET),
            layers,
            engram,
            rt,
        })
    }

    /// Reject configs whose tensor geometry would violate the quantization
    /// block constraints (32-divisibility) or the head/group wiring.
    fn validate_dims(cfg: &RefConfig) -> io::Result<()> {
        validate_dims(cfg)
    }

    /// Check every layout tensor exists with a compatible dtype and shape
    /// (checkpoint loads only; synthetic generation is by construction).
    fn validate_layout(&mut self) -> io::Result<()> {
        let cfg = &self.cfg;
        let store: &mut dyn WeightStore = &mut *self.store;
        validate_layout(cfg, store)
    }

    /// Install the tokenizer-derived compressed id map (engram; spec §11).
    /// Identity until set — synthetic tests and fresh checkpoints start
    /// there.
    pub fn set_token_map(&mut self, map: Vec<i64>) {
        self.rt.token_map = Some(map);
    }

    /// Next position `decode` expects.
    pub fn next_position(&self) -> u64 {
        self.rt.pos
    }

    /// Compressed-KV rows a kv-source layer has published so far
    /// (`(pos+1)/ratio`; spec §5). `None` when the layer stores no cache.
    pub fn compress_len(&self, layer: u32) -> Option<usize> {
        let li = &self.layers[layer as usize];
        li.kv_src
            .map(|src| self.rt.sources[src].compress.len() / self.cfg.head_dim as usize)
    }

    /// One sliding-window ring row (gate debugging; spec §3).
    pub fn window_row(&self, layer: u32, slot: usize) -> Vec<f32> {
        let hd = self.cfg.head_dim as usize;
        self.rt.window[layer as usize][slot * hd..(slot + 1) * hd].to_vec()
    }

    /// One published compressed-KV row (gate debugging; spec §5).
    pub fn compress_row(&self, layer: u32, j: usize) -> Option<Vec<f32>> {
        let li = &self.layers[layer as usize];
        let hd = self.cfg.head_dim as usize;
        li.kv_src
            .map(|src| self.rt.sources[src].compress[j * hd..(j + 1) * hd].to_vec())
    }

    /// One owned index-key row (gate debugging; spec §4).
    pub fn index_k_row(&self, layer: u32, j: usize) -> Option<Vec<f32>> {
        let li = &self.layers[layer as usize];
        let ihd = self.cfg.index_head_dim as usize;
        li.kv_src
            .map(|src| self.rt.sources[src].index_k[j * ihd..(j + 1) * ihd].to_vec())
    }

    /// Index-key rows a kv-source layer owns (`(pos+1)/ratio`; spec §4).
    pub fn index_k_len(&self, layer: u32) -> Option<usize> {
        let li = &self.layers[layer as usize];
        li.kv_src
            .map(|src| self.rt.sources[src].index_k.len() / self.cfg.index_head_dim as usize)
    }

    /// Sliding-window ring size (constant per layer).
    pub fn window_size(&self) -> usize {
        self.cfg.window_size as usize
    }

    /// Engram layout (hash primes/multipliers), for tests and tooling.
    pub fn engram_layout(&self) -> &EngramLayout {
        &self.engram
    }

    /// Debug: capture every attention sublayer's input x (gate tooling).
    pub fn debug_capture_x(&mut self, on: bool) {
        self.rt.capture_x = if on { Some(Vec::new()) } else { None };
    }

    /// Debug: take the captured (layer, x) pairs.
    pub fn debug_take_captures(&mut self) -> Vec<(u32, Vec<f32>)> {
        self.rt.capture_x.take().unwrap_or_default()
    }

    /// Speculative drafter (mtp.0-2, DSpark; spec §9). TODO(spec-decode):
    /// not implemented — greedy main-head decoding is the v1 contract.
    /// `main_hiddens` capture (mean over the hc copies at the target-layer
    /// inputs) is likewise skipped; this stub documents the boundary.
    pub fn drafter(&mut self) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "TODO(spec-decode): mtp.* drafter is not implemented; greedy main-head decoding is the v1 contract",
        ))
    }

    /// One decode step (spec §1). Returns the raw fp32 logits over the vocab
    /// for `token` at `position`. Positions must arrive sequentially from 0
    /// (prefill is repeated decode; per-token outputs are identical to the
    /// reference's chunked prefill because attention is causal and the
    /// compressor/window state evolves the same way).
    pub fn decode(&mut self, token: u32, position: u64) -> io::Result<Vec<f32>> {
        if position != self.rt.pos {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "decode expects sequential positions: got {position}, next is {}",
                    self.rt.pos
                ),
            ));
        }
        self.rt.pos += 1;
        crate::dump::set_pos(position);
        if token >= self.cfg.vocab {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "token id out of vocab",
            ));
        }
        let Self {
            cfg,
            store,
            lru,
            layers,
            engram,
            rt,
        } = self;
        let mut ctx = Ctx {
            store: &mut **store,
            lru,
        };

        let hc = cfg.hc_mult as usize;
        let dim = cfg.dim as usize;

        // Synthetic-bisection dump hook (src/bin/dump-synth.rs, spec §12):
        // per-layer streams and sublayer outputs as raw f32, only when
        // FERROSEEK_DUMP names a directory. Zero cost otherwise.
        let dump_dir = std::env::var_os("FERROSEEK_DUMP");
        let dump = |name: String, vals: &[f32]| {
            if let Some(dir) = &dump_dir {
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(std::path::Path::new(dir).join(name), bytes);
            }
        };

        // FERROSEEK_BF16_STREAM=1: transport shadow — round the hc stream
        // and the sublayer outputs to bf16 at every boundary, mimicking the
        // reference/vLLM bf16 storage against this engine's f32 (spec §12
        // known-divergences list). Real-checkpoint-scale control for the
        // gate: quantifies how much of the observed dlogprob is transport.
        let bf16_stream = std::env::var_os("FERROSEEK_BF16_STREAM").is_some();
        let snap_bf16 = |v: &mut [f32]| {
            if bf16_stream {
                for x in v.iter_mut() {
                    *x = ferroseek_core::quant::bf16_to_f32(ferroseek_core::quant::f32_to_bf16(*x));
                }
            }
        };

        // engram hashes for this position, all layers at once (model.py:1255)
        let hashes: Option<Vec<Vec<u64>>> = if engram.layer_ids.is_empty() {
            None
        } else {
            Some(engram::hash_step(cfg, engram, rt, token, position))
        };

        // embed -> 4 hyper-connection copies (model.py:1256, 1261)
        let emb = {
            let view = ctx.get("embed.weight")?;
            bf16_row(&view, token as usize, dim)?
        };
        let mut h = vec![0.0f32; hc * dim];
        for j in 0..hc {
            h[j * dim..(j + 1) * dim].copy_from_slice(&emb);
        }
        // initial one-hot mix [1,0,0,0] (model.py:1162-1166)
        let mut pre_mix = vec![0.0f32; hc];
        pre_mix[0] = 1.0;

        // TODO(spec-decode): main_hiddens capture at dspark_target_layer_ids
        // is a drafter-only concern; greedy main-head decoding is the v1
        // contract and mtp.* stays unimplemented.

        for li in layers.iter() {
            let l = li.layer;
            dump(format!("pos{position}_pre{l}.f32"), &h);
            // engram writes into the stream before the block (model.py:1264-1266)
            if let Some(ei) = li.engram_idx {
                let hash = hashes.as_ref().unwrap()[ei].clone();
                engram::merge(&mut *ctx.store, cfg, engram, ei, l, &mut h, &hash)?;
                snap_bf16(&mut h);
            }
            dump(format!("pos{position}_in{l}.f32"), &h);

            // attention sublayer (spec §2 wiring, model.py:970-996)
            let hc::HcMixes {
                pre: a_pre,
                post: a_post,
                comb: a_comb,
            } = {
                let s = ctx.dense(&format!("layers.{l}.hc_attn_scale"))?.to_vec();
                let b = ctx.dense(&format!("layers.{l}.hc_attn_base"))?.to_vec();
                let w = ctx.dense(&format!("layers.{l}.hc_attn_fn"))?;
                hc::hc_mixes(
                    &h,
                    w,
                    &s,
                    &b,
                    hc,
                    dim,
                    cfg.norm_eps,
                    cfg.hc_sinkhorn_iters as usize,
                    cfg.hc_eps,
                )
            };
            dump(format!("d_pos{position}_hca_pre{l}.f32"), &a_pre);
            dump(format!("d_pos{position}_hca_post{l}.f32"), &a_post);
            dump(format!("d_pos{position}_hca_comb{l}.f32"), &a_comb);
            let x = {
                let x = hc::hc_pre(&h, &pre_mix, hc, dim);
                let nrm = ctx.dense(&format!("layers.{l}.attn_norm.weight"))?;
                let mut x = num::rmsnorm(&x, nrm, cfg.norm_eps);
                snap_bf16(&mut x);
                x
            };
            let mut x = crate::attn::attention(cfg, li, &mut ctx, rt, &x, position)?;
            dump(format!("pos{position}_attn{l}.f32"), &x);
            snap_bf16(&mut x);
            let mut h2 = hc::hc_post(&x, &h, &a_post, &a_comb, hc, dim);
            snap_bf16(&mut h2);
            h = h2;

            // ffn sublayer; its pre feeds the NEXT block's attention
            let hc::HcMixes {
                pre: f_pre,
                post: f_post,
                comb: f_comb,
            } = {
                let s = ctx.dense(&format!("layers.{l}.hc_ffn_scale"))?.to_vec();
                let b = ctx.dense(&format!("layers.{l}.hc_ffn_base"))?.to_vec();
                let w = ctx.dense(&format!("layers.{l}.hc_ffn_fn"))?;
                hc::hc_mixes(
                    &h,
                    w,
                    &s,
                    &b,
                    hc,
                    dim,
                    cfg.norm_eps,
                    cfg.hc_sinkhorn_iters as usize,
                    cfg.hc_eps,
                )
            };
            dump(format!("d_pos{position}_hcf_pre{l}.f32"), &f_pre);
            dump(format!("d_pos{position}_hcf_post{l}.f32"), &f_post);
            dump(format!("d_pos{position}_hcf_comb{l}.f32"), &f_comb);
            let x = {
                let x = hc::hc_pre(&h, &a_pre, hc, dim);
                let nrm = ctx.dense(&format!("layers.{l}.ffn_norm.weight"))?;
                let mut x = num::rmsnorm(&x, nrm, cfg.norm_eps);
                snap_bf16(&mut x);
                x
            };
            let mut x = moe::moe(cfg, &mut ctx, l, &x)?;
            dump(format!("pos{position}_ffn{l}.f32"), &x);
            snap_bf16(&mut x);
            let mut h2 = hc::hc_post(&x, &h, &f_post, &f_comb, hc, dim);
            snap_bf16(&mut h2);
            h = h2;
            pre_mix = f_pre;
        }

        // collapse, final norm, head (model.py:1271-1272, spec §1)
        dump(format!("pos{position}_final.f32"), &h);
        let mut x = hc::hc_pre(&h, &pre_mix, hc, dim);
        snap_bf16(&mut x);
        let x = {
            let nrm = ctx.dense("norm.weight")?;
            num::rmsnorm(&x, nrm, cfg.norm_eps)
        };
        let logits = {
            let head = ctx.dense("head.weight")?;
            num::gemv(head, cfg.vocab as usize, dim, &x)
        };
        dump(format!("logits{position}.f32"), &logits);
        Ok(logits)
    }
}

/// Decode the tokenizer map helper used by tests (identity default).
pub fn identity_token_map(vocab: u32) -> Vec<i64> {
    (0..vocab as i64).collect()
}

/// Reject configs whose tensor geometry would violate the quantization
/// block constraints (32-divisibility) or the head/group wiring.
/// Public: the GPU engine validates the same constraints before allocating
/// device buffers.
pub fn validate_dims(cfg: &RefConfig) -> io::Result<()> {
    let err = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
    let hd = cfg.head_dim as usize;
    let heads = cfg.n_heads as usize;
    if hd % 32 != 0 {
        return Err(err("head_dim must be 32-divisible (fp8 act blocks)".into()));
    }
    if heads * hd % cfg.o_groups as usize != 0 {
        return Err(err(
            "n_heads * head_dim must be divisible by o_groups".into()
        ));
    }
    if heads * hd / cfg.o_groups as usize % 32 != 0 {
        return Err(err("wo_a group row must be 32-divisible".into()));
    }
    if cfg.dim as usize % 32 != 0 || cfg.q_lora as usize % 32 != 0 {
        return Err(err("dim and q_lora_rank must be 32-divisible".into()));
    }
    if cfg.o_groups as usize * cfg.o_lora as usize % 32 != 0 {
        return Err(err("o_groups * o_lora_rank must be 32-divisible".into()));
    }
    if cfg
        .index_sources
        .iter()
        .any(|&l| cfg.compress_ratios.get(l as usize).copied().unwrap_or(0) == 0)
    {
        return Err(err(
            "index_source_layer on a ratio-0 layer (spec forbids)".into()
        ));
    }
    for &s in &cfg.kv_sources {
        let r = cfg.compress_ratios.get(s as usize).copied().unwrap_or(0);
        if r == 0 {
            return Err(err("kv_source_layer with compress_ratio 0".into()));
        }
        if !cfg.index_sources.contains(&s) {
            return Err(err(
                "every kv source must also be an index source (spec §4)".into(),
            ));
        }
    }
    if !cfg.engram_layers.is_empty() && cfg.engram_head_dim as usize % 32 != 0 {
        return Err(err("engram_head_dim must be 32-divisible".into()));
    }
    Ok(())
}

/// Check every layout tensor exists with a compatible dtype and shape.
/// Public: the GPU engine runs the same validation before uploading.
pub fn validate_layout(cfg: &RefConfig, store: &mut dyn WeightStore) -> io::Result<()> {
    let specs = crate::layout::layout(cfg);
    for (name, spec) in &specs {
        let view = store.get(name)?;
        // FP4 weights pack two values per byte: on disk [rows, cols/2];
        // synthetic stores keep the logical shape. Accept both.
        let shape_ok = view.shape == spec.shape
            || (spec.kind == crate::layout::TKind::Fp4
                && view.shape.len() == 2
                && view.shape[0] == spec.shape[0]
                && view.shape[1] * 2 == spec.shape[1]);
        if !shape_ok {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{name}: shape {:?} != expected {:?}",
                    view.shape, spec.shape
                ),
            ));
        }
        let want = match spec.kind {
            crate::layout::TKind::Bf16 => crate::store::DType::Bf16,
            crate::layout::TKind::F32 => crate::store::DType::F32,
            crate::layout::TKind::Fp8 => crate::store::DType::Fp8E4m3,
            crate::layout::TKind::Fp4 => crate::store::DType::Fp4Packed,
            crate::layout::TKind::E8m0 => crate::store::DType::E8m0,
        };
        // The raw HF export keeps wo_a quantized (fp8+ue8m0) — convert.py
        // host-dequantizes it for the reference stack. Accept both; the
        // dense path dequantizes on read (spec §11).
        let want = if want == crate::store::DType::Bf16
            && view.dtype == crate::store::DType::Fp8E4m3
            && name.ends_with(".weight")
        {
            crate::store::DType::Fp8E4m3
        } else {
            want
        };
        // Small fp32-routine tensors (compressor wkv/wgate) are exported
        // bf16 and upcast at load; the dense path already decodes to f32.
        let dtype_ok = view.dtype == want
            || (want == crate::store::DType::F32 && view.dtype == crate::store::DType::Bf16);
        if !dtype_ok {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{name}: dtype {:?} != expected {:?}", view.dtype, want),
            ));
        }
    }
    Ok(())
}
