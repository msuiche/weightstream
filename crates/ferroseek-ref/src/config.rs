//! Decode-time configuration, flattened from `ModelConfig` + `config.json`
//! extras (spec §0). Named `ModelConfig` fields win; everything the spec
//! table lists but HF-style configs do not name lives in `extra` and is read
//! with the release (1017) values as defaults. Both key spellings (HF-style
//! like `hidden_size` and the reference's ModelArgs names like `dim`) are
//! accepted, matching how the core config parser treats the composite
//! `text_config`.

use ferroseek_core::ModelConfig;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io;

/// Decode-time configuration for the CPU reference engine (spec §0).
#[derive(Debug, Clone)]
pub struct RefConfig {
    pub vocab: u32,
    pub dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    /// Total per-head Q/V width; the rope tail is inside it (nope + rope).
    pub head_dim: u32,
    pub rope_dim: u32,
    pub nope_dim: u32,
    pub q_lora: u32,
    pub o_lora: u32,
    pub o_groups: u32,
    pub window_size: u32,
    pub norm_eps: f32,

    /// Per-layer KV policy: 0 = window only, 1 = full latent, 2 = CSA 2:1.
    pub compress_ratios: Vec<u32>,
    pub kv_sources: Vec<u32>,
    pub index_sources: Vec<u32>,
    pub candidate_source: i64,
    pub candidate_topk_blocks: u32,
    pub candidate_block_size: u32,
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,

    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
    pub hc_eps: f32,

    pub rope_theta: f64,
    pub compress_rope_theta: f64,
    pub original_seq_len: u32,
    pub rope_factor: f64,
    pub beta_fast: u32,
    pub beta_slow: u32,

    pub n_routed: u32,
    pub n_shared: u32,
    pub n_activated: u32,
    pub inter_dim: u32,
    pub score_func: String,
    pub gate_temp: f32,
    pub norm_topk_prob: bool,
    pub route_scale: f32,
    pub swiglu_limit: f32,

    pub engram_layers: Vec<u32>,
    pub engram_max_ngram: u32,
    pub engram_n_heads: u32,
    pub engram_head_dim: u32,
    pub engram_pad_id: u32,
    pub engram_compressed_vocab: u32,
    /// Bucket size each (n-gram size, head) starts searching primes from
    /// (`engram_vocab_size`; 16e6 on 1017, spec §8).
    pub engram_prime_start: u64,
    pub engram_num_embeddings: Vec<u64>,
}

impl RefConfig {
    /// Flatten a parsed checkpoint config. Accepts HF-style and ModelArgs-style
    /// key names; defaults are the 1017 release values (spec §0).
    pub fn from_model_config(cfg: &ModelConfig) -> io::Result<Self> {
        let extra = &cfg.extra;
        let err = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_string());

        let dim = cfg.hidden_size;
        let n_layers = cfg.num_hidden_layers;
        let n_heads = cfg.num_attention_heads;
        let rope_dim = cfg.qk_rope_head_dim;
        let head_dim = cfg
            .head_dim
            .or_else(|| opt_u32(extra, &["head_dim"]))
            .or_else(|| cfg.qk_nope_head_dim.map(|nope| nope + rope_dim))
            .ok_or_else(|| err("missing head_dim"))?;
        if head_dim < rope_dim {
            return Err(err("head_dim < qk_rope_head_dim"));
        }
        let q_lora = cfg
            .q_lora_rank
            .or_else(|| opt_u32(extra, &["q_lora_rank"]))
            .ok_or_else(|| err("missing q_lora_rank"))?;
        let o_lora = cfg
            .o_lora_rank
            .or_else(|| opt_u32(extra, &["o_lora_rank"]))
            .ok_or_else(|| err("missing o_lora_rank"))?;

        let compress_ratios = if cfg.compress_ratios.len() >= n_layers as usize {
            cfg.compress_ratios[..n_layers as usize].to_vec()
        } else {
            return Err(err("compress_ratios shorter than num_hidden_layers"));
        };
        let kv_sources = if cfg.kv_source_layer_ids.is_empty() {
            opt_vec_u32(extra, &["kv_source_layers"]).unwrap_or_default()
        } else {
            cfg.kv_source_layer_ids.clone()
        };
        let index_sources = if cfg.index_source_layer_ids.is_empty() {
            opt_vec_u32(extra, &["index_source_layers"]).unwrap_or_default()
        } else {
            cfg.index_source_layer_ids.clone()
        };

        let engram_layers = if cfg.engram_layer_ids.is_empty() {
            opt_vec_u32(extra, &["engram_layer_ids"]).unwrap_or_default()
        } else {
            cfg.engram_layer_ids.clone()
        };
        let engram_compressed_vocab =
            opt_u32(extra, &["engram_compressed_vocab_size"]).unwrap_or(0);
        let engram_num_embeddings =
            opt_vec_u64(extra, &["engram_num_embeddings"]).unwrap_or_default();
        if !engram_layers.is_empty() {
            if engram_compressed_vocab == 0 {
                return Err(err("engram layers require engram_compressed_vocab_size"));
            }
            if engram_num_embeddings.len() < engram_layers.len() {
                return Err(err(
                    "engram_num_embeddings must list one row count per engram layer",
                ));
            }
        }

        let n_routed = opt_u32(extra, &["n_routed_experts", "num_experts_per_tok__routed"])
            .or(cfg.n_routed_experts)
            .ok_or_else(|| err("missing n_routed_experts"))?;
        let n_activated = opt_u32(extra, &["num_experts_per_tok", "n_activated_experts"])
            .or(cfg.num_experts_per_tok)
            .ok_or_else(|| err("missing num_experts_per_tok"))?;
        let inter_dim = opt_u32(extra, &["moe_intermediate_size", "moe_inter_dim"])
            .or(cfg.moe_intermediate_size)
            .ok_or_else(|| err("missing moe_intermediate_size"))?;
        let n_shared = opt_u32(extra, &["n_shared_experts"])
            .or(cfg.n_shared_experts)
            .unwrap_or(1);
        let index_n_heads = opt_u32(extra, &["index_n_heads"])
            .or(cfg.index_n_heads)
            .unwrap_or(0);
        let index_head_dim = opt_u32(extra, &["index_head_dim"])
            .or(cfg.index_head_dim)
            .unwrap_or(0);
        let index_topk = opt_u32(extra, &["index_topk"])
            .or(cfg.index_topk)
            .unwrap_or(0);
        let hc_mult = opt_u32(extra, &["hc_mult"]).or(cfg.hc_mult).unwrap_or(4);

        Ok(RefConfig {
            vocab: cfg.vocab_size,
            dim,
            n_layers,
            n_heads,
            head_dim,
            rope_dim,
            nope_dim: head_dim - rope_dim,
            q_lora,
            o_lora,
            o_groups: opt_u32(extra, &["o_groups"]).unwrap_or(8),
            // the release export names this `sliding_window` (HF spelling)
            window_size: opt_u32(extra, &["window_size", "sliding_window"]).unwrap_or(128),
            norm_eps: opt_f32(extra, &["norm_eps", "rms_norm_eps"]).unwrap_or(1e-20),
            compress_ratios,
            kv_sources,
            index_sources,
            // the release export names this `candidate_source_layer_id`
            candidate_source: opt_i64(extra, &["candidate_source_layer", "candidate_source_layer_id"])
                .unwrap_or(-1),
            candidate_topk_blocks: opt_u32(extra, &["candidate_topk_blocks"]).unwrap_or(0),
            candidate_block_size: opt_u32(extra, &["candidate_block_size"]).unwrap_or(0),
            index_n_heads,
            index_head_dim,
            index_topk,
            hc_mult,
            hc_sinkhorn_iters: opt_u32(extra, &["hc_sinkhorn_iters"]).unwrap_or(20),
            hc_eps: opt_f32(extra, &["hc_eps"]).unwrap_or(1e-6),
            rope_theta: opt_f64(extra, &["rope_theta"]).unwrap_or(10000.0),
            compress_rope_theta: opt_f64(extra, &["compress_rope_theta"]).unwrap_or(160000.0),
            // YaRN: flat ModelArgs keys first; the release export nests them
            // under rope_scaling (factor/beta_fast/beta_slow/
            // original_max_position_embeddings)
            original_seq_len: opt_u32(extra, &["original_seq_len"])
                .or_else(|| rope_scaling_u32(extra, "original_max_position_embeddings"))
                .unwrap_or(65536),
            rope_factor: opt_f64(extra, &["rope_factor"])
                .or_else(|| rope_scaling_f64(extra, "factor"))
                .unwrap_or(16.0),
            beta_fast: opt_u32(extra, &["beta_fast"])
                .or_else(|| rope_scaling_u32(extra, "beta_fast"))
                .unwrap_or(32),
            beta_slow: opt_u32(extra, &["beta_slow"])
                .or_else(|| rope_scaling_u32(extra, "beta_slow"))
                .unwrap_or(1),
            n_routed,
            n_shared,
            n_activated,
            inter_dim,
            score_func: opt_string(extra, &["score_func", "scoring_func"])
                .unwrap_or_else(|| "sqrtsoftplus".into()),
            gate_temp: opt_f32(extra, &["gate_temp"]).unwrap_or(1.0),
            norm_topk_prob: opt_bool(extra, &["norm_topk_prob"]).unwrap_or(true),
            route_scale: opt_f32(extra, &["route_scale", "routed_scaling_factor"]).unwrap_or(1.5),
            swiglu_limit: opt_f32(extra, &["swiglu_limit"]).unwrap_or(10.0),
            engram_layers,
            engram_max_ngram: opt_u32(extra, &["engram_max_ngram_size"]).unwrap_or(4),
            engram_n_heads: opt_u32(extra, &["engram_n_heads"]).unwrap_or(8),
            engram_head_dim: opt_u32(extra, &["engram_head_dim"]).unwrap_or(256),
            // the release export names this `engram_pad_token_id`
            engram_pad_id: opt_u32(extra, &["engram_pad_id", "engram_pad_token_id"]).unwrap_or(2),
            engram_compressed_vocab,
            engram_prime_start: opt_u64(extra, &["engram_vocab_size"]).unwrap_or(16_000_000),
            engram_num_embeddings,
        })
    }

    /// mix_hc = (2 + hc) * hc rows of the hyper-connection mix (spec §2).
    pub fn mix_hc(&self) -> usize {
        ((2 + self.hc_mult) * self.hc_mult) as usize
    }
    /// hc_dim = hc * dim, width of the flattened stream (spec §2).
    pub fn hc_dim(&self) -> usize {
        (self.hc_mult * self.dim) as usize
    }
}

fn get<'a>(extra: &'a BTreeMap<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|k| extra.get(*k))
}

/// Read one field out of the nested HF `rope_scaling` object
/// ({rope_type, factor, beta_fast, beta_slow, original_max_position_embeddings}).
fn rope_scaling_value<'a>(
    extra: &'a BTreeMap<String, Value>,
    key: &str,
) -> Option<&'a Value> {
    extra.get("rope_scaling").and_then(|v| v.get(key))
}

fn rope_scaling_u32(extra: &BTreeMap<String, Value>, key: &str) -> Option<u32> {
    rope_scaling_value(extra, key)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
}

fn rope_scaling_f64(extra: &BTreeMap<String, Value>, key: &str) -> Option<f64> {
    rope_scaling_value(extra, key).and_then(Value::as_f64)
}

fn opt_u32(extra: &BTreeMap<String, Value>, keys: &[&str]) -> Option<u32> {
    get(extra, keys)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
}

fn opt_u64(extra: &BTreeMap<String, Value>, keys: &[&str]) -> Option<u64> {
    get(extra, keys).and_then(Value::as_u64)
}

fn opt_i64(extra: &BTreeMap<String, Value>, keys: &[&str]) -> Option<i64> {
    get(extra, keys).and_then(Value::as_i64)
}

fn opt_f32(extra: &BTreeMap<String, Value>, keys: &[&str]) -> Option<f32> {
    get(extra, keys).and_then(Value::as_f64).map(|v| v as f32)
}

fn opt_f64(extra: &BTreeMap<String, Value>, keys: &[&str]) -> Option<f64> {
    get(extra, keys).and_then(Value::as_f64)
}

fn opt_bool(extra: &BTreeMap<String, Value>, keys: &[&str]) -> Option<bool> {
    get(extra, keys).and_then(Value::as_bool)
}

fn opt_string(extra: &BTreeMap<String, Value>, keys: &[&str]) -> Option<String> {
    get(extra, keys).and_then(Value::as_str).map(str::to_string)
}

fn opt_vec_u32(extra: &BTreeMap<String, Value>, keys: &[&str]) -> Option<Vec<u32>> {
    get(extra, keys).and_then(Value::as_array).map(|a| {
        a.iter()
            .filter_map(|v| v.as_u64().and_then(|x| u32::try_from(x).ok()))
            .collect()
    })
}

fn opt_vec_u64(extra: &BTreeMap<String, Value>, keys: &[&str]) -> Option<Vec<u64>> {
    get(extra, keys)
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0)).collect())
}
