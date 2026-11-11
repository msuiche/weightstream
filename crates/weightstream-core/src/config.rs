//! DeepSeek V4.1 Flash `config.json` parsing.
//!
//! The 1017 checkpoint nests the text model under `text_config` next to a
//! `vision_config`; older V3-style configs are flat. Both parse into the
//! same struct, and fields the engine does not consume yet land in `extra`
//! so new checkpoint revisions never fail to load.
//!
//! DS4.1F does not name its MLA dims the V3 way. There is no
//! `kv_lora_rank`: the compressed latent is `num_key_value_heads = 1` head
//! of `head_dim` (512) plus the shared `qk_rope_head_dim` (64) — the same
//! 576-wide latent, and `compress_ratios` then says which layers keep KV at
//! that width (1), halved by the CSA compressor (2), or sourced from
//! another layer (0, see `kv_source_layer_ids`).

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,

    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    #[serde(default)]
    pub num_key_value_heads: Option<u32>,

    // attention / MLA
    #[serde(default)]
    pub q_lora_rank: Option<u32>,
    #[serde(default)]
    pub o_lora_rank: Option<u32>,
    #[serde(default)]
    pub head_dim: Option<u32>,
    /// V3-style name; absent on DS4.1F (see struct docs).
    #[serde(default)]
    pub kv_lora_rank: Option<u32>,
    #[serde(default)]
    pub qk_nope_head_dim: Option<u32>,
    pub qk_rope_head_dim: u32,
    #[serde(default)]
    pub v_head_dim: Option<u32>,
    /// Per-layer KV policy: 0 = none/sourced, 1 = full latent, 2 = CSA
    /// ratio-2 compressed. Longer than `num_hidden_layers` when draft
    /// layers are included; index by layer.
    #[serde(default)]
    pub compress_ratios: Vec<u32>,
    #[serde(default)]
    pub kv_source_layer_ids: Vec<u32>,

    // MoE
    #[serde(default)]
    pub n_routed_experts: Option<u32>,
    #[serde(default)]
    pub n_shared_experts: Option<u32>,
    #[serde(default)]
    pub num_experts_per_tok: Option<u32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<u32>,
    #[serde(default)]
    pub first_k_dense_replace: Option<u32>,

    // DSA indexer (absent on pre-DSA checkpoints)
    #[serde(default)]
    pub index_head_dim: Option<u32>,
    #[serde(default)]
    pub index_n_heads: Option<u32>,
    #[serde(default)]
    pub index_topk: Option<u32>,
    /// Layers carrying the indexer query projections (8 on DS4.1F-1017;
    /// the kv_norm/wk half of the indexer lives on kv_source_layer_ids).
    #[serde(default)]
    pub index_source_layer_ids: Vec<u32>,

    // DSpark drafter
    #[serde(default)]
    pub dspark_n_routed_experts: Option<u32>,

    // hyper-connection
    #[serde(default)]
    pub hc_mult: Option<u32>,

    // Engram n-gram tables live on these layers (2 on DS4.1F-1017)
    #[serde(default)]
    pub engram_layer_ids: Vec<u32>,
    // DSpark draft modules (mtp.0..N)
    #[serde(default)]
    pub num_nextn_predict_layers: Option<u32>,

    #[serde(default)]
    pub max_position_embeddings: Option<u32>,

    /// Everything else the checkpoint declares (Engram, DSpark, quant…).
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl ModelConfig {
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_str(&text).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn from_str(text: &str) -> Result<Self, serde_json::Error> {
        let mut value: serde_json::Value = serde_json::from_str(text)?;
        if let Some(obj) = value.as_object_mut() {
            if let Some(inner) = obj.remove("text_config") {
                // Composite config: text fields win; top-level model_type,
                // architectures and quantization_config survive via `extra`
                // unless the text config redeclares them.
                let mut merged = inner;
                if let Some(m) = merged.as_object_mut() {
                    for key in ["model_type", "architectures", "quantization_config"] {
                        if !m.contains_key(key) {
                            if let Some(v) = obj.remove(key) {
                                m.insert(key.to_string(), v);
                            }
                        }
                    }
                }
                value = merged;
            }
        }
        serde_json::from_value(value)
    }

    pub fn num_nextn_predict_layers(&self) -> u32 {
        self.num_nextn_predict_layers.unwrap_or(0)
    }

    pub fn is_dsa(&self) -> bool {
        self.index_topk.is_some() && self.index_head_dim.is_some()
    }

    /// Compressed KV latent width per token (elements).
    /// V3-style: kv_lora_rank + qk_rope_head_dim. DS4.1F: the wkv output is
    /// head_dim wide and the RoPE tail lives *inside* it (spec §3), so the
    /// latent is head_dim alone. `None` when the config names neither layout.
    pub fn kv_width(&self) -> Option<u32> {
        match (self.kv_lora_rank, self.head_dim, self.num_key_value_heads) {
            (Some(rank), _, _) => Some(rank + self.qk_rope_head_dim),
            (None, Some(head_dim), Some(1)) => Some(head_dim),
            _ => None,
        }
    }
}
