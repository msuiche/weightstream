//! KV-cache sizing for DS4.1F's shared-cache layout (model-spec §0/§5).
//!
//! The architecture does NOT store one cache per layer. Only the four
//! kv_source layers publish compressed caches, and every consumer re-reads
//! the publisher's cache:
//!   - compressed KV: L2/L8/L14 at ratio 2 (one latent per 2 tokens),
//!     L20 at ratio 1 — latent is head_dim (512) wide, bf16 storage with
//!     fp4-snapped values (block 16, E4M3 scales)
//!   - indexer K: same four layers, 128 wide, bf16, fp4-snapped (block 32,
//!     E8M0 scales)
//!   - sliding windows: every attention layer (40 + 3 MTP) keeps its own
//!     128-slot bf16 ring — constant per sequence, not per token

use crate::config::ModelConfig;

pub const BF16_BYTES: u64 = 2;
pub const FP8_BYTES: u64 = 1;
/// Per-block FP8 scales plus alignment slack ≈ 3.5% (block 32).
pub const FP8_SCALE_OVERHEAD: f64 = 0.035;

pub struct KvLayout {
    /// Compressed-cache elements stored per token (all publishers).
    pub compressed_per_token: u64,
    /// Indexer K elements stored per token.
    pub index_per_token: u64,
    /// Window ring elements total (constant per sequence).
    pub window_total: u64,
    pub latent_width: u32,
}

pub fn layout(cfg: &ModelConfig, window_size: u32, n_attn_layers: u32) -> Option<KvLayout> {
    let width = cfg.kv_width()?;
    let mut compressed = 0u64;
    let mut index = 0u64;
    for (i, &layer) in cfg.kv_source_layer_ids.iter().enumerate() {
        let ratio = cfg
            .compress_ratios
            .get(layer as usize)
            .copied()
            .unwrap_or(1)
            .max(1) as u64;
        compressed += width as u64 / ratio;
        // indexer K width 128 on the same layers, same ratio
        index += 128 / ratio;
        let _ = i;
    }
    Some(KvLayout {
        compressed_per_token: compressed,
        index_per_token: index,
        window_total: n_attn_layers as u64 * window_size as u64 * width as u64,
        latent_width: width,
    })
}

/// Bytes per token of compressed + indexer cache at `dtype_bytes` per
/// element (bf16 = 2 on the reference layout).
pub fn kv_bytes_per_token(cfg: &ModelConfig, dtype_bytes: u64) -> Option<u64> {
    let l = layout(cfg, 128, 43)?;
    let raw = (l.compressed_per_token + l.index_per_token) * dtype_bytes;
    Some(if dtype_bytes == FP8_BYTES {
        raw + (raw as f64 * FP8_SCALE_OVERHEAD) as u64
    } else {
        raw
    })
}

/// Constant per-sequence window ring bytes.
pub fn window_bytes(cfg: &ModelConfig, dtype_bytes: u64) -> Option<u64> {
    Some(layout(cfg, 128, 43)?.window_total * dtype_bytes)
}

/// Tokens that fit in `budget_bytes`, after `reserve_bytes` for activation
/// workspace and the memory guard. Window rings are charged once.
pub fn pool_tokens(cfg: &ModelConfig, dtype_bytes: u64, budget_bytes: u64, reserve_bytes: u64) -> Option<u64> {
    let per = kv_bytes_per_token(cfg, dtype_bytes)?;
    let windows = window_bytes(cfg, dtype_bytes)?;
    Some(budget_bytes.saturating_sub(reserve_bytes + windows) / per)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The real DS4.1F-1017 shape: composite config, 40 layers, 512-wide
    /// latent with the RoPE tail inside head_dim, CSA ratio-2 publishers on
    /// layers 2/8/14, ratio-1 on 20.
    pub(crate) fn fixture_json() -> String {
        let mut ratios = vec![0u32; 2];
        ratios.extend([2u32; 18]);
        ratios.extend([1u32; 20]);
        ratios.extend([0u32; 3]); // nextn draft layers
        serde_json::json!({
            "model_type": "deepseek_v41",
            "architectures": ["DeepseekV41ForCausalLM"],
            "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"},
            "text_config": {
                "vocab_size": 129280,
                "hidden_size": 5120,
                "num_hidden_layers": 40,
                "num_attention_heads": 64,
                "num_key_value_heads": 1,
                "q_lora_rank": 1280,
                "o_lora_rank": 1024,
                "head_dim": 512,
                "qk_rope_head_dim": 64,
                "compress_ratios": ratios,
                "kv_source_layer_ids": [2, 8, 14, 20],
                "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
                "engram_layer_ids": [1, 14],
                "num_nextn_predict_layers": 3,
                "dspark_n_routed_experts": 128,
                "n_routed_experts": 384,
                "n_shared_experts": 1,
                "num_experts_per_tok": 6,
                "index_head_dim": 128,
                "index_n_heads": 32,
                "index_topk": 512,
                "hc_mult": 4,
                "max_position_embeddings": 1048576
            },
            "vision_config": {"hidden_size": 1152}
        })
        .to_string()
    }

    pub(crate) fn fixture() -> ModelConfig {
        ModelConfig::from_str(&fixture_json()).unwrap()
    }

    #[test]
    fn unwraps_composite_config() {
        let cfg = fixture();
        assert_eq!(cfg.model_type, "deepseek_v41");
        assert_eq!(cfg.num_hidden_layers, 40);
        assert!(cfg.is_dsa());
        assert_eq!(cfg.kv_width(), Some(512));
        assert!(cfg.extra.contains_key("vision_config") || cfg.extra.contains_key("quantization_config"));
    }

    #[test]
    fn only_publishers_store_compressed_kv() {
        let cfg = fixture();
        let l = layout(&cfg, 128, 43).unwrap();
        // 3 × (512/2) + 512 = 1280, indexer 3 × 64 + 128 = 320
        assert_eq!(l.compressed_per_token, 1280);
        assert_eq!(l.index_per_token, 320);
        assert_eq!(l.window_total, 43 * 128 * 512);
    }

    #[test]
    fn one_m_context_per_sequence_is_cheap() {
        let cfg = fixture();
        let per = kv_bytes_per_token(&cfg, BF16_BYTES).unwrap();
        assert_eq!(per, (1280 + 320) * 2);
        let gib = 1_048_576u64 as f64 * per as f64 / (1u64 << 30) as f64;
        // ~3.1 GiB per 1M-token sequence (bf16 snapped storage)
        assert!(gib > 2.8 && gib < 3.4);
    }
}
