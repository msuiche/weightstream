//! KV-cache sizing.
//!
//! One compressed latent per token per KV-bearing layer — not per-head K
//! and V — and DS4.1F's `compress_ratios` shrinks that further: ratio-2
//! layers store half the latent (CSA compressor), ratio-0 layers store none
//! (they re-source KV from `kv_source_layer_ids`). On GB10 the latent is
//! kept in FP8 with per-block scales (`fp8_ds_mla`); the maths here is
//! dtype-agnostic.

use crate::config::ModelConfig;

pub const BF16_BYTES: u64 = 2;
pub const FP8_BYTES: u64 = 1;
/// Per-block FP8 scales plus alignment slack on the 576-wide latent ≈ 3.5%.
pub const FP8_SCALE_OVERHEAD: f64 = 0.035;

/// Latent elements stored per token, summed over layers.
/// Returns `None` when the config names no MLA layout at all.
pub fn kv_elements_per_token(cfg: &ModelConfig) -> Option<u64> {
    let width = cfg.kv_width()? as u64;
    if cfg.compress_ratios.is_empty() {
        return Some(cfg.num_hidden_layers as u64 * width);
    }
    let mut total = 0u64;
    for layer in 0..cfg.num_hidden_layers as usize {
        match cfg.compress_ratios.get(layer).copied().unwrap_or(1) {
            0 => {} // re-sources KV from a source layer; stores nothing
            ratio => total += width / ratio as u64,
        }
    }
    Some(total)
}

pub fn kv_bytes_per_token(cfg: &ModelConfig, dtype_bytes: u64) -> Option<u64> {
    let raw = kv_elements_per_token(cfg)? * dtype_bytes;
    Some(if dtype_bytes == FP8_BYTES {
        raw + (raw as f64 * FP8_SCALE_OVERHEAD) as u64
    } else {
        raw
    })
}

/// Tokens that fit in `budget_bytes`, after `reserve_bytes` for activation
/// workspace and the memory guard.
pub fn pool_tokens(cfg: &ModelConfig, dtype_bytes: u64, budget_bytes: u64, reserve_bytes: u64) -> Option<u64> {
    let per = kv_bytes_per_token(cfg, dtype_bytes)?;
    Some(budget_bytes.saturating_sub(reserve_bytes) / per)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The real DS4.1F-1017 shape: composite config, 40 layers, 576-wide
    /// latent via head_dim 512 + rope 64, CSA ratio-2 on layers 2–19,
    /// full rate on 20–39, none on 0–1.
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
        assert_eq!(cfg.kv_width(), Some(576));
        assert!(cfg.extra.contains_key("vision_config") || cfg.extra.contains_key("quantization_config"));
    }

    #[test]
    fn compression_ratios_shrink_the_pool() {
        let cfg = fixture();
        // 18 layers × 288 + 20 layers × 576 = 16704, not 40 × 576 = 23040.
        assert_eq!(kv_elements_per_token(&cfg), Some(16_704));
        let fp8 = kv_bytes_per_token(&cfg, FP8_BYTES).unwrap();
        assert!(fp8 > 16_704 && fp8 < 17_500);
    }

    #[test]
    fn one_m_context_per_sequence_is_cheap() {
        let cfg = fixture();
        let fp8 = kv_bytes_per_token(&cfg, FP8_BYTES).unwrap();
        let gib = 1_048_576u64 as f64 * fp8 as f64 / (1u64 << 30) as f64;
        // ~16.9 GiB per 1M-token sequence per rank at TP1.
        assert!(gib > 15.0 && gib < 18.0);
    }
}
