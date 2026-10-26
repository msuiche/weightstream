//! Synthetic bisection dumper (spec §12 gate tooling).
//!
//! Builds a synthetic mini-model sized so BOTH engines can run it — this
//! crate's CPU decoder and the reference Python stack
//! (`/tmp/ds41f-ref/inference/model.py`, TP1) — dumps its weights as a real
//! safetensors checkpoint with the tensor names/dtypes/shapes `model.py`
//! loads, emits the matching `ModelArgs` JSON, then decodes a fixed token
//! sequence with `FERROSEEK_DUMP` pointed at the output directory so the
//! per-layer stream dumps land next to the weights for comparison.
//!
//! ```text
//! cargo run -p ferroseek-ref --bin dump-synth -- /tmp/ferroseek-bisect
//! ```

#![recursion_limit = "512"]

use ferroseek_core::ModelConfig;
use ferroseek_ref::engram::EngramLayout;
use ferroseek_ref::store::{DType, SyntheticWeights, WeightStore};
use ferroseek_ref::{RefConfig, RefModel};
use std::path::Path;

/// Dims chosen to satisfy both engines' kernel constraints: every fp8/fp4
/// GEMM K is 32-divisible, every act_quant N is 32-divisible, head_dim is
/// 16-divisible (compressed-KV fp4 block), the fp8 GEMM N tiles fit
/// tilelang's 128-wide blocks. Layer wiring covers every mechanism: L0
/// ratio-0 (window only) with an engram, L1 ratio-2 kv+index source with an
/// engram, L2 ratio-2 consumer, L3 ratio-1 kv+index source and candidate
/// source, L4 ratio-1 indexer consumer of L3's candidates.
const TOKENS: [u32; 8] = [3, 5, 7, 11, 13, 17, 19, 23];

fn bisect_config_json(engram_rows: &[u64]) -> String {
    serde_json::json!({
        "vocab_size": 512,
        "hidden_size": 256,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "q_lora_rank": 128,
        "o_lora_rank": 128,
        "o_groups": 2,
        "head_dim": 128,
        "qk_rope_head_dim": 32,
        "compress_ratios": [0, 2, 2, 1, 1],
        "kv_source_layer_ids": [1, 3],
        "index_source_layer_ids": [1, 3, 4],
        "candidate_source_layer": 3,
        "candidate_topk_blocks": 2,
        "candidate_block_size": 2,
        "index_n_heads": 4,
        "index_head_dim": 64,
        "index_topk": 8,
        "window_size": 16,
        "norm_eps": 1e-20,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "score_func": "sqrtsoftplus",
        "gate_temp": 1.0,
        "norm_topk_prob": true,
        "route_scale": 1.5,
        "swiglu_limit": 10.0,
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "original_seq_len": 65536,
        "rope_factor": 16.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "engram_layer_ids": [0, 1],
        "engram_max_ngram_size": 4,
        "engram_n_heads": 2,
        "engram_head_dim": 64,
        "engram_pad_id": 2,
        "engram_compressed_vocab_size": 512,
        "engram_vocab_size": 100,
        "engram_num_embeddings": engram_rows
    })
    .to_string()
}

/// `ModelArgs` kwargs for the reference `model.py` (field names are the
/// config JSON keys, model.py:44-136). Scale-independent values and the
/// engram/compress wiring match the Rust config above.
fn model_args_json(engram_rows: &[u64]) -> String {
    serde_json::json!({
        "max_batch_size": 1,
        "max_seq_len": 512,
        "temperature": 1.0,
        "dtype": "fp8",
        "expert_dtype": "fp4",
        "vocab_size": 512,
        "dim": 256,
        "moe_inter_dim": 256,
        "n_layers": 5,
        "n_mtp_layers": 1,
        "n_heads": 4,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
        "score_func": "sqrtsoftplus",
        "gate_temp": 1.0,
        "norm_topk_prob": true,
        "route_scale": 1.5,
        "swiglu_limit": 10.0,
        "q_lora_rank": 128,
        "head_dim": 128,
        "rope_head_dim": 32,
        "norm_eps": 1e-20,
        "o_groups": 2,
        "o_lora_rank": 128,
        "window_size": 16,
        "compress_ratios": [0, 2, 2, 1, 1],
        "kv_source_layers": [1, 3],
        "index_source_layers": [1, 3, 4],
        "compress_rope_theta": 160000.0,
        "original_seq_len": 65536,
        "rope_theta": 10000.0,
        "rope_factor": 16.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "index_n_heads": 4,
        "index_head_dim": 64,
        "index_topk": 8,
        "candidate_source_layer": 3,
        "candidate_topk_blocks": 2,
        "candidate_block_size": 2,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "engram_layer_ids": [0, 1],
        "engram_num_embeddings": engram_rows,
        "engram_max_ngram_size": 4,
        "engram_vocab_size": 100,
        "engram_n_heads": 2,
        "engram_head_dim": 64,
        "engram_pad_id": 2,
        "engram_compressed_vocab_size": 512,
        "vision_n_layers": 0,
        "dspark_block_size": 0,
        "dspark_target_layer_ids": []
    })
    .to_string()
}

/// safetensors dtype string + on-disk shape for a stored tensor (convert.py
/// conventions: fp4 packs two values per byte along K, e8m0 scales are bare).
fn st_entry(view: &ferroseek_ref::TensorView<'_>) -> (String, Vec<usize>) {
    match view.dtype {
        DType::Bf16 => ("BF16".into(), view.shape.clone()),
        DType::F32 => ("F32".into(), view.shape.clone()),
        DType::Fp8E4m3 => ("F8_E4M3".into(), view.shape.clone()),
        DType::E8m0 => ("F8_E8M0".into(), view.shape.clone()),
        // safetensors F4 elements are 4 bits: write the LOGICAL [rows, cols]
        // shape over the packed bytes; torch loads it as float4_e2m1fn_x2.
        DType::Fp4Packed => ("F4".into(), view.shape.clone()),
        DType::U8 => ("U8".into(), view.shape.clone()),
    }
}

fn write_safetensors(
    out: &Path,
    weights: &mut SyntheticWeights,
) -> std::io::Result<()> {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let names: Vec<String> = weights.names().cloned().collect();
    for name in &names {
        let view = weights.get(name)?;
        let (dt, shape) = st_entry(&view);
        let start = payload.len() as u64;
        payload.extend_from_slice(view.bytes);
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dt,
                "shape": shape,
                "data_offsets": [start, payload.len() as u64]
            }),
        );
    }
    let hdr = serde_json::to_vec(&serde_json::Value::Object(header))?;
    let mut shard = (hdr.len() as u64).to_le_bytes().to_vec();
    shard.extend_from_slice(&hdr);
    shard.extend_from_slice(&payload);
    std::fs::write(out.join("synth.safetensors"), shard)
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/ferroseek-bisect".into());
    let out = Path::new(&out);
    std::fs::create_dir_all(out).expect("create output dir");

    // engram table rows: offsets+primes the layout needs, per layer
    let probe = ModelConfig::from_str(&bisect_config_json(&[1, 1])).expect("config parses");
    let rcfg = RefConfig::from_model_config(&probe).expect("ref config");
    let rows = EngramLayout::build(&rcfg)
        .expect("engram layout")
        .required_rows();

    let cfg = ModelConfig::from_str(&bisect_config_json(&rows)).expect("config parses");
    let seed = 1234;
    let mut weights = SyntheticWeights::generate(&cfg, seed).expect("synthetic weights");

    write_safetensors(out, &mut weights).expect("write safetensors");
    std::fs::write(out.join("model_args.json"), model_args_json(&rows)).expect("write args");
    std::fs::write(
        out.join("tokens.json"),
        serde_json::to_vec(&TOKENS).expect("tokens json"),
    )
    .expect("write tokens");

    let mut model = RefModel::from_synthetic(cfg, weights).expect("model builds");
    std::env::set_var("FERROSEEK_DUMP", out);
    for (p, &t) in TOKENS.iter().enumerate() {
        let logits = model.decode(t, p as u64).expect("decode");
        assert!(logits.iter().all(|v| v.is_finite()), "pos {p}");
    }
    println!("dumped {} layers x {} positions to {}", 5, TOKENS.len(), out.display());
}
