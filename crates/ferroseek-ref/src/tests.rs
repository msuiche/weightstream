//! Gate tests: a synthetic mini model exercising every mechanism (window
//! wrap, CSA ratio 2 open pairs, ratio-1 re-sourcing, two-level top-k with
//! candidates, engram, hyper-connections, fp4/fp8 paths) through the same
//! `WeightStore` dequantization the checkpoint path uses.

use crate::config::RefConfig;
use crate::model::RefModel;
use crate::sample::argmax;
use crate::store::SyntheticWeights;
use ferroseek_core::ModelConfig;

/// The mini config JSON, shared with the moe unit test.
pub(crate) fn mini_config_json_public() -> String {
    mini_config_json(mini_engram_rows())
}

/// Mini config: 4 layers, hidden 64, 4 heads, head_dim 32 (rope 8), 8
/// experts top-2, window 8 (wraps inside an 8-step test), compress_ratios
/// [0, 2, 1, 1] — ratio 0/1/2 all covered; kv sources {1, 2}; index sources
/// {1, 2, 3} so layer 3 exercises the consumer path; candidate source 2
/// (block size 2, topk 2); engram on layer 1; hc_mult 4.
fn mini_config_json(engram_rows: u64) -> String {
    serde_json::json!({
        "vocab_size": 32,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "q_lora_rank": 32,
        "o_lora_rank": 16,
        "o_groups": 2,
        "head_dim": 32,
        "qk_rope_head_dim": 8,
        "compress_ratios": [0, 2, 1, 1],
        "kv_source_layer_ids": [1, 2],
        "index_source_layer_ids": [1, 2, 3],
        "candidate_source_layer": 2,
        "candidate_topk_blocks": 2,
        "candidate_block_size": 2,
        "index_n_heads": 2,
        "index_head_dim": 32,
        "index_topk": 4,
        "window_size": 8,
        "norm_eps": 1e-20,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "score_func": "sqrtsoftplus",
        "gate_temp": 1.0,
        "norm_topk_prob": true,
        "route_scale": 1.5,
        "swiglu_limit": 10.0,
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "original_seq_len": 0,
        "rope_factor": 16.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "engram_layer_ids": [1],
        "engram_max_ngram_size": 4,
        "engram_n_heads": 2,
        "engram_head_dim": 32,
        "engram_pad_id": 2,
        "engram_compressed_vocab_size": 32,
        "engram_vocab_size": 100,
        "engram_num_embeddings": [engram_rows]
    })
    .to_string()
}

fn mini_config(engram_rows: u64) -> ModelConfig {
    ModelConfig::from_str(&mini_config_json(engram_rows)).expect("config parses")
}

/// Engram table rows the layout needs for the mini config (offsets+primes).
fn mini_engram_rows() -> u64 {
    let cfg = RefConfig::from_model_config(&mini_config(1)).unwrap();
    let layout = crate::engram::EngramLayout::build(&cfg).unwrap();
    layout.required_rows()[0]
}

fn mini_model(seed: u64) -> RefModel<'static> {
    let cfg = mini_config(mini_engram_rows());
    let weights = SyntheticWeights::generate(&cfg, seed).expect("synthetic weights");
    RefModel::from_synthetic(cfg, weights).expect("model builds")
}

const TOKENS: [u32; 8] = [3, 5, 7, 11, 13, 17, 19, 23];

fn decode_steps(model: &mut RefModel<'_>, tokens: &[u32]) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for (p, &t) in tokens.iter().enumerate() {
        let logits = model.decode(t, p as u64).expect("decode");
        assert_eq!(logits.len(), 32);
        out.push(logits);
    }
    out
}

#[test]
fn synthetic_mini_decode_is_finite_and_deterministic() {
    // same synthetic seed twice -> bit-identical logits at every step
    let mut a = mini_model(42);
    let mut b = mini_model(42);
    let la = decode_steps(&mut a, &TOKENS);
    let lb = decode_steps(&mut b, &TOKENS);
    for (step, (x, y)) in la.iter().zip(lb.iter()).enumerate() {
        assert_eq!(x.len(), y.len());
        assert_eq!(
            x, y,
            "step {step}: logits must be bit-identical for identical runs"
        );
        assert!(
            x.iter().all(|v| v.is_finite()),
            "step {step}: logits must be finite"
        );
    }

    // argmax changes plausibly across steps (random weights -> different
    // logits; a constant argmax would indicate a broken pipeline)
    let picks: Vec<u32> = la.iter().map(|l| argmax(l)).collect();
    assert!(
        picks
            .iter()
            .enumerate()
            .any(|(i, &p)| i > 0 && p != picks[0]),
        "argmax never moved: {picks:?}"
    );
    assert!(
        la.windows(2).any(|w| w[0] != w[1]),
        "logits identical across steps"
    );

    // KV / indexer / compressor state follows compress_ratios (spec §0/§5):
    // after 8 tokens: L0 (ratio 0) has no cache; L1 (ratio 2) has 4 rows;
    // L2 (ratio 1) has 8 rows of both compress and index keys.
    assert_eq!(
        a.compress_len(0),
        None,
        "ratio-0 layer stores no compressed KV"
    );
    assert_eq!(a.compress_len(1), Some(4));
    assert_eq!(a.index_k_len(1), Some(4));
    assert_eq!(a.compress_len(2), Some(8));
    assert_eq!(a.index_k_len(2), Some(8));
    // L3 is a consumer: it re-sources L2's caches and keeps no state of its own
    assert_eq!(a.compress_len(3), None);
    assert_eq!(a.index_k_len(3), None);
    assert_eq!(a.next_position(), 8);
}

#[test]
fn causal_prefix_logits_are_identical_until_divergence() {
    // decoding the same prefix then diverging must give identical logits up
    // to the divergence point (catches cache-layout / cross-position bugs)
    let mut a = mini_model(7);
    let mut b = mini_model(7);
    let diverged = [29u32, 30, 1, 31]; // different tail, still inside the vocab
    let la = decode_steps(&mut a, &TOKENS);
    let lb = decode_steps(
        &mut b,
        &TOKENS[..4]
            .iter()
            .chain(diverged.iter())
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
    );

    // prefix of 4 tokens: logits identical bit-for-bit
    for step in 0..4 {
        assert_eq!(la[step], lb[step], "prefix step {step} diverged");
    }
    // after divergence: state must differ (at least one later step's logits)
    let diverged_after = (4..8).any(|i| la[i] != lb[i]);
    assert!(diverged_after, "divergence had no effect on later logits");
}

#[test]
fn decode_rejects_nonsequential_positions() {
    let mut m = mini_model(1);
    assert!(m.decode(0, 0).is_ok());
    assert!(m.decode(0, 0).is_err(), "replaying a position must fail");
    assert!(m.decode(0, 5).is_err(), "skipping positions must fail");
    assert!(m.decode(200, 1).is_err(), "out-of-vocab token must fail");
}

#[test]
fn greedy_decode_runs_twenty_steps_from_zero() {
    // the v1 contract: greedy main-head decoding, one token per step,
    // positions start at 0 (prefill = repeated decode). 20 steps takes the
    // window ring past its first overwrite (window 8), the ratio-2
    // compressor through 10 complete pairs, and the candidate blocks well
    // past pinning a single block.
    let mut m = mini_model(99);
    let mut token = 5u32;
    let mut picks = Vec::new();
    for p in 0..20u64 {
        let logits = m.decode(token, p).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()), "step {p}");
        token = argmax(&logits);
        picks.push(token);
    }
    assert_eq!(picks.len(), 20);
    assert!(picks.iter().all(|&t| t < 32));
    // compress caches follow (pos+1)/ratio exactly (spec §5)
    assert_eq!(m.compress_len(1), Some(10));
    assert_eq!(m.compress_len(2), Some(20));
    assert_eq!(m.index_k_len(1), Some(10));
    assert_eq!(m.index_k_len(2), Some(20));
}

#[test]
fn gumbel_sampling_agrees_with_argmax_at_low_temperature() {
    let mut m = mini_model(3);
    let logits = m.decode(1, 0).unwrap();
    let mut g = crate::GumbelSampler::reference();
    assert_eq!(g.sample(&logits, 0.0), argmax(&logits));
    // sampled but strongly peaked: still lands on argmax
    let top = argmax(&logits);
    let hit = g.sample(&logits, 1e-4);
    assert_eq!(hit, top);
}

#[test]
fn synthetic_generation_covers_the_layout() {
    // SyntheticWeights must generate every tensor the layout names — the
    // same inventory from_checkpoint validates — so the test path and the
    // checkpoint path consume identical tensor sets.
    let cfg = mini_config(mini_engram_rows());
    let rcfg = RefConfig::from_model_config(&cfg).unwrap();
    let names = crate::layout::layout(&rcfg);
    let mut store = SyntheticWeights::generate(&cfg, 5).unwrap();
    use crate::store::WeightStore as _;
    for name in names.keys() {
        store.get(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    // and the census (ferroseek_core::weights, minus mtp) is a subset
    let census = ferroseek_core::weights::expected_tensors(&cfg);
    for name in &census {
        if name.starts_with("mtp.") {
            continue; // TODO(spec-decode): drafter is stubbed
        }
        if name.ends_with("wo_a.scale") {
            // raw-HF-export artifact; convert.py dequantizes wo_a to bf16
            // dense (spec §11), so the converted checkpoint has no scale
            continue;
        }
        assert!(
            names.contains_key(name),
            "census tensor {name} missing from the ref layout"
        );
    }
}

#[test]
fn release_config_json_parses_with_release_key_spellings() {
    // The real 1017 export's config.json uses HF key spellings the ModelArgs
    // names do not (`engram_pad_token_id`, `candidate_source_layer_id`,
    // `sliding_window`, `scoring_func`, `routed_scaling_factor`, and YaRN
    // nested under `rope_scaling`). Regression: before the alias fix these
    // all silently fell back to defaults that happened to coincide — a
    // revision that changes any of them would silently rewire the engine.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("config-1017.json");
    let cfg = ferroseek_core::ModelConfig::from_file(&path).expect("release config parses");
    let r = RefConfig::from_model_config(&cfg).expect("ref config");

    assert_eq!(r.dim, 5120);
    assert_eq!(r.n_layers, 40);
    assert_eq!(r.n_heads, 64);
    assert_eq!(r.head_dim, 512);
    assert_eq!(r.rope_dim, 64);
    assert_eq!(r.q_lora, 1280);
    assert_eq!(r.o_lora, 1024);
    assert_eq!(r.o_groups, 8);
    assert_eq!(r.window_size, 128, "sliding_window alias");
    assert_eq!(r.norm_eps, 1e-20);
    assert_eq!(r.kv_sources, vec![2, 8, 14, 20]);
    assert_eq!(r.index_sources, vec![2, 8, 14, 20, 24, 28, 32, 36]);
    assert_eq!(r.candidate_source, 20, "candidate_source_layer_id alias");
    assert_eq!(r.candidate_topk_blocks, 2048);
    assert_eq!(r.candidate_block_size, 8);
    assert_eq!(r.index_n_heads, 32);
    assert_eq!(r.index_head_dim, 128);
    assert_eq!(r.index_topk, 512);
    assert_eq!(r.hc_mult, 4);
    assert_eq!(r.hc_sinkhorn_iters, 20);
    assert_eq!(r.hc_eps, 1e-6);
    // YaRN reads out of rope_scaling
    assert_eq!(r.original_seq_len, 65536);
    assert_eq!(r.rope_factor, 16.0);
    assert_eq!(r.beta_fast, 32);
    assert_eq!(r.beta_slow, 1);
    assert_eq!(r.compress_rope_theta, 160000.0);
    assert_eq!(r.rope_theta, 10000.0);
    assert_eq!(r.n_routed, 384);
    assert_eq!(r.n_activated, 6);
    assert_eq!(r.n_shared, 1);
    assert_eq!(r.inter_dim, 2304);
    assert_eq!(r.score_func, "sqrtsoftplus", "scoring_func alias");
    assert_eq!(r.route_scale, 1.5, "routed_scaling_factor alias");
    assert_eq!(r.swiglu_limit, 10.0);
    // engram wiring and the pad alias
    assert_eq!(r.engram_layers, vec![1, 14]);
    assert_eq!(r.engram_max_ngram, 4);
    assert_eq!(r.engram_n_heads, 8);
    assert_eq!(r.engram_head_dim, 256);
    assert_eq!(r.engram_pad_id, 2, "engram_pad_token_id alias");
    assert_eq!(r.engram_compressed_vocab, 99092);
    assert_eq!(r.engram_prime_start, 16_000_000);
    assert_eq!(r.engram_num_embeddings, vec![384006168, 384016682]);
    assert_eq!(&r.compress_ratios[..4], &[0, 0, 2, 2]);
    assert_eq!(r.compress_ratios[20], 1);
    assert_eq!(r.compress_ratios[39], 1);
    // the engram layout the release table row counts must cover
    let layout = crate::engram::EngramLayout::build(&r).unwrap();
    assert_eq!(layout.layer_ids, vec![1, 14]);
    for (need, have) in layout
        .required_rows()
        .iter()
        .zip(r.engram_num_embeddings.iter())
    {
        assert!(have >= need, "engram rows {have} < required {need}");
    }
}

#[test]
fn checkpoint_path_matches_synthetic_path_bit_for_bit() {
    // Serialize the synthetic weights as a real on-disk safetensors checkpoint
    // (config.json + index + shard), load it through Checkpoint::open +
    // RefModel::from_checkpoint — the mmap/safetensors/dtype-validation path
    // real checkpoints take — and require bit-identical logits. This is the
    // gate-honesty proof: both weight strategies feed the engine the same
    // bytes, and only the byte source differs.
    use crate::store::{DType, WeightStore};
    use ferroseek_core::loader::Checkpoint;
    use std::path::PathBuf;

    let dir: PathBuf =
        std::env::temp_dir().join(format!("ferroseek-ref-ckpt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let cfg = mini_config(mini_engram_rows());
    let mut weights = SyntheticWeights::generate(&cfg, 42).unwrap();

    // config.json + shard + index
    std::fs::write(
        dir.join("config.json"),
        mini_config_json(mini_engram_rows()),
    )
    .unwrap();
    let mut entries: Vec<(String, &'static str, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut store = &mut weights;
    let names: Vec<String> = store.names().cloned().collect();
    for name in &names {
        let v = store.get(name).unwrap();
        let dt = match v.dtype {
            DType::Bf16 => "BF16",
            DType::F32 => "F32",
            DType::Fp8E4m3 => "F8_E4M3",
            DType::E8m0 => "F8_E8M0",
            DType::Fp4Packed => "I8", // packed int8, like convert.py
            DType::U8 => "U8",
        };
        entries.push((name.clone(), dt, v.shape.clone(), v.bytes.to_vec()));
    }
    drop(store);
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, dt, shape, data) in &entries {
        let start = payload.len() as u64;
        payload.extend_from_slice(data);
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dt, "shape": shape, "data_offsets": [start, payload.len() as u64]}),
        );
    }
    let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut shard = (hdr.len() as u64).to_le_bytes().to_vec();
    shard.extend_from_slice(&hdr);
    shard.extend_from_slice(&payload);
    let shard_name = "model-00001-of-00001.safetensors";
    std::fs::write(dir.join(shard_name), &shard).unwrap();
    let idx = serde_json::json!({
        "metadata": {"total_size": payload.len()},
        "weight_map": entries.iter().map(|(n, _, _, _)| (n.clone(), shard_name.to_string())).collect::<std::collections::BTreeMap<_, _>>()
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&idx).unwrap(),
    );

    let mut ckpt = Checkpoint::open(&dir).unwrap();
    let mut from_ckpt = RefModel::from_checkpoint(&mut ckpt).unwrap();
    let mut from_synth = mini_model(42);

    for (p, &t) in TOKENS.iter().take(6).enumerate() {
        let a = from_ckpt.decode(t, p as u64).unwrap();
        let b = from_synth.decode(t, p as u64).unwrap();
        assert_eq!(a, b, "checkpoint and synthetic paths diverge at step {p}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
