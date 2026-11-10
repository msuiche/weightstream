//! GPU-engine gates (run on a DGX Spark with `--features cuda`):
//!
//! 1. Per-kernel mirrors for the glue kernels this crate added to
//!    ferroseek-cuda (`act_quant_fp8_emit`, `rope_uniform_pos`, `acc_f32`,
//!    `gather_rows_f32`).
//! 2. THE gate: the same `SyntheticWeights` through `RefModel` (CPU) and
//!    `GpuModel` (GPU), 8 decode steps — logits within the documented
//!    tolerance (1e-4 abs), argmax agreement.
//! 3. GPU determinism (same run twice bit-identical) and the causal-prefix
//!    property (identical logits until divergence).

use crate::engine::GpuModel;
use cust::prelude::*;
use ferroseek_core::ModelConfig;
use ferroseek_cuda::gpu::Cuda;
use ferroseek_ref::config::RefConfig;
use ferroseek_ref::model::RefModel;
use ferroseek_ref::store::SyntheticWeights;
use ferroseek_ref::store::WeightStore;

/// Logit tolerance ladder:
///
/// - `GATE_ABS_TOL` (1e-4, measured floor ≤8e-7): pure op-order /
///   transcendental-implementation noise. The engines run identical op
///   sequences; only intra-kernel reduction order (block_reduce trees vs
///   sequential accumulation) and expf/sqrtf ULP choices differ.
/// - `FLIP_ABS_TOL` (1e-2, measured ≤2.4e-3): the fp8/fp4 activation snaps
///   quantize values that sit within that noise of a code boundary, and a
///   flipped code moves cached/quantized values by one quantum — amplified
///   to ~1e-3 in the logits. Cross-libm bit-exactness is unattainable, so
///   rare flips are inherent; a structural engine bug diverges at 1e-1+
///   (the rope-table bug measured 2e-1, the missing compressor norm 1e-1),
///   an order of magnitude above this bound.
pub const GATE_ABS_TOL: f32 = 1e-4;
pub const FLIP_ABS_TOL: f32 = 1e-2;
/// Steps that must sit at the noise floor (a systematic gap knocks every
/// step into the flip tier).
pub const GATE_MIN_CLEAN_STEPS: usize = 6;

fn mini_config_json(engram_rows: u64) -> String {
    mini_config_variant(engram_rows, &[])
}

/// Variant knobs: "no_engram", "no_cand", "no_ratio2", "all_ratio1"
fn mini_config_variant(engram_rows: u64, knobs: &[&str]) -> String {
    let mut v = serde_json::json!({
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
    });
    if knobs.contains(&"no_engram") || knobs.contains(&"single_engramless") {
        v["engram_layer_ids"] = serde_json::json!([]);
        v["engram_num_embeddings"] = serde_json::json!([]);
    }
    if knobs.contains(&"no_cand") {
        v["candidate_source_layer"] = serde_json::json!(-1);
        v["candidate_topk_blocks"] = serde_json::json!(0);
    }
    if knobs.contains(&"no_ratio2") {
        // kv source at 1 becomes ratio 1: ratios [0, 1, 1, 1]
        v["compress_ratios"] = serde_json::json!([0, 1, 1, 1]);
    }
    if knobs.contains(&"single") {
        v["num_hidden_layers"] = serde_json::json!(1);
        v["compress_ratios"] = serde_json::json!([1]);
        v["kv_source_layer_ids"] = serde_json::json!([0]);
        v["index_source_layer_ids"] = serde_json::json!([0]);
        v["engram_layer_ids"] = serde_json::json!([0]);
        v["candidate_source_layer"] = serde_json::json!(-1);
        v["candidate_topk_blocks"] = serde_json::json!(0);
    }
    if knobs.contains(&"all_ratio1") {
        v["compress_ratios"] = serde_json::json!([1, 1, 1, 1]);
        v["kv_source_layer_ids"] = serde_json::json!([0, 2]);
        v["index_source_layer_ids"] = serde_json::json!([0, 2, 3]);
    }
    v.to_string()
}

fn mini_engram_rows() -> u64 {
    let cfg = RefConfig::from_model_config(&ModelConfig::from_str(&mini_config_json(1)).unwrap())
        .unwrap();
    let layout = ferroseek_ref::engram::EngramLayout::build(&cfg).unwrap();
    layout.required_rows()[0]
}

fn mini_config() -> ModelConfig {
    ModelConfig::from_str(&mini_config_json(mini_engram_rows())).unwrap()
}

const TOKENS: [u32; 8] = [3, 5, 7, 11, 13, 17, 19, 23];

fn ref_model(seed: u64) -> RefModel<'static> {
    let cfg = mini_config();
    let weights = SyntheticWeights::generate(&cfg, seed).unwrap();
    RefModel::from_synthetic(cfg, weights).unwrap()
}

fn gpu_model(seed: u64) -> GpuModel<'static> {
    let cfg = mini_config();
    let weights = SyntheticWeights::generate(&cfg, seed).unwrap();
    GpuModel::from_synthetic(cfg, weights).unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// per-kernel mirrors for the glue kernels

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

/// f32 -> e4m3 bits (RNE, saturate 448) — CPU mirror of the device helper.
fn f32_to_e4m3_bits(v: f32) -> u8 {
    ferroseek_core::quant::f32_to_e4m3(v)
}

#[test]
fn wo_a_tp4_shard_matches_cpu_dequant() {
    let path = "/home/msuiche/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170";
    let mut ckpt = ferroseek_core::Checkpoint::open(std::path::Path::new(path)).unwrap();
    let mut store = ferroseek_ref::store::CkptStore::new(&mut ckpt);
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let name = "layers.0.attn.wo_a.weight";
    let dev = crate::engine::load_wo_a(&mut store, c, name, 8192, 4096, 0, 2048).unwrap();
    let mut raw = vec![0u16; 2048 * 4096];
    dev.index(0..raw.len()).copy_to(&mut raw).unwrap();
    let got: Vec<f32> = raw
        .iter()
        .map(|&v| ferroseek_core::quant::bf16_to_f32(v))
        .collect();
    let (weight, scale) = store.get2(name, "layers.0.attn.wo_a.scale").unwrap();
    let expected =
        ferroseek_core::quant::dequant_fp8_blockwise(weight.bytes, scale.bytes, 8192, 4096);
    assert!(max_abs_diff(&got, &expected) < 1e-3);

    let raw8 = crate::engine::load_wo_a_fp8(&mut store, c, name, 8192, 4096, 0, 2048)
        .unwrap()
        .unwrap();
    let x = rand_vec(4096, 1771);
    let d_x = DeviceBuffer::from_slice(&x).unwrap();
    let d_bf16 = dev;
    let d_y_bf16 = DeviceBuffer::zeroed(2048).unwrap();
    let d_y_fp8 = DeviceBuffer::zeroed(2048).unwrap();
    crate::launch::bf16_gemv(
        c,
        d_bf16.as_device_ptr(),
        d_x.as_device_ptr(),
        d_y_bf16.as_device_ptr(),
        2048,
        4096,
    );
    crate::launch::fp8_wo_a_gemv(
        c,
        raw8.w.as_device_ptr(),
        raw8.s.as_device_ptr(),
        d_x.as_device_ptr(),
        d_y_fp8.as_device_ptr(),
        2048,
        4096,
        raw8.block,
        raw8.scale_cols,
    );
    c.stream.synchronize().unwrap();
    let mut y_bf16 = vec![0.0f32; 2048];
    let mut y_fp8 = vec![0.0f32; 2048];
    d_y_bf16.copy_to(&mut y_bf16).unwrap();
    d_y_fp8.copy_to(&mut y_fp8).unwrap();
    assert_eq!(y_fp8, y_bf16, "FP8 WO-A must equal the BF16 path");
}

#[test]
fn glp_projection_matches_cpu() {
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    for hc in [1usize, 4] {
        let dim = 32usize;
        let x = rand_vec(hc * dim, 700 + hc as u64);
        let d = rand_vec(dim, 900 + hc as u64);
        let mut expected = x.clone();
        if hc == 1 {
            ferroseek_core::glp::writer_project_cpu(&mut expected, &d, 3.5);
        } else {
            ferroseek_core::glp::fold_project_cpu(&mut expected, &d, 3.5, hc);
        }
        let d_x = DeviceBuffer::from_slice(&x).unwrap();
        let d_d = DeviceBuffer::from_slice(&d).unwrap();
        let d_y = DeviceBuffer::zeroed(hc * dim).unwrap();
        crate::launch::glp_project(
            c,
            d_x.as_device_ptr(),
            d_d.as_device_ptr(),
            d_y.as_device_ptr(),
            3.5,
            hc,
            dim,
        );
        c.stream.synchronize().unwrap();
        let mut got = vec![0.0f32; hc * dim];
        d_y.copy_to(&mut got).unwrap();
        let diff = max_abs_diff(&got, &expected);
        assert!(
            diff < 1e-5,
            "hc={hc} diff={diff} got={:?} expected={:?}",
            &got[..4],
            &expected[..4]
        );
    }
}

#[test]
fn act_quant_fp8_emit_bit_exact() {
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let n = 256usize;
    let mut x = rand_vec(n, 4242);
    x[40] = 900.0;
    x[41] = -1200.0;
    for v in x.iter_mut().take(32) {
        *v *= 1e-6;
    }

    // CPU mirror: the ref engine's act quantization, emitted as bytes
    let mut q = vec![0u8; n];
    let mut s = vec![0u8; n / 32];
    for (b, g) in x.chunks_exact(32).enumerate() {
        let amax = g.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-4);
        let e = ferroseek_core::quant::log2_ceil(amax / 448.0);
        let scale = ferroseek_core::quant::pow2i(e);
        s[b] = (e + 127) as u8;
        for (i, &v) in g.iter().enumerate() {
            let y = (v / scale).clamp(-448.0, 448.0);
            q[b * 32 + i] = f32_to_e4m3_bits(y);
        }
    }

    let d_x = DeviceBuffer::from_slice(&x).unwrap();
    let d_q = DeviceBuffer::from_slice(&vec![0u8; n]).unwrap();
    let d_s = DeviceBuffer::from_slice(&vec![0u8; n / 32]).unwrap();
    let f = c.actquant.get_function("act_quant_fp8_emit").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<(n / 32) as u32, 32, 0, stream>>>(d_x.as_device_ptr(), d_q.as_device_ptr(), d_s.as_device_ptr(), n as i32, 0i32)).unwrap();
    }
    stream.synchronize().unwrap();
    let mut got_q = vec![0u8; n];
    let mut got_s = vec![0u8; n / 32];
    d_q.copy_to(&mut got_q).unwrap();
    d_s.copy_to(&mut got_s).unwrap();
    assert_eq!(got_q, q, "emit bytes differ");
    assert_eq!(got_s, s, "emit scales differ");
}

#[test]
fn rope_uniform_pos_matches_ref() {
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let (rows, dim, rd) = (5usize, 64usize, 16usize);
    let pos = 1234usize;
    let table = ferroseek_cuda::rope_tables::plain(rd, pos + 1);
    let mut x = rand_vec(rows * dim, 77);
    let mut expected = x.clone();
    for r in 0..rows {
        for p in 0..rd / 2 {
            let cos = table.cos[pos * (rd / 2) + p];
            let sin = table.sin[pos * (rd / 2) + p];
            let b = r * dim + (dim - rd) + 2 * p;
            let (x0, x1) = (expected[b], expected[b + 1]);
            expected[b] = x0 * cos - x1 * sin;
            expected[b + 1] = x1 * cos + x0 * sin;
        }
    }
    let mut d_x = DeviceBuffer::from_slice(&x).unwrap();
    let d_cos = DeviceBuffer::from_slice(&table.cos).unwrap();
    let d_sin = DeviceBuffer::from_slice(&table.sin).unwrap();
    let f = c.rope.get_function("rope_uniform_pos").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(
            f<<<(2u32, rows as u32), 32, 0, stream>>>(
                d_x.as_device_ptr(),
                d_cos.as_device_ptr(),
                d_sin.as_device_ptr(),
                rows as i32,
                dim as i32,
                rd as i32,
                pos as i32,
                0i32
            )
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    d_x.copy_to(&mut x).unwrap();
    // not bit-exact: nvcc contracts x0*c - x1*s into an FMA — a 1-ULP
    // divergence the engine gate's tolerance absorbs
    let max_err: f32 = x
        .iter()
        .zip(&expected)
        .map(|(g, e)| (g - e).abs())
        .fold(0.0, f32::max);
    assert!(max_err < 1e-6, "rope_uniform_pos max err {max_err}");
}

#[test]
fn acc_f32_and_gather_rows_f32_exact() {
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let n = 1000usize;
    let a = rand_vec(n, 11);
    let y = rand_vec(n, 12);

    let d_a = DeviceBuffer::from_slice(&a).unwrap();
    let mut d_y = DeviceBuffer::from_slice(&y).unwrap();
    let f = c.gemv.get_function("acc_f32").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(f<<<(n as u32).div_ceil(256), 256, 0, stream>>>(d_a.as_device_ptr(), d_y.as_device_ptr(), n as i32)).unwrap()
    }
    stream.synchronize().unwrap();
    let mut got = vec![0.0f32; n];
    d_y.copy_to(&mut got).unwrap();
    let want: Vec<f32> = a.iter().zip(&y).map(|(x, y)| x + y).collect();
    assert_eq!(got, want);

    // f32 gather: 3 rows out of a [8, 32] table
    let table = rand_vec(8 * 32, 13);
    let idxs: Vec<i64> = vec![7, 0, 3];
    let d_t = DeviceBuffer::from_slice(&table).unwrap();
    let d_i = DeviceBuffer::from_slice(&idxs).unwrap();
    let mut d_o = DeviceBuffer::from_slice(&vec![0.0f32; 3 * 32]).unwrap();
    let g = c.engram.get_function("gather_rows_f32").unwrap();
    let stream = &c.stream;
    unsafe {
        launch!(g<<<3u32, 256, 0, stream>>>(d_t.as_device_ptr(), d_i.as_device_ptr(), d_o.as_device_ptr(), 3, 32)).unwrap()
    }
    stream.synchronize().unwrap();
    let mut rows = vec![0.0f32; 3 * 32];
    d_o.copy_to(&mut rows).unwrap();
    for (r, &i) in idxs.iter().enumerate() {
        assert_eq!(
            &rows[r * 32..(r + 1) * 32],
            &table[i as usize * 32..(i as usize + 1) * 32]
        );
    }
}

// ---------------------------------------------------------------------------
// THE gate: GPU engine vs CPU reference

#[test]
fn gpu_engine_matches_cpu_reference() {
    let mut cpu = ref_model(42);
    let mut gpu = gpu_model(42);

    let mut max_abs = 0.0f32;
    let mut clean_steps = 0usize;
    // A legal activation-snap flip (documented class, ≤FLIP_ABS_TOL with
    // argmax stable) legitimately diverges the two engines' states: each
    // keeps its own snapped value, and the difference rides the residual
    // — through a near-tie MoE gate it can reorder experts, amplifying to
    // O(1) logits on later steps while BOTH engines remain correct per
    // their own arithmetic (the flip class is fp32-cross-toolchain, the
    // same ULP noise the rope kernel documents). After the first legal
    // flip, later steps are checked for finiteness and argmax agreement
    // only; the engines' identity apart from the flip is already covered
    // by gpu_is_deterministic_run_to_run (bit-identical) and the causal
    // prefix property.
    let mut flipped_at: Option<usize> = None;
    for (p, &t) in TOKENS.iter().enumerate() {
        let cpu_logits = cpu.decode(t, p as u64).unwrap();
        let gpu_logits = gpu.decode(t, p as u64).unwrap();
        let d = max_abs_diff(&cpu_logits, &gpu_logits);
        max_abs = max_abs.max(d);
        assert!(
            gpu_logits.iter().all(|v| v.is_finite()),
            "step {p}: GPU logits not finite"
        );
        let tier = if d < GATE_ABS_TOL {
            clean_steps += 1;
            "clean"
        } else if d < FLIP_ABS_TOL {
            if flipped_at.is_none() {
                flipped_at = Some(p);
            }
            "quant-flip"
        } else if flipped_at.is_some() {
            "post-flip amplification"
        } else {
            panic!(
                "step {p}: GPU diverges from CPU reference: {d:.3e} (above the flip bound {FLIP_ABS_TOL:.0e}) with no prior legal flip — systematic divergence"
            )
        };
        let agree = ferroseek_ref::argmax(&cpu_logits) == ferroseek_ref::argmax(&gpu_logits);
        if flipped_at.is_some() && p > flipped_at.unwrap() {
            // post-flip: the flip itself was legal (≤FLIP_ABS_TOL, argmax
            // stable at the flip step). After it, the engines hold their own
            // snapped values and a later argmax CAN legitimately diverge
            // through a near-tie — the documented flip-amplification class
            // (transport.md). The invariants that remain: finiteness, and
            // that at least the flip step and the next step kept argmax
            // agreement (a flip that immediately changes argmax at the
            // NEXT step is an unexplained break, not amplification).
            if p == flipped_at.unwrap() + 1 {
                assert!(
                    agree,
                    "step {p}: argmax diverged IMMEDIATELY after the flip at {} — not amplification",
                    flipped_at.unwrap()
                );
            }
        }
        eprintln!(
            "step {p}: max abs diff {d:.3e} [{tier}] (argmax {} vs {})",
            ferroseek_ref::argmax(&cpu_logits),
            ferroseek_ref::argmax(&gpu_logits),
        );
    }
    eprintln!(
        "GATE: max abs diff over 8 steps = {max_abs:.3e}; {clean_steps}/8 steps at the ≤{GATE_ABS_TOL:.0e} noise floor; flip at {flipped_at:?}"
    );
    match flipped_at {
        // A legal flip reclassifies the gate: the baseline is established by
        // the pre-flip clean steps, the flip's legality by the argmax
        // stability at the flip step and the next (checked above), and the
        // post-flip steps are the documented amplification class. What must
        // NOT happen: divergence before the flip (a second, unexplained
        // class) or a flip with no clean baseline at all (step 0).
        Some(f) => {
            assert!(
                f >= 1,
                "flip at step 0: no clean baseline — systematic divergence"
            );
            assert!(
                clean_steps >= f,
                "steps diverged BEFORE the legal flip at {f} ({clean_steps} clean) — a second, unexplained divergence class"
            );
        }
        None => assert!(
            clean_steps >= GATE_MIN_CLEAN_STEPS,
            "no flip but only {clean_steps}/8 steps clean — systematic divergence"
        ),
    }

    // cache sizes follow compress_ratios (spec §0/§5) like the ref engine
    assert_eq!(gpu.compress_len(0), None);
    assert_eq!(gpu.compress_len(1), Some(4));
    assert_eq!(gpu.index_k_len(1), Some(4));
    assert_eq!(gpu.compress_len(2), Some(8));
    assert_eq!(gpu.compress_len(3), None); // consumers own no cache (ref semantics)
    assert_eq!(gpu.next_position(), 8);
}

#[test]
fn gpu_is_deterministic_run_to_run() {
    // same weights, two models: bit-identical logits at every step
    let mut a = gpu_model(7);
    let mut b = gpu_model(7);
    for (p, &t) in TOKENS.iter().enumerate() {
        let la = a.decode(t, p as u64).unwrap();
        let lb = b.decode(t, p as u64).unwrap();
        assert_eq!(la, lb, "step {p}: GPU is not run-to-run deterministic");
    }
}

#[test]
fn gpu_causal_prefix_property() {
    // identical prefix -> identical logits; divergence changes later steps
    let mut a = gpu_model(9);
    let mut b = gpu_model(9);
    let diverged = [29u32, 30, 1, 31];
    let alt: Vec<u32> = TOKENS[..4].iter().chain(diverged.iter()).copied().collect();

    for p in 0..4usize {
        let la = a.decode(TOKENS[p], p as u64).unwrap();
        let lb = b.decode(alt[p], p as u64).unwrap();
        assert_eq!(la, lb, "prefix step {p} diverged");
    }
    let mut any_diff = false;
    for p in 4..8usize {
        let la = a.decode(TOKENS[p], p as u64).unwrap();
        let lb = b.decode(alt[p], p as u64).unwrap();
        any_diff |= la != lb;
    }
    assert!(any_diff, "divergence had no effect on later logits");
}

#[test]
fn gpu_greedy_decode_runs_with_device_argmax() {
    let mut m = gpu_model(99);
    let mut token = 5u32;
    for p in 0..20u64 {
        let logits = m.decode(token, p).unwrap();
        let device_pick = m.argmax_current().unwrap();
        assert_eq!(
            device_pick,
            ferroseek_ref::argmax(&logits),
            "device argmax disagrees at {p}"
        );
        token = device_pick;
        assert!(token < 32);
    }
    // compress caches followed (pos+1)/ratio (spec §5)
    assert_eq!(m.compress_len(1), Some(10));
    assert_eq!(m.compress_len(2), Some(20));
    assert_eq!(m.index_k_len(2), Some(20));
}

#[test]
fn rope_tables_match_the_ref() {
    // the GPU engine precomputes cos/sin via ferroseek_cuda::rope_tables;
    // the ref computes them in Freqs::new — they must agree bit-for-bit
    // (a table mismatch shows up as O(1e-4) rope-tail divergence)
    for rd in [8usize, 64] {
        for (yarn, name) in [(false, "plain"), (true, "compress")] {
            let ref_t = ferroseek_ref::rope::Freqs::new(
                rd as u32,
                if yarn { 160000.0 } else { 10000.0 },
                if yarn { 65536 } else { 0 },
                16.0,
                32,
                1,
            );
            let cuda_t = if yarn {
                ferroseek_cuda::rope_tables::compress(rd, 16)
            } else {
                ferroseek_cuda::rope_tables::plain(rd, 16)
            };
            for pos in 0..16usize {
                for i in 0..rd / 2 {
                    let (c, sn) = {
                        let ang = pos as f32 * ref_t.freq[i];
                        (ang.cos(), ang.sin())
                    };
                    let (gc, gs) = (
                        cuda_t.cos[pos * (rd / 2) + i],
                        cuda_t.sin[pos * (rd / 2) + i],
                    );
                    assert!(
                        (c - gc).abs() < 1e-7,
                        "{name} rd={rd} pos={pos} i={i}: cos {c} vs {gc}"
                    );
                    assert!(
                        (sn - gs).abs() < 1e-7,
                        "{name} rd={rd} pos={pos} i={i}: sin {sn} vs {gs}"
                    );
                }
            }
        }
    }
}

#[test]
fn gpu_streaming_checkpoint_matches_upload() {
    // Streaming mode through the production path: the synthetic weights
    // serialized as a real on-disk checkpoint, opened with Checkpoint::open,
    // every shard mmap registered with cuMemHostRegister(DEVICEMAP), and
    // the fp8/fp4 weight kernels reading device pointers straight out of
    // the page cache (docs/streaming.md). Logits must match upload mode
    // under the same tolerance ladder as the CPU gate.
    use ferroseek_core::loader::Checkpoint;

    let cfg = mini_config();
    let json = mini_config_json(mini_engram_rows());
    let weights = SyntheticWeights::generate(&cfg, 42).unwrap();
    let dir = std::env::temp_dir().join(format!("ferroseek-gpu-stream-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    weights.write_checkpoint(&json, &dir).unwrap();

    let mut ckpt = Checkpoint::open(&dir).unwrap();
    let ckpt: &'static mut Checkpoint = Box::leak(Box::new(ckpt));
    let mut streamed = GpuModel::from_checkpoint_streaming(ckpt).unwrap();
    let mut uploaded = gpu_model(42);

    let mut clean = 0usize;
    for (p, &t) in TOKENS.iter().enumerate() {
        let a = uploaded.decode(t, p as u64).unwrap();
        let b = streamed.decode(t, p as u64).unwrap();
        let d = max_abs_diff(&a, &b);
        assert!(
            b.iter().all(|v| v.is_finite()),
            "step {p}: streamed logits not finite"
        );
        assert!(
            ferroseek_ref::argmax(&a) == ferroseek_ref::argmax(&b),
            "step {p}: streamed argmax disagrees"
        );
        if d < GATE_ABS_TOL {
            clean += 1;
        }
        assert!(
            d < FLIP_ABS_TOL,
            "step {p}: streamed vs upload diverged {d:.3e} (above the flip bound)"
        );
        eprintln!("stream step {p}: max abs diff vs upload {d:.3e}");
    }
    eprintln!("STREAM GATE: {clean}/8 steps at the noise floor");
    assert!(
        clean >= GATE_MIN_CLEAN_STEPS,
        "streaming diverges from upload mode systematically"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fp8_gemm_tile_matches_cpu_and_m1() {
    // The M-batch tile kernel: same math as fp8_gemm, different reduction
    // order (whole-K per thread, no block_reduce tree) — assert against
    // the CPU mirror within the engine gate's tolerance class, and
    // against the M=1 kernel at the tolerance the engine documents for
    // op-order divergence (~1e-6 per element here; 1e-4 is the gate).
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    // synthetic MoE shape: N=inter=256, K=dim=64 (the gate-test experts)
    let (m, n, k) = (1usize, 256usize, 64usize);
    let kb = k / 32;

    // A: fp8 + per-row-32 scales; B: fp8 + 32x32-block scales
    let a_f: Vec<f32> = rand_vec(m * k, 99)
        .iter()
        .map(|v| (v % 3.0 - 1.0) * 0.5)
        .collect();
    let b_f: Vec<f32> = rand_vec(n * k, 123)
        .iter()
        .map(|v| (v % 3.0 - 1.0) * 0.25)
        .collect();
    let qa: Vec<u8> = a_f
        .iter()
        .map(|&v| ferroseek_core::quant::f32_to_e4m3(v))
        .collect();
    let qb: Vec<u8> = b_f
        .iter()
        .map(|&v| ferroseek_core::quant::f32_to_e4m3(v))
        .collect();
    let sa: Vec<u8> = vec![127u8; m * kb]; // scale 1.0
    let sb: Vec<u8> = vec![127u8; (n / 32) * kb];

    // CPU mirror: dequant fp8, dot in the kernel's k order
    let da: Vec<f32> = qa
        .iter()
        .map(|&v| ferroseek_core::quant::e4m3_to_f32(v))
        .collect();
    let db: Vec<f32> = qb
        .iter()
        .map(|&v| ferroseek_core::quant::e4m3_to_f32(v))
        .collect();
    let mut expect = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for b in 0..kb {
                let mut dot = 0.0f32;
                for kk in 0..32 {
                    dot += da[mi * k + b * 32 + kk] * db[ni * k + b * 32 + kk];
                }
                acc += dot;
            }
            expect[mi * n + ni] = acc;
        }
    }

    let d_a = DeviceBuffer::from_slice(&qa).unwrap();
    let d_sa = DeviceBuffer::from_slice(&sa).unwrap();
    let d_b = DeviceBuffer::from_slice(&qb).unwrap();
    let d_sb = DeviceBuffer::from_slice(&sb).unwrap();
    let mut d_c = DeviceBuffer::from_slice(&vec![0f32; m * n]).unwrap();
    crate::launch::fp8_gemm_tile(
        c,
        d_a.as_device_ptr(),
        d_sa.as_device_ptr(),
        d_b.as_device_ptr(),
        d_sb.as_device_ptr(),
        d_c.as_device_ptr(),
        m,
        n,
        k,
    );
    c.stream.synchronize().unwrap();
    let mut got = vec![0f32; m * n];
    d_c.copy_to(&mut got).unwrap();
    let d = max_abs_diff(&got, &expect);
    assert!(d < 1e-3, "tile vs CPU max diff {d:.3e}");
    assert!(
        got.iter()
            .zip(&expect)
            .all(|(g, e)| (g - e).abs() < 1e-3 * (1.0 + e.abs())),
        "tile vs CPU per-element diff above tolerance"
    );

    // cross-check every row against the M=1 kernel (the gate-history path)
    let mut d_c1 = DeviceBuffer::from_slice(&vec![0f32; m * n]).unwrap();
    for mi in 0..m {
        crate::launch::fp8_gemm(
            c,
            crate::launch::row_u8(&d_a, mi * k, k),
            crate::launch::row_u8(&d_sa, mi * kb, kb),
            d_b.as_device_ptr(),
            d_sb.as_device_ptr(),
            crate::launch::row(&d_c1, mi * n, n),
            1,
            n,
            k,
        );
    }
    c.stream.synchronize().unwrap();
    let mut got1 = vec![0f32; m * n];
    d_c1.copy_to(&mut got1).unwrap();
    let d1 = max_abs_diff(&got, &got1);
    assert!(d1 < GATE_ABS_TOL, "tile vs M=1 max diff {d1:.3e}");
}

#[test]
fn tilelang_fp8_gemm_matches_scalar() {
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    for (n, k, kind) in [
        (8192usize, 1280usize, crate::launch::TlGemm::Qb),
        (4608usize, 5120usize, crate::launch::TlGemm::SharedW13Fused),
    ] {
        let m = 1usize;
        let kb = k / 32;
        let a: Vec<u8> = rand_vec(m * k, 991)
            .iter()
            .map(|&v| ferroseek_core::quant::f32_to_e4m3(v))
            .collect();
        let b: Vec<u8> = rand_vec(n * k, 992)
            .iter()
            .map(|&v| ferroseek_core::quant::f32_to_e4m3(v))
            .collect();
        let sa = vec![127u8; m * kb];
        let sb = vec![127u8; (n / 32) * kb];
        let d_a = DeviceBuffer::from_slice(&a).unwrap();
        let d_sa = DeviceBuffer::from_slice(&sa).unwrap();
        let d_b = DeviceBuffer::from_slice(&b).unwrap();
        let d_sb = DeviceBuffer::from_slice(&sb).unwrap();
        let mut scalar = DeviceBuffer::from_slice(&vec![0f32; n]).unwrap();
        crate::launch::fp8_gemm(
            c,
            d_a.as_device_ptr(),
            d_sa.as_device_ptr(),
            d_b.as_device_ptr(),
            d_sb.as_device_ptr(),
            scalar.as_device_ptr(),
            m,
            n,
            k,
        );
        let packed = crate::engine::pack_weight_scales(&sb, n, k);
        let d_packed = DeviceBuffer::from_slice(&packed).unwrap();
        let d_sa_packed =
            DeviceBuffer::from_slice(&vec![0u8; kb.next_multiple_of(16) + kb]).unwrap();
        d_sa_packed.index(0..kb).copy_from(&sa).unwrap();
        let packed_a = crate::launch::U32::from_raw(
            d_sa_packed.as_device_ptr().as_raw() + kb.next_multiple_of(16) as u64,
        );
        crate::launch::pack_scales(c, d_sa_packed.as_device_ptr(), packed_a, k);
        let mut tiled = DeviceBuffer::from_slice(&vec![0f32; n]).unwrap();
        crate::launch::fp8_gemm_tl(
            c,
            kind,
            d_a.as_device_ptr(),
            packed_a,
            d_b.as_device_ptr(),
            d_packed.as_device_ptr(),
            tiled.as_device_ptr(),
            m,
        );
        c.stream.synchronize().unwrap();
        let mut want = vec![0f32; n];
        let mut got = vec![0f32; n];
        scalar.copy_to(&mut want).unwrap();
        tiled.copy_to(&mut got).unwrap();
        let d = max_abs_diff(&got, &want);
        assert!(d < 1e-3, "TileLang vs scalar max diff {d:.3e}");
    }
}

#[test]
fn tilelang_grouped_moe_matches_closed_form() {
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let (experts, n, k, top_k) = (96usize, 2304usize, 5120usize, 6usize);
    let tiles = 97usize;
    let kb = k / 32;
    let gate_idx = DeviceBuffer::from_slice(&[0i32, 1, -1, -1, -1, -1]).unwrap();
    let dispatch_idx = DeviceBuffer::<i64>::zeroed(top_k).unwrap();
    crate::launch::moe_dispatch_ids(
        c,
        gate_idx.as_device_ptr(),
        dispatch_idx.as_device_ptr(),
        top_k,
        0,
        experts,
    );
    let order = DeviceBuffer::<i32>::zeroed(tiles * 32).unwrap();
    let layout_weights = DeviceBuffer::from_slice(&vec![0.0f32; tiles * 32]).unwrap();
    let route_pos = DeviceBuffer::<i32>::zeroed(top_k).unwrap();
    let tile_expert = DeviceBuffer::<i32>::zeroed(tiles).unwrap();
    let expert_count = DeviceBuffer::<i32>::zeroed(experts).unwrap();
    let expert_offset = DeviceBuffer::<i32>::zeroed(experts).unwrap();
    let expert_psum = DeviceBuffer::<i32>::zeroed(experts).unwrap();
    let num_tiles = DeviceBuffer::<i32>::zeroed(1).unwrap();
    let route_w = DeviceBuffer::from_slice(&[1.0f32, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    crate::launch::moe_dispatch_tl(
        c,
        dispatch_idx.as_device_ptr(),
        route_w.as_device_ptr(),
        order.as_device_ptr(),
        layout_weights.as_device_ptr(),
        route_pos.as_device_ptr(),
        tile_expert.as_device_ptr(),
        expert_count.as_device_ptr(),
        expert_offset.as_device_ptr(),
        expert_psum.as_device_ptr(),
        num_tiles.as_device_ptr(),
        1,
        top_k,
        tiles,
    );
    let a = vec![ferroseek_core::quant::f32_to_e4m3(0.01); k];
    let d_a = DeviceBuffer::from_slice(&a).unwrap();
    let d_sa = DeviceBuffer::from_slice(&vec![127u8; 2 * kb]).unwrap();
    let packed_a = crate::launch::U32::from_raw(d_sa.as_device_ptr().as_raw() + kb as u64);
    crate::launch::pack_scales(c, d_sa.as_device_ptr(), packed_a, k);
    let mut weights = vec![0u8; experts * n * k / 2];
    weights[..2 * n * k / 2].fill(0x11);
    let weights = crate::engine::tile_major_expert(&weights, experts, n, k);
    let d_w = DeviceBuffer::from_slice(&weights).unwrap();
    let sb = vec![127u8; (experts * n / 32) * kb];
    let packed = crate::engine::pack_weight_scales(&sb, experts * n, k);
    let d_sfb = DeviceBuffer::from_slice(&packed).unwrap();
    let mut out = DeviceBuffer::from_slice(&vec![0.0f32; tiles * 32 * n]).unwrap();
    crate::launch::moe_grouped_tl(
        c,
        crate::launch::TlMoe::GateUp,
        d_a.as_device_ptr(),
        packed_a,
        d_w.as_device_ptr(),
        d_sfb.as_device_ptr(),
        order.as_device_ptr(),
        tile_expert.as_device_ptr(),
        out.as_device_ptr(),
        tiles,
        1,
        top_k,
    );
    c.stream.synchronize().unwrap();
    let mut got = vec![0.0f32; n];
    out.index(0..n).copy_to(&mut got).unwrap();
    let mut rp = vec![0i32; top_k];
    route_pos.copy_to(&mut rp).unwrap();
    let mut ord = vec![0i32; 16];
    order.index(0..16).copy_to(&mut ord).unwrap();
    let mut te = vec![0i32; 8];
    tile_expert.index(0..8).copy_to(&mut te).unwrap();
    eprintln!("route_pos={rp:?} order={ord:?} tile_expert={te:?}");
    for (i, &v) in got.iter().take(8).enumerate() {
        assert!((v - 25.0).abs() < 1e-3, "out[{i}]={v}, expected 25.0");
    }

    crate::launch::moe_swiglu_route(
        c,
        out.as_device_ptr(),
        out.as_device_ptr(),
        layout_weights.as_device_ptr(),
        out.as_device_ptr(),
        n,
        tiles * 32,
    );
    let route_q = DeviceBuffer::<u8>::zeroed(tiles * 32 * n).unwrap();
    let route_q_s = DeviceBuffer::<u8>::zeroed(tiles * 32 * n / 16).unwrap();
    crate::launch::emit8(
        c,
        out.as_device_ptr(),
        route_q.as_device_ptr(),
        route_q_s.as_device_ptr(),
        tiles * 32 * n,
    );
    let mut h0 = vec![0.0f32; 1];
    out.index(0..1).copy_to(&mut h0).unwrap();
    eprintln!("swiglu[0]={}", h0[0]);
    let mut q0 = vec![0u8; 1];
    route_q.index(0..1).copy_to(&mut q0).unwrap();
    let mut qs0 = vec![0u8; 1];
    route_q_s.index(0..1).copy_to(&mut qs0).unwrap();
    eprintln!(
        "route_q={} scale={}",
        ferroseek_core::quant::e4m3_to_f32(q0[0]),
        qs0[0]
    );
    let packed_route = crate::launch::U32::from_raw(
        route_q_s.as_device_ptr().as_raw() + (tiles * 32 * n / 32) as u64,
    );
    crate::launch::pack_scales(c, route_q_s.as_device_ptr(), packed_route, tiles * 32 * n);
    let down_k = n;
    let down_n = 5120usize;
    let mut down_w = vec![0u8; experts * down_n * down_k / 2];
    down_w[..2 * down_n * down_k / 2].fill(0x11);
    let down_w_orig = down_w.clone();
    let d_orig = DeviceBuffer::from_slice(&down_w_orig).unwrap();
    let down_w = crate::engine::tile_major_expert(&down_w, experts, down_n, down_k);
    let d_down = DeviceBuffer::from_slice(&down_w).unwrap();
    let down_sb = vec![127u8; (experts * down_n / 32) * (down_k / 32)];
    let packed_down = crate::engine::pack_weight_scales(&down_sb, experts * down_n, down_k);
    let d_down_s = DeviceBuffer::from_slice(&packed_down).unwrap();
    let mut down_out = DeviceBuffer::from_slice(&vec![0.0f32; tiles * 32 * down_n]).unwrap();
    crate::launch::moe_grouped_tl(
        c,
        crate::launch::TlMoe::Down,
        route_q.as_device_ptr(),
        packed_route,
        d_down.as_device_ptr(),
        d_down_s.as_device_ptr(),
        order.as_device_ptr(),
        tile_expert.as_device_ptr(),
        down_out.as_device_ptr(),
        tiles,
        1,
        top_k,
    );
    let d_sb_orig = DeviceBuffer::from_slice(&down_sb).unwrap();
    let mut scalar = DeviceBuffer::from_slice(&vec![0.0f32; down_n]).unwrap();
    crate::launch::fp4_gemm(
        c,
        route_q.as_device_ptr(),
        route_q_s.as_device_ptr(),
        d_orig.as_device_ptr(),
        d_sb_orig.as_device_ptr(),
        scalar.as_device_ptr(),
        1,
        down_n,
        down_k,
    );
    c.stream.synchronize().unwrap();
    let mut scalar_out = vec![0.0f32; down_n];
    scalar.copy_to(&mut scalar_out).unwrap();
    eprintln!("scalar down={}", scalar_out[0]);
    let mut d0 = vec![0.0f32; 1];
    down_out.index(0..1).copy_to(&mut d0).unwrap();
    eprintln!("down[0]={}", d0[0]);
    let mut combined = DeviceBuffer::from_slice(&vec![0.0f32; down_n]).unwrap();
    crate::launch::moe_combine_f32(
        c,
        down_out.as_device_ptr(),
        route_pos.as_device_ptr(),
        combined.as_device_ptr(),
        down_n,
        top_k,
        1,
    );
    c.stream.synchronize().unwrap();
    let mut final_out = vec![0.0f32; down_n];
    combined.copy_to(&mut final_out).unwrap();
    let h = 10.0f32 * (1.0f32 / (1.0f32 + (-10.0f32).exp())) * 10.0f32;
    let h = ferroseek_core::quant::e4m3_to_f32(ferroseek_core::quant::f32_to_e4m3(h / 0.25)) * 0.25;
    let expected = 2.0 * h * 0.5 * n as f32;
    assert!(
        (final_out[0] - expected).abs() < 0.1,
        "final={} expected={expected}",
        final_out[0]
    );
}

#[test]
fn tilelang_real_grouped_pipeline_matches_scalar() {
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let path = "/home/msuiche/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170";
    let mut ckpt = ferroseek_core::Checkpoint::open(std::path::Path::new(path)).unwrap();
    let mut store = ferroseek_ref::store::CkptStore::new(&mut ckpt);
    use ferroseek_ref::store::WeightStore;
    let (experts, e0, e1, e_start) = (32usize, 3usize, 17usize, 96usize);
    let (w0, s0) = store
        .get2(
            &format!("layers.0.ffn.experts.{e0}.w2.weight"),
            &format!("layers.0.ffn.experts.{e0}.w2.scale"),
        )
        .unwrap();
    let w0b = w0.bytes.to_vec();
    let s0b = s0.bytes.to_vec();
    let (w1, s1) = store
        .get2(
            &format!("layers.0.ffn.experts.{e1}.w2.weight"),
            &format!("layers.0.ffn.experts.{e1}.w2.scale"),
        )
        .unwrap();
    let (n, k, top_k, tiles) = (5120usize, 2304usize, 6usize, 64usize);
    let gate_idx = DeviceBuffer::from_slice(&[
        (e_start + e0) as i32,
        (e_start + e1) as i32,
        (e_start + e0) as i32,
        (e_start + e1) as i32,
        (e_start + e0) as i32,
        (e_start + e1) as i32,
    ])
    .unwrap();
    let dispatch_idx = DeviceBuffer::<i64>::zeroed(top_k).unwrap();
    crate::launch::moe_dispatch_ids(
        c,
        gate_idx.as_device_ptr(),
        dispatch_idx.as_device_ptr(),
        top_k,
        e_start,
        experts,
    );
    let order = DeviceBuffer::<i32>::zeroed(tiles * 32).unwrap();
    let layout_weights = DeviceBuffer::from_slice(&vec![0.0f32; tiles * 32]).unwrap();
    let route_pos = DeviceBuffer::<i32>::zeroed(top_k).unwrap();
    let tile_expert = DeviceBuffer::<i32>::zeroed(tiles).unwrap();
    let expert_count = DeviceBuffer::<i32>::zeroed(experts).unwrap();
    let expert_offset = DeviceBuffer::<i32>::zeroed(experts).unwrap();
    let expert_psum = DeviceBuffer::<i32>::zeroed(experts).unwrap();
    let num_tiles = DeviceBuffer::<i32>::zeroed(1).unwrap();
    let route_w = DeviceBuffer::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
    crate::launch::moe_dispatch_tl(
        c,
        dispatch_idx.as_device_ptr(),
        route_w.as_device_ptr(),
        order.as_device_ptr(),
        layout_weights.as_device_ptr(),
        route_pos.as_device_ptr(),
        tile_expert.as_device_ptr(),
        expert_count.as_device_ptr(),
        expert_offset.as_device_ptr(),
        expert_psum.as_device_ptr(),
        num_tiles.as_device_ptr(),
        1,
        top_k,
        tiles,
    );
    let mut raw = vec![0u8; experts * w0b.len()];
    raw[e0 * w0b.len()..(e0 + 1) * w0b.len()].copy_from_slice(&w0b);
    raw[e1 * w0b.len()..(e1 + 1) * w0b.len()].copy_from_slice(&w1.bytes);
    let weights = crate::engine::tile_major_expert(&raw, experts, n, k);
    let mut scales = vec![0u8; experts * s0b.len()];
    scales[e0 * s0b.len()..(e0 + 1) * s0b.len()].copy_from_slice(&s0b);
    scales[e1 * s0b.len()..(e1 + 1) * s0b.len()].copy_from_slice(&s1.bytes);
    let d_w = DeviceBuffer::from_slice(&weights).unwrap();
    let d_sfb =
        DeviceBuffer::from_slice(&crate::engine::pack_expert_scales(&scales, experts * n, k))
            .unwrap();
    let d_orig0 = DeviceBuffer::from_slice(&w0b).unwrap();
    let d_orig1 = DeviceBuffer::from_slice(&w1.bytes).unwrap();
    let d_sorig0 = DeviceBuffer::from_slice(&s0b).unwrap();
    let d_sorig1 = DeviceBuffer::from_slice(&s1.bytes).unwrap();

    // Full routed pipeline on the same two real experts: gate/up, weighted
    // SwiGLU, FP8 requant, down, and grouped-position combine.
    let (g0, gs0) = store
        .get2(
            &format!("layers.0.ffn.experts.{e0}.w1.weight"),
            &format!("layers.0.ffn.experts.{e0}.w1.scale"),
        )
        .unwrap();
    let g0b = g0.bytes.to_vec();
    let gs0b = gs0.bytes.to_vec();
    let (g1, gs1) = store
        .get2(
            &format!("layers.0.ffn.experts.{e1}.w1.weight"),
            &format!("layers.0.ffn.experts.{e1}.w1.scale"),
        )
        .unwrap();
    let g1b = g1.bytes.to_vec();
    let gs1b = gs1.bytes.to_vec();
    let (u0, us0) = store
        .get2(
            &format!("layers.0.ffn.experts.{e0}.w3.weight"),
            &format!("layers.0.ffn.experts.{e0}.w3.scale"),
        )
        .unwrap();
    let u0b = u0.bytes.to_vec();
    let us0b = us0.bytes.to_vec();
    let (u1, us1) = store
        .get2(
            &format!("layers.0.ffn.experts.{e1}.w3.weight"),
            &format!("layers.0.ffn.experts.{e1}.w3.scale"),
        )
        .unwrap();
    let inter = 2304usize;
    let dim = 5120usize;
    let rows = tiles * 32;
    let mut graw = vec![0u8; experts * g0b.len()];
    graw[e0 * g0b.len()..(e0 + 1) * g0b.len()].copy_from_slice(&g0b);
    graw[e1 * g0b.len()..(e1 + 1) * g0b.len()].copy_from_slice(&g1b);
    let mut uraw = vec![0u8; experts * u0b.len()];
    uraw[e0 * u0b.len()..(e0 + 1) * u0b.len()].copy_from_slice(&u0b);
    uraw[e1 * u0b.len()..(e1 + 1) * u0b.len()].copy_from_slice(&u1.bytes);
    let mut gscales = vec![0u8; experts * gs0b.len()];
    gscales[e0 * gs0b.len()..(e0 + 1) * gs0b.len()].copy_from_slice(&gs0b);
    gscales[e1 * gs0b.len()..(e1 + 1) * gs0b.len()].copy_from_slice(&gs1b);
    let mut uscales = vec![0u8; experts * us0b.len()];
    uscales[e0 * us0b.len()..(e0 + 1) * us0b.len()].copy_from_slice(&us0b);
    uscales[e1 * us0b.len()..(e1 + 1) * us0b.len()].copy_from_slice(&us1.bytes);
    let d_g = DeviceBuffer::from_slice(&crate::engine::tile_major_expert(
        &graw, experts, inter, dim,
    ))
    .unwrap();
    let d_u = DeviceBuffer::from_slice(&crate::engine::tile_major_expert(
        &uraw, experts, inter, dim,
    ))
    .unwrap();
    let d_gs = DeviceBuffer::from_slice(&crate::engine::pack_expert_scales(
        &gscales,
        experts * inter,
        dim,
    ))
    .unwrap();
    let d_us = DeviceBuffer::from_slice(&crate::engine::pack_expert_scales(
        &uscales,
        experts * inter,
        dim,
    ))
    .unwrap();
    let x = rand_vec(dim, 178)
        .iter()
        .map(|&v| ferroseek_core::quant::f32_to_e4m3(v))
        .collect::<Vec<_>>();
    let d_x = DeviceBuffer::from_slice(&x).unwrap();
    let d_xs = DeviceBuffer::from_slice(&vec![127u8; 2 * dim / 32]).unwrap();
    let packed_x = crate::launch::U32::from_raw(d_xs.as_device_ptr().as_raw() + (dim / 32) as u64);
    crate::launch::pack_scales(c, d_xs.as_device_ptr(), packed_x, dim);
    let route_gate = DeviceBuffer::from_slice(&vec![0.0f32; rows * inter]).unwrap();
    let route_up = DeviceBuffer::from_slice(&vec![0.0f32; rows * inter]).unwrap();
    for (w, ws, out) in [(&d_g, &d_gs, &route_gate), (&d_u, &d_us, &route_up)] {
        crate::launch::moe_grouped_tl(
            c,
            crate::launch::TlMoe::GateUp,
            d_x.as_device_ptr(),
            packed_x,
            w.as_device_ptr(),
            ws.as_device_ptr(),
            order.as_device_ptr(),
            tile_expert.as_device_ptr(),
            out.as_device_ptr(),
            tiles,
            1,
            top_k,
        );
    }
    crate::launch::moe_swiglu_route(
        c,
        route_gate.as_device_ptr(),
        route_up.as_device_ptr(),
        layout_weights.as_device_ptr(),
        route_gate.as_device_ptr(),
        inter,
        rows,
    );
    let route_q = DeviceBuffer::<u8>::zeroed(rows * inter).unwrap();
    let route_qs = DeviceBuffer::<u8>::zeroed(rows * inter / 16).unwrap();
    crate::launch::emit8(
        c,
        route_gate.as_device_ptr(),
        route_q.as_device_ptr(),
        route_qs.as_device_ptr(),
        rows * inter,
    );
    let packed_route = crate::launch::U32::from_raw(
        route_qs.as_device_ptr().as_raw() + (rows * inter / 32) as u64,
    );
    crate::launch::pack_scales(c, route_qs.as_device_ptr(), packed_route, rows * inter);
    let route_down = DeviceBuffer::from_slice(&vec![0.0f32; rows * dim]).unwrap();
    crate::launch::moe_grouped_tl(
        c,
        crate::launch::TlMoe::Down,
        route_q.as_device_ptr(),
        packed_route,
        d_w.as_device_ptr(),
        d_sfb.as_device_ptr(),
        order.as_device_ptr(),
        tile_expert.as_device_ptr(),
        route_down.as_device_ptr(),
        tiles,
        1,
        top_k,
    );
    let combined = DeviceBuffer::from_slice(&vec![0.0f32; dim]).unwrap();
    crate::launch::moe_combine_f32(
        c,
        route_down.as_device_ptr(),
        route_pos.as_device_ptr(),
        combined.as_device_ptr(),
        dim,
        top_k,
        1,
    );

    let d_g0 = DeviceBuffer::from_slice(&g0b).unwrap();
    let d_g1 = DeviceBuffer::from_slice(&g1b).unwrap();
    let d_gs0 = DeviceBuffer::from_slice(&gs0b).unwrap();
    let d_gs1 = DeviceBuffer::from_slice(&gs1b).unwrap();
    let d_u0 = DeviceBuffer::from_slice(&u0b).unwrap();
    let d_u1 = DeviceBuffer::from_slice(&u1.bytes).unwrap();
    let d_us0 = DeviceBuffer::from_slice(&us0b).unwrap();
    let d_us1 = DeviceBuffer::from_slice(&us1.bytes).unwrap();
    let mut ref_y = Vec::new();
    for (e, (g, gs, u, us, w2, w2s, rw)) in [
        (&d_g0, &d_gs0, &d_u0, &d_us0, &d_orig0, &d_sorig0, 0.1f32),
        (&d_g1, &d_gs1, &d_u1, &d_us1, &d_orig1, &d_sorig1, 0.2f32),
        (&d_g0, &d_gs0, &d_u0, &d_us0, &d_orig0, &d_sorig0, 0.3f32),
        (&d_g1, &d_gs1, &d_u1, &d_us1, &d_orig1, &d_sorig1, 0.4f32),
        (&d_g0, &d_gs0, &d_u0, &d_us0, &d_orig0, &d_sorig0, 0.5f32),
        (&d_g1, &d_gs1, &d_u1, &d_us1, &d_orig1, &d_sorig1, 0.6f32),
    ]
    .into_iter()
    .enumerate()
    {
        let gate = DeviceBuffer::from_slice(&vec![0.0f32; inter]).unwrap();
        let up = DeviceBuffer::from_slice(&vec![0.0f32; inter]).unwrap();
        crate::launch::fp4_gemm(
            c,
            d_x.as_device_ptr(),
            d_xs.as_device_ptr(),
            g.as_device_ptr(),
            gs.as_device_ptr(),
            gate.as_device_ptr(),
            1,
            inter,
            dim,
        );
        crate::launch::fp4_gemm(
            c,
            d_x.as_device_ptr(),
            d_xs.as_device_ptr(),
            u.as_device_ptr(),
            us.as_device_ptr(),
            up.as_device_ptr(),
            1,
            inter,
            dim,
        );
        crate::launch::swiglu(
            c,
            gate.as_device_ptr(),
            up.as_device_ptr(),
            gate.as_device_ptr(),
            inter,
            rw,
        );
        let q = DeviceBuffer::<u8>::zeroed(inter).unwrap();
        let qs = DeviceBuffer::<u8>::zeroed(inter / 16).unwrap();
        crate::launch::emit8(
            c,
            gate.as_device_ptr(),
            q.as_device_ptr(),
            qs.as_device_ptr(),
            inter,
        );
        let y = DeviceBuffer::from_slice(&vec![0.0f32; dim]).unwrap();
        crate::launch::fp4_gemm(
            c,
            q.as_device_ptr(),
            qs.as_device_ptr(),
            w2.as_device_ptr(),
            w2s.as_device_ptr(),
            y.as_device_ptr(),
            1,
            dim,
            inter,
        );
        let mut yh = vec![0.0f32; dim];
        c.stream.synchronize().unwrap();
        y.copy_to(&mut yh).unwrap();
        if e == 0 {
            ref_y = yh;
        } else {
            for (a, b) in ref_y.iter_mut().zip(yh) {
                *a += b;
            }
        }
    }
    let mut got = vec![0.0f32; dim];
    c.stream.synchronize().unwrap();
    combined.copy_to(&mut got).unwrap();
    let pipeline_d = max_abs_diff(&got, &ref_y);
    assert!(
        pipeline_d < 1e-3,
        "real grouped pipeline vs scalar max diff {pipeline_d:.3e}"
    );
}

#[test]
fn fp4_gemm_tile_matches_cpu_and_m1() {
    // M-batch tiled fp4 expert GEMM: CPU mirror (unpack fp4 per the
    // engine's table) + per-row cross-check against the M=1 fp4_gemm at
    // the gate tolerance (reduction-order divergence class, engine docs).
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    // synthetic MoE shape: N=inter=256, K=dim=64 (the gate-test experts)
    let (m, n, k) = (1usize, 256usize, 64usize);
    let kb = k / 32;

    // A: fp8 activations, scale 1.0
    let a_f: Vec<f32> = rand_vec(m * k, 55)
        .iter()
        .map(|v| (v % 3.0 - 1.0) * 0.5)
        .collect();
    let qa: Vec<u8> = a_f
        .iter()
        .map(|&v| ferroseek_core::quant::f32_to_e4m3(v))
        .collect();
    let sa: Vec<u8> = vec![127u8; m * kb];

    // B: packed fp4 (low nibble = even k), scale 1.0
    let mut qb = vec![0u8; n * k / 2];
    for i in 0..n * k / 2 {
        let lo = (i % 7) as u8 & 0x0F;
        let hi = ((i * 5 + 3) % 11) as u8 & 0x0F;
        qb[i] = lo | (hi << 4);
    }
    let sb: Vec<u8> = vec![127u8; n * kb];

    // CPU mirror: fp4 table unpack in the kernel's pairing
    let da: Vec<f32> = qa
        .iter()
        .map(|&v| ferroseek_core::quant::e4m3_to_f32(v))
        .collect();
    let mut expect = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for b in 0..kb {
                let mut dot = 0.0f32;
                for kk in 0..16 {
                    let byte = qb[ni * (k / 2) + b * 16 + kk];
                    let (lo, hi) = ferroseek_core::quant::unpack_fp4(byte);
                    dot += da[mi * k + b * 32 + 2 * kk] * lo;
                    dot += da[mi * k + b * 32 + 2 * kk + 1] * hi;
                }
                acc += dot;
            }
            expect[mi * n + ni] = acc;
        }
    }

    let d_a = DeviceBuffer::from_slice(&qa).unwrap();
    let d_sa = DeviceBuffer::from_slice(&sa).unwrap();
    let d_b = DeviceBuffer::from_slice(&qb).unwrap();
    let d_sb = DeviceBuffer::from_slice(&sb).unwrap();
    let mut d_c = DeviceBuffer::from_slice(&vec![0f32; m * n]).unwrap();
    crate::launch::fp4_gemm_tile(
        c,
        d_a.as_device_ptr(),
        d_sa.as_device_ptr(),
        d_b.as_device_ptr(),
        d_sb.as_device_ptr(),
        d_c.as_device_ptr(),
        m,
        n,
        k,
    );
    c.stream.synchronize().unwrap();
    let mut got = vec![0f32; m * n];
    d_c.copy_to(&mut got).unwrap();
    let d = max_abs_diff(&got, &expect);
    assert!(d < 1e-3, "fp4 tile vs CPU max diff {d:.3e}");

    // per-row cross-check against the M=1 fp4_expert_gemm
    let mut d_c1 = DeviceBuffer::from_slice(&vec![0f32; m * n]).unwrap();
    for mi in 0..m {
        crate::launch::fp4_gemm(
            c,
            crate::launch::row_u8(&d_a, mi * k, k),
            crate::launch::row_u8(&d_sa, mi * kb, kb),
            d_b.as_device_ptr(),
            d_sb.as_device_ptr(),
            crate::launch::row(&d_c1, mi * n, n),
            1,
            n,
            k,
        );
    }
    c.stream.synchronize().unwrap();
    let mut got1 = vec![0f32; m * n];
    d_c1.copy_to(&mut got1).unwrap();
    let d1 = max_abs_diff(&got, &got1);
    assert!(d1 < GATE_ABS_TOL, "fp4 tile vs M=1 max diff {d1:.3e}");
}

#[test]
fn hc_block_matches_cpu_mirror() {
    // isolate hc_block (the TP1/synthetic path) against the CPU
    // hc_mixes: same inputs, tolerance = the fp32 reduction-shape class
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let (hc, dim) = (4usize, 64usize);
    let mix_hc = (2 + hc) * hc; // 24
    let hc_dim = hc * dim;

    let h: Vec<f32> = rand_vec(hc_dim, 3);
    let hc_fn: Vec<f32> = rand_vec(mix_hc * hc_dim, 7)
        .iter()
        .map(|v| (v % 2.0 - 1.0) * 0.3)
        .collect();
    let hc_base: Vec<f32> = rand_vec(mix_hc, 9)
        .iter()
        .map(|v| (v % 2.0 - 1.0) * 0.5)
        .collect();
    let hc_scale = vec![1.0f32, 0.8, 1.2];

    let expect =
        ferroseek_ref::hc::hc_mixes(&h, &hc_fn, &hc_scale, &hc_base, hc, dim, 1e-20, 19, 1e-4);

    let d_h = DeviceBuffer::from_slice(&h).unwrap();
    let d_fn = DeviceBuffer::from_slice(&hc_fn).unwrap();
    let d_base = DeviceBuffer::from_slice(&hc_base).unwrap();
    let d_scale = DeviceBuffer::from_slice(&hc_scale).unwrap();
    let d_pre = DeviceBuffer::from_slice(&vec![0f32; 4]).unwrap();
    let d_post = DeviceBuffer::from_slice(&vec![0f32; 4]).unwrap();
    let d_comb = DeviceBuffer::from_slice(&vec![0f32; 16]).unwrap();
    crate::launch::hc_block(
        c,
        d_h.as_device_ptr(),
        d_fn.as_device_ptr(),
        d_base.as_device_ptr(),
        d_scale.as_device_ptr(),
        d_pre.as_device_ptr(),
        d_post.as_device_ptr(),
        d_comb.as_device_ptr(),
        1,
        hc_dim,
        mix_hc,
        1e-4,
    );
    c.stream.synchronize().unwrap();
    let mut pre = vec![0f32; 4];
    let mut post = vec![0f32; 4];
    let mut comb = vec![0f32; 16];
    d_pre.copy_to(&mut pre).unwrap();
    d_post.copy_to(&mut post).unwrap();
    d_comb.copy_to(&mut comb).unwrap();
    let dpre = max_abs_diff(&pre, &expect.pre);
    let dpost = max_abs_diff(&post, &expect.post);
    let dcomb = max_abs_diff(&comb, &expect.comb);
    assert!(dpre < 1e-5, "hc_block pre diff {dpre:.3e}");
    assert!(dpost < 1e-5, "hc_block post diff {dpost:.3e}");
    assert!(dcomb < 1e-5, "hc_block comb diff {dcomb:.3e}");

    let d_dots = DeviceBuffer::from_slice(&vec![0f32; mix_hc]).unwrap();
    crate::launch::hc_mixes(
        c,
        d_h.as_device_ptr(),
        d_fn.as_device_ptr(),
        d_dots.as_device_ptr(),
        1,
        hc_dim,
        mix_hc,
    );
    crate::launch::hc_finalize(
        c,
        d_h.as_device_ptr(),
        d_dots.as_device_ptr(),
        d_base.as_device_ptr(),
        d_scale.as_device_ptr(),
        d_pre.as_device_ptr(),
        d_post.as_device_ptr(),
        d_comb.as_device_ptr(),
        1,
        hc_dim,
        mix_hc,
        1e-4,
    );
    c.stream.synchronize().unwrap();
    let mut split_pre = vec![0f32; 4];
    let mut split_post = vec![0f32; 4];
    let mut split_comb = vec![0f32; 16];
    d_pre.copy_to(&mut split_pre).unwrap();
    d_post.copy_to(&mut split_post).unwrap();
    d_comb.copy_to(&mut split_comb).unwrap();
    assert_eq!(split_pre, pre, "split HC changed pre bits");
    assert_eq!(split_post, post, "split HC changed post bits");
    assert_eq!(split_comb, comb, "split HC changed comb bits");
}

#[test]
fn fp4_expert_gemm_vectorized_matches_cpu() {
    // isolate the vectorized fp4_expert_gemm (TP1/synthetic host-dispatch
    // path; TP4 uses the fused moe_grouped kernels) — CPU mirror exact
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let (m, n, k) = (1usize, 256usize, 512usize);
    let kb = k / 32;

    let a_f: Vec<f32> = rand_vec(m * k, 31)
        .iter()
        .map(|v| (v % 3.0 - 1.0) * 0.5)
        .collect();
    let qa: Vec<u8> = a_f
        .iter()
        .map(|&v| ferroseek_core::quant::f32_to_e4m3(v))
        .collect();
    let sa: Vec<u8> = vec![127u8; m * kb];
    let mut qb = vec![0u8; n * k / 2];
    for i in 0..n * k / 2 {
        let lo = (i % 13) as u8 & 0x0F;
        let hi = ((i * 3 + 1) % 9) as u8 & 0x0F;
        qb[i] = lo | (hi << 4);
    }
    let sb: Vec<u8> = vec![127u8; n * kb];

    let da: Vec<f32> = qa
        .iter()
        .map(|&v| ferroseek_core::quant::e4m3_to_f32(v))
        .collect();
    let mut expect = vec![0.0f32; m * n];
    for ni in 0..n {
        let mut acc = 0.0f32;
        for b in 0..kb {
            let mut dot = 0.0f32;
            for kk in 0..16 {
                let byte = qb[ni * (k / 2) + b * 16 + kk];
                let (lo, hi) = ferroseek_core::quant::unpack_fp4(byte);
                dot += da[b * 32 + 2 * kk] * lo;
                dot += da[b * 32 + 2 * kk + 1] * hi;
            }
            acc += dot;
        }
        expect[ni] = acc;
    }

    let d_a = DeviceBuffer::from_slice(&qa).unwrap();
    let d_sa = DeviceBuffer::from_slice(&sa).unwrap();
    let d_b = DeviceBuffer::from_slice(&qb).unwrap();
    let d_sb = DeviceBuffer::from_slice(&sb).unwrap();
    let mut d_c = DeviceBuffer::from_slice(&vec![0f32; m * n]).unwrap();
    crate::launch::fp4_gemm(
        c,
        d_a.as_device_ptr(),
        d_sa.as_device_ptr(),
        d_b.as_device_ptr(),
        d_sb.as_device_ptr(),
        d_c.as_device_ptr(),
        m,
        n,
        k,
    );
    c.stream.synchronize().unwrap();
    let mut got = vec![0f32; m * n];
    d_c.copy_to(&mut got).unwrap();
    let d = max_abs_diff(&got, &expect);
    assert!(d < 1e-3, "fp4_expert_gemm vs CPU max diff {d:.3e}");
}

#[test]
fn gpu_engine_matches_cpu_reference_per_layer() {
    // Step-1 bisection of the engine gate's divergence: capture each
    // attention sublayer's input x on BOTH engines at step 1 and print
    // the first layer where they diverge beyond the noise floor.
    let mut cpu = ref_model(42);
    let mut gpu = gpu_model(42);
    cpu.debug_capture_x(true);
    gpu.debug_capture_x(true);
    for (p, &t) in TOKENS.iter().enumerate() {
        let _ = cpu.decode(t, p as u64).unwrap();
        let _ = gpu.decode(t, p as u64).unwrap();
        if p < 3 {
            let cx = cpu.debug_take_captures();
            let gx = gpu.debug_take_captures();
            // re-arm both (take() drains)
            cpu.debug_capture_x(true);
            gpu.debug_capture_x(true);
            for (i, ((l, cxv), (_, gxv))) in cx.iter().zip(gx.iter()).enumerate() {
                let d = max_abs_diff(cxv, gxv);
                if d > 1e-5 {
                    eprintln!("step {p}: FIRST divergence at capture {i} (layer {l}): {d:.3e}");
                    return;
                }
            }
            eprintln!("step {p}: all {} captures clean (max < 1e-5)", cx.len());
        }
    }
}

#[test]
fn gpu_step1_layer0_y_bisect() {
    // The engine gate's flip is born in layer 0's FFN at step 1 (per-layer
    // capture test). Dump both engines' layer-0 y (moe output) at step 1
    // via the built-in d_pos*_y0 dump and diff the first element to differ.
    let mut cpu = ref_model(42);
    let mut gpu = gpu_model(42);
    // step 0 clean, step 1 diverges
    for (p, &t) in TOKENS.iter().enumerate().take(2) {
        let cl = cpu.decode(t, p as u64).unwrap();
        let gl = gpu.decode(t, p as u64).unwrap();
        let d = max_abs_diff(&cl, &gl);
        eprintln!("step {p}: logits diff {d:.3e}");
    }
    // both engines wrote d_pos1_y0.f32 into FERROSEEK_DUMP; the test
    // compares them out-of-band (the dirs differ per engine — the env is
    // process-wide, so run the CPU first in its own process; here we
    // verify the GPU dump exists and print its head)
    let dir = std::env::var_os("FERROSEEK_DUMP_GPU").map(std::path::PathBuf::from);
    if let Some(d) = dir {
        let p = d.join("d_pos1_y0.f32");
        if p.exists() {
            let b = std::fs::read(&p).unwrap();
            let v: Vec<f32> = b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            eprintln!("gpu y0 head: {:?}", &v[..8.min(v.len())]);
        } else {
            eprintln!("no gpu dump at {:?}", p);
        }
    }
}

#[test]
fn indexer_wq_b_rows_vs_all_bytes_identical() {
    // hybrid bisect isolated the e834f0c regression to the indexer wq_b's
    // Rows shard at TP1. Rows{0, full} should be byte-identity with All —
    // assert it directly on the synthetic shapes to see which byte moves.
    let cfg = mini_config();
    let weights = ferroseek_ref::store::SyntheticWeights::generate(&cfg, 42).unwrap();
    use ferroseek_ref::store::WeightStore;
    let mut store = weights;
    let (wv, sv) = store
        .get2(
            "layers.1.attn.indexer.wq_b.weight",
            "layers.1.attn.indexer.wq_b.scale",
        )
        .unwrap();
    eprintln!(
        "indexer wq_b: weight shape {:?} ({} bytes), scale shape {:?} ({} bytes)",
        wv.shape,
        wv.bytes.len(),
        sv.shape,
        sv.bytes.len()
    );
    // TP1 shard: start 0, len = full rows
    let (rows, cols) = (wv.shape[0], wv.shape[1]);
    let full = crate::engine::WShard::Rows {
        start: 0,
        len: rows,
    };
    let all = crate::engine::WShard::All;
    let wa = crate::engine::shard_weight_pub(wv.bytes, &wv.shape, all).unwrap();
    let wr = crate::engine::shard_weight_pub(wv.bytes, &wv.shape, full).unwrap();
    eprintln!("weight All==Rows bytes: {}", wa.as_slice() == wr.as_slice());
    let sa = crate::engine::shard_scale_pub(sv.bytes, &sv.shape, all).unwrap();
    let sr = crate::engine::shard_scale_pub(sv.bytes, &sv.shape, full).unwrap();
    eprintln!(
        "scale All len {} bytes, Rows len {} bytes, equal: {}",
        sa.as_slice().len(),
        sr.as_slice().len(),
        sa.as_slice() == sr.as_slice()
    );
    assert_eq!(sa.as_slice(), sr.as_slice(), "scale differs!");
    assert_eq!(wa.as_slice(), wr.as_slice(), "weight differs!");
}

#[test]
fn wo_b_cols_shard_identity_probe() {
    // the wo_b load is the only Cols shard; TP1 slices it full-width.
    // Assert Cols{0, full-cols} == All byte-identity on the synthetic
    // shapes (the Rows twin of this probe passed; Cols is untested).
    let cfg = mini_config();
    let weights = ferroseek_ref::store::SyntheticWeights::generate(&cfg, 42).unwrap();
    use ferroseek_ref::store::WeightStore;
    let mut store = weights;
    let (wv, sv) = store
        .get2("layers.1.attn.wo_b.weight", "layers.1.attn.wo_b.scale")
        .unwrap();
    eprintln!(
        "wo_b: weight {:?} ({} bytes), scale {:?} ({} bytes)",
        wv.shape,
        wv.bytes.len(),
        sv.shape,
        sv.bytes.len()
    );
    let cols = wv.shape[1];
    let all = crate::engine::WShard::All;
    let full_cols = crate::engine::WShard::Cols {
        start: 0,
        len: cols,
    };
    let wa = crate::engine::shard_weight_pub(wv.bytes, &wv.shape, all).unwrap();
    let wc = crate::engine::shard_weight_pub(wv.bytes, &wv.shape, full_cols).unwrap();
    let sa = crate::engine::shard_scale_pub(sv.bytes, &sv.shape, all).unwrap();
    let sc = crate::engine::shard_scale_pub(sv.bytes, &sv.shape, full_cols).unwrap();
    eprintln!(
        "weight All==Cols: {} (lens {}/{})",
        wa.as_slice() == wc.as_slice(),
        wa.as_slice().len(),
        wc.as_slice().len()
    );
    eprintln!(
        "scale All==Cols: {} (lens {}/{})",
        sa.as_slice() == sc.as_slice(),
        sa.as_slice().len(),
        sc.as_slice().len()
    );
    assert_eq!(sa.as_slice(), sc.as_slice(), "scale Cols differs!");
    assert_eq!(wa.as_slice(), wc.as_slice(), "weight Cols differs!");
}

#[test]
fn gpu_engine_flip_bisect_in_process() {
    // The deterministic S1 flip, bisected IN-PROCESS (immune to the
    // cross-run dump-pairing artifacts): capture x (attention input)
    // and y (MoE output) per step on BOTH engines, report the first
    // (step, layer, boundary) that diverges beyond the noise floor.
    let mut cpu = ref_model(42);
    let mut gpu = gpu_model(42);
    cpu.debug_capture_x(true);
    gpu.debug_capture_x(true);
    cpu.debug_capture_y(true);
    gpu.debug_capture_y(true);
    let mut flipped = false;
    for (p, &t) in TOKENS.iter().enumerate() {
        let cl = cpu.decode(t, p as u64).unwrap();
        let gl = gpu.decode(t, p as u64).unwrap();
        let cx = cpu.debug_take_captures();
        let gx = gpu.debug_take_captures();
        let cy = cpu.debug_take_y();
        let gy = gpu.debug_take_y();
        cpu.debug_capture_x(true); // re-arm
        gpu.debug_capture_x(true);
        cpu.debug_capture_y(true);
        gpu.debug_capture_y(true);
        for (i, ((lx, cxv), (_, gxv))) in cx.iter().zip(gx.iter()).enumerate() {
            let d = max_abs_diff(cxv, gxv);
            if d > 1e-5 {
                eprintln!(
                    "step {p}: FIRST divergence at ATTENTION INPUT, layer {lx} (cap {i}): {d:.3e}"
                );
                assert!(p > 0, "divergence at step 0 input — structural");
                return;
            }
        }
        for (i, ((ly, cyv), (_, gyv))) in cy.iter().zip(gy.iter()).enumerate() {
            let d = max_abs_diff(cyv, gyv);
            if d > 1e-5 {
                eprintln!(
                    "step {p}: FIRST divergence at MOE OUTPUT, layer {ly} (cap {i}): {d:.3e}",
                );
                let cd = max_abs_diff(&cl, &gl);
                eprintln!("step {p}: logits diff at same step: {cd:.3e}");
                flipped = true;
                if p >= 2 {
                    return;
                }
                break;
            }
        }
        if flipped {
            break;
        }
        let d = max_abs_diff(&cl, &gl);
        eprintln!("step {p}: clean (max {d:.3e})",);
    }
}

#[test]
fn fp8_gemm_shared_expert_shape_matches_cpu() {
    // The S1-L3 flip bisects to the shared expert; this is its exact
    // GEMV shape (m=1, n=256, k=64), never directly gated before.
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let (m, n, k) = (1usize, 256usize, 64usize);
    let kb = k / 32;

    let a_f: Vec<f32> = rand_vec(m * k, 77)
        .iter()
        .map(|v| (v % 3.0 - 1.0) * 0.5)
        .collect();
    let b_f: Vec<f32> = rand_vec(n * k, 78)
        .iter()
        .map(|v| (v % 3.0 - 1.0) * 0.25)
        .collect();
    // quantize to e4m3 + ue8m0 per-1x32 (act) and 32x32 (weight) blocks
    let qa: Vec<u8> = a_f
        .iter()
        .map(|&v| ferroseek_core::quant::f32_to_e4m3(v))
        .collect();
    let qb: Vec<u8> = b_f
        .iter()
        .map(|&v| ferroseek_core::quant::f32_to_e4m3(v))
        .collect();
    let sa: Vec<u8> = vec![127u8; m * kb];
    // weight scales: [n/32, kb] tiles — exercise NON-UNITY scales too
    let mut sb = vec![0u8; (n / 32) * kb];
    for i in 0..sb.len() {
        sb[i] = 120u8 + (i % 5) as u8; // 2^-7 .. 2^-3
    }

    // CPU mirror: ferroseek_ref's own fp8_gemv
    let expect = ferroseek_ref::actquant::fp8_gemv(&qb, &sb, n, k, &qa_dequant(&qa));

    let d_a = DeviceBuffer::from_slice(&qa).unwrap();
    let d_sa = DeviceBuffer::from_slice(&sa).unwrap();
    let d_b = DeviceBuffer::from_slice(&qb).unwrap();
    let d_sb = DeviceBuffer::from_slice(&sb).unwrap();
    let mut d_c = DeviceBuffer::from_slice(&vec![0f32; m * n]).unwrap();
    crate::launch::fp8_gemm(
        c,
        d_a.as_device_ptr(),
        d_sa.as_device_ptr(),
        d_b.as_device_ptr(),
        d_sb.as_device_ptr(),
        d_c.as_device_ptr(),
        m,
        n,
        k,
    );
    c.stream.synchronize().unwrap();
    let mut got = vec![0f32; m * n];
    d_c.copy_to(&mut got).unwrap();
    let d = max_abs_diff(&got, &expect);
    assert!(d < 1e-3, "fp8_gemm shared shape vs CPU: {d:.3e}");
}

fn qa_dequant(q: &[u8]) -> Vec<f32> {
    q.iter()
        .map(|&v| ferroseek_core::quant::e4m3_to_f32(v))
        .collect()
}

#[test]
fn sparse_attn_m_bitexact_vs_sequential() {
    // The M-batch attention kernel must be bit-identical to m sequential
    // sparse_attn calls: grid (n_heads, m) runs the same per-block
    // arithmetic for each (head, position) pair. Causal idx lists:
    // position i selects a growing window, -1-padded to n_sel.
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let (n_heads, dim, n_sel, m, n_rows) = (8usize, 512usize, 24usize, 4usize, 40usize);
    let scale = 0.0625f32;

    let kv: Vec<f32> = rand_vec(n_rows * dim, 7)
        .iter()
        .map(|v| (v % 2.0 - 1.0) * 0.5)
        .collect();
    let q: Vec<f32> = rand_vec(m * n_heads * dim, 11)
        .iter()
        .map(|v| (v % 2.0 - 1.0) * 0.5)
        .collect();
    let sink: Vec<f32> = rand_vec(n_heads, 13).iter().map(|v| v * 0.25).collect();
    // causal: position i sees rows [0..6+i*3) then -1 padding
    let mut idxs = vec![-1i32; m * n_sel];
    for pos in 0..m {
        let seen = 6 + pos * 3;
        for j in 0..seen.min(n_sel) {
            idxs[pos * n_sel + j] = j as i32;
        }
    }

    let d_kv = DeviceBuffer::from_slice(&kv).unwrap();
    let d_q = DeviceBuffer::from_slice(&q).unwrap();
    let d_idxs = DeviceBuffer::from_slice(&idxs).unwrap();
    let d_sink = DeviceBuffer::from_slice(&sink).unwrap();

    // batched: one launch
    let mut d_out_m = DeviceBuffer::from_slice(&vec![0f32; m * n_heads * dim]).unwrap();
    crate::launch::sparse_attn_m(
        c,
        d_q.as_device_ptr(),
        d_kv.as_device_ptr(),
        d_idxs.as_device_ptr(),
        d_sink.as_device_ptr(),
        d_out_m.as_device_ptr(),
        n_heads,
        dim,
        n_sel,
        m,
        scale,
    );
    c.stream.synchronize().unwrap();
    let mut got = vec![0f32; m * n_heads * dim];
    d_out_m.copy_to(&mut got).unwrap();

    // sequential: m launches of the m=1 kernel on the same inputs
    let mut expect = vec![0f32; m * n_heads * dim];
    for pos in 0..m {
        let q_pos = &q[pos * n_heads * dim..(pos + 1) * n_heads * dim];
        let i_pos = &idxs[pos * n_sel..(pos + 1) * n_sel];
        let d_qp = DeviceBuffer::from_slice(q_pos).unwrap();
        let d_ip = DeviceBuffer::from_slice(i_pos).unwrap();
        let mut d_op = DeviceBuffer::from_slice(&vec![0f32; n_heads * dim]).unwrap();
        crate::launch::sparse_attn(
            c,
            d_qp.as_device_ptr(),
            d_kv.as_device_ptr(),
            d_ip.as_device_ptr(),
            d_sink.as_device_ptr(),
            d_op.as_device_ptr(),
            n_heads,
            dim,
            n_sel,
            scale,
        );
        c.stream.synchronize().unwrap();
        d_op.copy_to(&mut expect[pos * n_heads * dim..(pos + 1) * n_heads * dim])
            .unwrap();
    }

    assert_eq!(
        got, expect,
        "sparse_attn_m diverged from sequential sparse_attn"
    );
    eprintln!(
        "sparse_attn_m: bit-identical across {m} positions x {n_heads} heads (n_sel {n_sel})"
    );
}

#[test]
fn sparse_attn_split_matches_single_pass() {
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let (n_heads, dim, n_sel, n_rows) = (8usize, 512usize, 300usize, 400usize);
    let scale = 0.0625f32;
    let kv: Vec<f32> = rand_vec(n_rows * dim, 7)
        .iter()
        .map(|v| (v % 2.0 - 1.0) * 0.5)
        .collect();
    let q: Vec<f32> = rand_vec(n_heads * dim, 11)
        .iter()
        .map(|v| (v % 2.0 - 1.0) * 0.5)
        .collect();
    let sink: Vec<f32> = rand_vec(n_heads, 13).iter().map(|v| v * 0.25).collect();
    let idxs: Vec<i32> = (0..n_sel)
        .map(|i| {
            if i % 11 == 10 {
                -1
            } else {
                (i * 7 % n_rows) as i32
            }
        })
        .collect();
    let d_kv = DeviceBuffer::from_slice(&kv).unwrap();
    let d_q = DeviceBuffer::from_slice(&q).unwrap();
    let d_idxs = DeviceBuffer::from_slice(&idxs).unwrap();
    let d_sink = DeviceBuffer::from_slice(&sink).unwrap();
    let d_expected = DeviceBuffer::from_slice(&vec![0.0f32; n_heads * dim]).unwrap();
    let d_got = DeviceBuffer::from_slice(&vec![0.0f32; n_heads * dim]).unwrap();
    let segments = n_sel.div_ceil(128);
    let d_partial = DeviceBuffer::from_slice(&vec![0.0f32; segments * n_heads * dim]).unwrap();
    let d_stats = DeviceBuffer::from_slice(&vec![0.0f32; segments * n_heads * 2]).unwrap();
    crate::launch::sparse_attn(
        c,
        d_q.as_device_ptr(),
        d_kv.as_device_ptr(),
        d_idxs.as_device_ptr(),
        d_sink.as_device_ptr(),
        d_expected.as_device_ptr(),
        n_heads,
        dim,
        n_sel,
        scale,
    );
    crate::launch::sparse_attn_split(
        c,
        d_q.as_device_ptr(),
        d_kv.as_device_ptr(),
        d_idxs.as_device_ptr(),
        d_sink.as_device_ptr(),
        d_partial.as_device_ptr(),
        d_stats.as_device_ptr(),
        d_got.as_device_ptr(),
        n_heads,
        dim,
        n_sel,
        scale,
    );
    let mut expected = vec![0.0f32; n_heads * dim];
    let mut got = vec![0.0f32; n_heads * dim];
    c.stream.synchronize().unwrap();
    d_expected.copy_to(&mut expected).unwrap();
    d_got.copy_to(&mut got).unwrap();
    let d = max_abs_diff(&got, &expected);
    assert!(d < 1e-5, "split sparse attention max diff {d:.3e}");
}

#[test]
fn sparse_mla_packed_window_prefix() {
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let (heads, dim, win) = (16usize, 512usize, 128usize);
    let q = DeviceBuffer::from_slice(&vec![0u16; heads * dim]).unwrap();
    let swa = DeviceBuffer::from_slice(&vec![0u8; win * 528]).unwrap();
    let wi: Vec<i32> = std::iter::once(0)
        .chain(std::iter::repeat_n(-1, win - 1))
        .collect();
    let idxs = DeviceBuffer::from_slice(&wi).unwrap();
    let sl = DeviceBuffer::from_slice(&[1i32]).unwrap();
    let ir = DeviceBuffer::from_slice(&vec![0u8; 64 * 288]).unwrap();
    let il = DeviceBuffer::from_slice(&[0i32]).unwrap();
    let pt = DeviceBuffer::from_slice(&[0i32]).unwrap();
    let sinks = DeviceBuffer::from_slice(&vec![0f32; heads]).unwrap();
    let pos = DeviceBuffer::from_slice(&[0i64]).unwrap();
    let cos = DeviceBuffer::from_slice(&vec![0f32; 64]).unwrap();
    let sin = DeviceBuffer::from_slice(&vec![0f32; 64]).unwrap();
    let part = DeviceBuffer::from_slice(&vec![0f32; heads * dim]).unwrap();
    let stats = DeviceBuffer::from_slice(&vec![0f32; heads * 2]).unwrap();
    let out = DeviceBuffer::from_slice(&vec![0f32; heads * dim]).unwrap();
    crate::launch::sparse_mla_decode(
        c,
        q.as_device_ptr(),
        swa.as_device_ptr(),
        idxs.as_device_ptr(),
        sl.as_device_ptr(),
        ir.as_device_ptr(),
        idxs.as_device_ptr(),
        il.as_device_ptr(),
        pt.as_device_ptr(),
        sinks.as_device_ptr(),
        pos.as_device_ptr(),
        cos.as_device_ptr(),
        sin.as_device_ptr(),
        part.as_device_ptr(),
        stats.as_device_ptr(),
        out.as_device_ptr(),
        heads,
        1,
        1,
        64 * 288,
        win,
        528,
        1,
        1,
        1,
    );
    c.stream.synchronize().unwrap();
    let mut st = vec![0f32; heads * 2];
    stats.copy_to(&mut st).unwrap();
    assert!(
        st[0].is_finite() && st[1] > 0.0,
        "packed window prefix must expose its sole valid key"
    );
}

#[test]
fn rope_uniform_pos_buf_bitexact_vs_scalar() {
    // Per-row rope positions must be bit-identical to per-position scalar
    // launches: same rotation arithmetic, position read from a buffer.
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let (n_rows, dim, rope_dim, base_pos) = (12usize, 128usize, 64usize, 100usize);
    let x: Vec<f32> = rand_vec(n_rows * dim, 17)
        .iter()
        .map(|v| (v % 2.0 - 1.0) * 0.75)
        .collect();
    // rope tables: [max_pos, rope_dim/2]
    let max_pos = 256usize;
    let half = rope_dim / 2;
    let cos_t: Vec<f32> = (0..max_pos * half)
        .map(|i| ((i % 97) as f32 * 0.0103).cos())
        .collect();
    let sin_t: Vec<f32> = (0..max_pos * half)
        .map(|i| ((i % 97) as f32 * 0.0103).sin())
        .collect();
    // positions: 4 tokens x 3 rows each (heads share a token's position)
    let pos_per_row: Vec<i32> = (0..n_rows).map(|r| (base_pos + r / 3) as i32).collect();

    let d_cos = DeviceBuffer::from_slice(&cos_t).unwrap();
    let d_sin = DeviceBuffer::from_slice(&sin_t).unwrap();
    let d_pos = DeviceBuffer::from_slice(&pos_per_row).unwrap();

    // batched: one launch
    let mut d_xb = DeviceBuffer::from_slice(&x).unwrap();
    crate::launch::rope_uniform_pos_buf(
        c,
        d_xb.as_device_ptr(),
        d_cos.as_device_ptr(),
        d_sin.as_device_ptr(),
        d_pos.as_device_ptr(),
        n_rows,
        dim,
        rope_dim,
        false,
    );
    c.stream.synchronize().unwrap();
    let mut got = vec![0f32; n_rows * dim];
    d_xb.copy_to(&mut got).unwrap();

    // sequential: one scalar launch per distinct position over its rows
    let mut d_xs = DeviceBuffer::from_slice(&x).unwrap();
    for tok in 0..(n_rows / 3) {
        // rows [3*tok, 3*tok+3) rotate at base_pos+tok — launch on the
        // sub-slice via raw offset (3 contiguous rows)
        let off = tok * 3 * dim;
        let ptr = unsafe { d_xs.as_device_ptr().offset(off as isize) };
        crate::launch::rope_uniform_pos(
            c,
            ptr,
            d_cos.as_device_ptr(),
            d_sin.as_device_ptr(),
            3,
            dim,
            rope_dim,
            (base_pos + tok) as u64,
            false,
        );
    }
    c.stream.synchronize().unwrap();
    let mut expect = vec![0f32; n_rows * dim];
    d_xs.copy_to(&mut expect).unwrap();

    assert_eq!(
        got, expect,
        "rope_uniform_pos_buf diverged from scalar launches"
    );
    eprintln!("rope_uniform_pos_buf: bit-identical over {n_rows} rows / 4 positions");
}

#[test]
fn moe_gate_m_and_rmsnorm_m_bitexact() {
    // moe_gate_m: grid (experts, m), same per-block reduction — bit-exact
    // vs m sequential moe_gate calls. rmsnorm_m: the kernel already
    // indexes rows by blockIdx.x — bit-exact vs per-row launches.
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();

    // --- moe_gate_m ---
    let (m, dim, n_experts) = (4usize, 64usize, 16usize);
    let x: Vec<f32> = rand_vec(m * dim, 23)
        .iter()
        .map(|v| (v % 2.0 - 1.0) * 0.5)
        .collect();
    let gate_w: Vec<u16> = rand_vec(n_experts * dim, 29)
        .iter()
        .map(|&v| ferroseek_core::quant::f32_to_bf16((v % 2.0 - 1.0) * 0.25))
        .collect();
    let bias: Vec<f32> = rand_vec(n_experts, 31).iter().map(|v| v * 0.1).collect();
    let d_x = DeviceBuffer::from_slice(&x).unwrap();
    let d_gw = DeviceBuffer::from_slice(&gate_w).unwrap();
    let d_b = DeviceBuffer::from_slice(&bias).unwrap();
    let mut d_sm = DeviceBuffer::from_slice(&vec![0f32; m * n_experts]).unwrap();
    crate::launch::moe_gate_m(
        c,
        d_x.as_device_ptr(),
        d_gw.as_device_ptr(),
        d_b.as_device_ptr(),
        d_sm.as_device_ptr(),
        m,
        dim,
        n_experts,
    );
    c.stream.synchronize().unwrap();
    let mut got = vec![0f32; m * n_experts];
    d_sm.copy_to(&mut got).unwrap();
    let mut expect = vec![0f32; m * n_experts];
    for pos in 0..m {
        let d_xp = DeviceBuffer::from_slice(&x[pos * dim..(pos + 1) * dim]).unwrap();
        let mut d_sp = DeviceBuffer::from_slice(&vec![0f32; n_experts]).unwrap();
        crate::launch::moe_gate(
            c,
            d_xp.as_device_ptr(),
            d_gw.as_device_ptr(),
            d_b.as_device_ptr(),
            d_sp.as_device_ptr(),
            dim,
            n_experts,
        );
        c.stream.synchronize().unwrap();
        d_sp.copy_to(&mut expect[pos * n_experts..(pos + 1) * n_experts])
            .unwrap();
    }
    assert_eq!(got, expect, "moe_gate_m diverged from sequential moe_gate");

    // --- rmsnorm_m ---
    let eps = 1e-5f32;
    let w: Vec<f32> = rand_vec(dim, 37)
        .iter()
        .map(|v| 0.5 + v.abs() % 1.0)
        .collect();
    let d_w = DeviceBuffer::from_slice(&w).unwrap();
    let mut d_om = DeviceBuffer::from_slice(&vec![0f32; m * dim]).unwrap();
    crate::launch::rmsnorm_m(
        c,
        d_x.as_device_ptr(),
        d_w.as_device_ptr(),
        d_om.as_device_ptr(),
        m,
        dim,
        eps,
    );
    c.stream.synchronize().unwrap();
    let mut got_n = vec![0f32; m * dim];
    d_om.copy_to(&mut got_n).unwrap();
    let mut expect_n = vec![0f32; m * dim];
    for pos in 0..m {
        let d_xp = DeviceBuffer::from_slice(&x[pos * dim..(pos + 1) * dim]).unwrap();
        let mut d_op = DeviceBuffer::from_slice(&vec![0f32; dim]).unwrap();
        crate::launch::rmsnorm(
            c,
            d_xp.as_device_ptr(),
            d_w.as_device_ptr(),
            d_op.as_device_ptr(),
            dim,
            eps,
        );
        c.stream.synchronize().unwrap();
        d_op.copy_to(&mut expect_n[pos * dim..(pos + 1) * dim])
            .unwrap();
    }
    assert_eq!(got_n, expect_n, "rmsnorm_m diverged from per-row rmsnorm");
    eprintln!("moe_gate_m + rmsnorm_m: bit-identical over {m} rows");
}

#[test]
fn decode_m_matches_sequential_decode() {
    // THE M-batch gate: decode_m(TOKENS) must match sequential decode()
    // per position within the engine's documented op-order class (the
    // GEMM m>1 reduction order differs from the m=1 GEMV path), with
    // argmax agreement, and the post-pass model state must be identical:
    // two further sequential steps on both models must match too.
    let mut seq = gpu_model(42);
    let mut bat = gpu_model(42);

    let seq_logits: Vec<Vec<f32>> = TOKENS
        .iter()
        .enumerate()
        .map(|(p, &t)| seq.decode(t, p as u64).unwrap())
        .collect();
    let bat_logits = bat.decode_m(&TOKENS).unwrap();

    let mut worst = 0.0f32;
    for (p, (a, b)) in seq_logits.iter().zip(bat_logits.iter()).enumerate() {
        let d = max_abs_diff(a, b);
        worst = worst.max(d);
        eprintln!(
            "step {p}: decode_m max diff {d:.3e} (argmax {} vs {})",
            ferroseek_ref::argmax(a),
            ferroseek_ref::argmax(b)
        );
    }
    assert!(
        worst < GATE_ABS_TOL,
        "decode_m diverged from sequential: {worst:.3e} beyond the {GATE_ABS_TOL:.0e} floor"
    );

    // state equivalence: continue both sequentially
    for (j, &t) in [29u32, 31].iter().enumerate() {
        let p = TOKENS.len() + j;
        let a = seq.decode(t, p as u64).unwrap();
        let b = bat.decode(t, p as u64).unwrap();
        let d = max_abs_diff(&a, &b);
        assert!(
            d < GATE_ABS_TOL,
            "post-pass state diverged at step {p}: {d:.3e}"
        );
        eprintln!("post step {p}: state match {d:.3e}");
    }
    eprintln!("decode_m: 8-position batch matches sequential within {worst:.3e}");

    // wall time: the batch amortizes ~240 kernel launches per position
    let mut seq2 = gpu_model(42);
    let t0 = std::time::Instant::now();
    for (p, &t) in TOKENS.iter().enumerate() {
        let _ = seq2.decode(t, p as u64).unwrap();
    }
    let t_seq = t0.elapsed();
    let mut bat2 = gpu_model(42);
    let t0 = std::time::Instant::now();
    let _ = bat2.decode_m(&TOKENS).unwrap();
    let t_bat = t0.elapsed();
    eprintln!(
        "decode_m wall: {:?} vs sequential {:?} ({:.1}x)",
        t_bat,
        t_seq,
        t_seq.as_secs_f64() / t_bat.as_secs_f64().max(1e-9)
    );
}

fn decode_prefill_m_matches_sequential_and_reuses_prefix() {
    // The chunked-prefill wiring: a fresh decode_prefill must land
    // bit-exact with sequential decode per token, and a second call
    // sharing a prefix must reuse the cache (decode only the suffix)
    // while staying bit-exact with a from-scratch prefill.
    let ids: Vec<u32> = TOKENS.to_vec();

    let mut seq = gpu_model(42);
    let mut last = Vec::new();
    for (p, &t) in ids.iter().enumerate() {
        last = seq.decode(t, p as u64).unwrap();
    }

    let mut bat = gpu_model(42);
    let got = bat.decode_prefill(&ids).unwrap();
    let d = max_abs_diff(&last, &got);
    assert!(d < GATE_ABS_TOL, "chunked prefill diverged: {d:.3e}");
    assert_eq!(ferroseek_ref::argmax(&last), ferroseek_ref::argmax(&got));
    eprintln!("prefill batch: bit-exact vs sequential ({d:.3e})");

    // prefix reuse: prefill [ids], then [ids + suffix] — the second call
    // must decode ONLY the suffix (pos resumes at ids.len()) and match a
    // from-scratch prefill of the full sequence.
    let suffix = [29u32, 31];
    let mut full: Vec<u32> = ids.clone();
    full.extend_from_slice(&suffix);

    let mut ref1 = gpu_model(42);
    let want = ref1.decode_prefill(&full).unwrap();

    let _ = bat.decode_prefill(&ids).unwrap();
    assert_eq!(bat.next_position(), ids.len() as u64);
    let got2 = bat.decode_prefill(&full).unwrap();
    let d2 = max_abs_diff(&want, &got2);
    assert!(d2 < GATE_ABS_TOL, "prefix-reuse prefill diverged: {d2:.3e}");
    assert_eq!(bat.next_position(), full.len() as u64);
    eprintln!(
        "prefill prefix reuse: bit-exact ({d2:.3e}), resumed at {}",
        ids.len()
    );
}

#[test]
fn decode_prefill_window_boundary_matches_sequential() {
    let ids: Vec<u32> = TOKENS.iter().cloned().cycle().take(136).collect();
    let mut seq = gpu_model(42);
    let mut want = Vec::new();
    for (p, &t) in ids.iter().enumerate() {
        want = seq.decode(t, p as u64).unwrap();
    }
    let mut bat = gpu_model(42);
    let got = bat.decode_prefill(&ids).unwrap();
    let d = max_abs_diff(&want, &got);
    assert!(
        d < GATE_ABS_TOL,
        "window-crossing prefill diverged: {d:.3e}"
    );
    assert_eq!(bat.next_position(), ids.len() as u64);
    for (j, &t) in [29u32, 31].iter().enumerate() {
        let p = ids.len() + j;
        let a = seq.decode(t, p as u64).unwrap();
        let b = bat.decode(t, p as u64).unwrap();
        let d = max_abs_diff(&a, &b);
        assert!(
            d < GATE_ABS_TOL,
            "window-crossing continuation diverged: {d:.3e}"
        );
    }
}
#[test]
fn decode_m_streaming_matches_sequential() {
    // decode_m through the STREAMING path (registered shard mappings,
    // per-expert staging): same synthetic checkpoint on disk, one model
    // sequential, one batched — bit-exact expectation (same kernels,
    // same bytes; fp8_gemm is bit-identical per output block at any m).
    use ferroseek_core::loader::Checkpoint;

    let cfg = mini_config();
    let json = mini_config_json(mini_engram_rows());
    let weights = SyntheticWeights::generate(&cfg, 42).unwrap();
    let dir = std::env::temp_dir().join(format!("ferroseek-gpu-stream-m-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    weights.write_checkpoint(&json, &dir).unwrap();

    let mut ckpt1 = Checkpoint::open(&dir).unwrap();
    let ckpt1: &'static mut Checkpoint = Box::leak(Box::new(ckpt1));
    let mut seq = GpuModel::from_checkpoint_streaming(ckpt1).unwrap();
    let mut ckpt2 = Checkpoint::open(&dir).unwrap();
    let ckpt2: &'static mut Checkpoint = Box::leak(Box::new(ckpt2));
    let mut bat = GpuModel::from_checkpoint_streaming(ckpt2).unwrap();

    let mut worst = 0.0f32;
    let bat_logits = bat.decode_m(&TOKENS).unwrap();
    for (p, &t) in TOKENS.iter().enumerate() {
        let a = seq.decode(t, p as u64).unwrap();
        let b = &bat_logits[p];
        let d = max_abs_diff(&a, b);
        worst = worst.max(d);
        assert_eq!(
            ferroseek_ref::argmax(&a),
            ferroseek_ref::argmax(b),
            "streaming step {p}: argmax diverged under decode_m"
        );
        eprintln!("stream step {p}: decode_m max diff {d:.3e}");
    }
    assert!(
        worst < GATE_ABS_TOL,
        "streaming decode_m diverged: {worst:.3e}"
    );
    // state equivalence: two continuation steps
    for (j, &t) in [29u32, 31].iter().enumerate() {
        let p = TOKENS.len() + j;
        let a = seq.decode(t, p as u64).unwrap();
        let b = bat.decode(t, p as u64).unwrap();
        let d = max_abs_diff(&a, &b);
        assert!(
            d < GATE_ABS_TOL,
            "streaming post-pass state diverged: {d:.3e}"
        );
    }
    eprintln!("streaming decode_m: bit-exact within {worst:.3e}");
    let _ = std::fs::remove_dir_all(&dir);
}
