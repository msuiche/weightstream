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
        launch!(f<<<(n / 32) as u32, 32, 0, stream>>>(d_x.as_device_ptr(), d_q.as_device_ptr(), d_s.as_device_ptr(), n as i32)).unwrap();
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
            "quant-flip"
        } else {
            panic!("step {p}: GPU diverges from CPU reference: {d:.3e} (above the flip bound {FLIP_ABS_TOL:.0e})")
        };
        eprintln!(
            "step {p}: max abs diff {d:.3e} [{tier}] (argmax {} vs {})",
            ferroseek_ref::argmax(&cpu_logits),
            ferroseek_ref::argmax(&gpu_logits),
        );
    }
    eprintln!("GATE: max abs diff over 8 steps = {max_abs:.3e}; {clean_steps}/8 steps at the ≤{GATE_ABS_TOL:.0e} noise floor");
    assert!(
        clean_steps >= GATE_MIN_CLEAN_STEPS,
        "only {clean_steps}/8 steps at the noise floor — systematic divergence"
    );

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
    let (m, n, k) = (6usize, 512usize, 1024usize);
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
fn fp4_gemm_tile_matches_cpu_and_m1() {
    // M-batch tiled fp4 expert GEMM: CPU mirror (unpack fp4 per the
    // engine's table) + per-row cross-check against the M=1 fp4_gemm at
    // the gate tolerance (reduction-order divergence class, engine docs).
    let guard = Cuda::global();
    let c = guard.as_ref().unwrap();
    let (m, n, k) = (6usize, 512usize, 1024usize);
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
}
