mod serve;

use ferroseek_core::{kv, ModelConfig};
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "ferroseek — DGX Spark-native inference for DeepSeek V4.1 Flash

USAGE:
    ferroseek info <model-dir>     Parse config.json, print shape and KV sizing
    ferroseek weights <model-dir>  Validate shard coverage against the DS4.1F census
    ferroseek tokens <model-dir> <text>  Encode text, print token ids
    ferroseek decode <model-dir> <prompt> [max-tokens] [--engine cpu|gpu] [--stream] [--max-seq N]
                                   [--tp4 --rank N --ring nodes-ring4.local.json]
                                   [--prompts-file path]  More requests after the first (one per line)
                                   Greedy decode (default: cpu reference engine;
                                   gpu + --stream reads weights straight from the
                                   registered checkpoint mmaps, docs/streaming.md)
    ferroseek serve <model-dir> [--tp4 --ring nodes-ring4.local.json] [--port N]
                                   OpenAI-compatible endpoint on the head rank;
                                   POST /v1/completions (prompt, max_tokens, stream);
                                   workers wait on the control channel
    ferroseek bench <model-dir>    Decode/prefill microbench (phase 4)

Docs: docs/architecture.md
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") => cmd_info(args.get(1)),
        Some("weights") => cmd_weights(args.get(1)),
        Some("tokens") => cmd_tokens(args.get(1), args.get(2)),
        Some("decode") => cmd_decode(&args),
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        Some("serve") => cmd_serve(&args),
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        Some("serve") => {
            eprintln!("serve needs the cuda feature on linux");
            ExitCode::from(2)
        }
        Some("bench") => {
            eprintln!("bench: not yet — phases and kernel strategy in docs/architecture.md");
            ExitCode::from(2)
        }
        _ => {
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn cmd_info(dir: Option<&String>) -> ExitCode {
    let Some(dir) = dir else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let path = Path::new(dir).join("config.json");
    let cfg = match ModelConfig::from_file(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    println!("model_type      {}", cfg.model_type);
    println!("architectures   {}", cfg.architectures.join(", "));
    println!(
        "layers          {} (hidden {}, heads {}, vocab {})",
        cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.vocab_size
    );
    if let (Some(e), Some(k)) = (cfg.n_routed_experts, cfg.num_experts_per_tok) {
        println!("moe             {e} routed experts, top-{k}");
    }
    if let Some(mult) = cfg.hc_mult {
        println!("mhc             {mult}x hyper-connection");
    }
    match cfg.kv_width() {
        Some(w) => {
            let l = kv::layout(&cfg, 128, 43).unwrap();
            println!(
                "mla             {w}-wide latent, {} publisher caches ({} compressed + {} index el/token)",
                cfg.kv_source_layer_ids.len(),
                l.compressed_per_token,
                l.index_per_token
            );
            if let Some(bf16) = kv::kv_bytes_per_token(&cfg, kv::BF16_BYTES) {
                println!("kv/token        {bf16} B (bf16, fp4-snapped)");
                for ctx in [262_144u64, 524_288, 1_048_576] {
                    let gib = ctx as f64 * bf16 as f64 / (1u64 << 30) as f64;
                    println!("kv @ {ctx:>9} ctx   {gib:.1} GiB per sequence");
                }
            }
        }
        None => println!("mla             layout not recognised — KV sizing skipped"),
    }
    if cfg.is_dsa() {
        println!(
            "dsa             indexer dim {}, topk {}",
            cfg.index_head_dim.unwrap_or(0),
            cfg.index_topk.unwrap_or(0)
        );
    }
    ExitCode::SUCCESS
}

fn cmd_weights(dir: Option<&String>) -> ExitCode {
    let Some(dir) = dir else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let dir = Path::new(dir);
    let cfg = match ModelConfig::from_file(&dir.join("config.json")) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}: {e}", dir.join("config.json").display());
            return ExitCode::FAILURE;
        }
    };
    let index_path = dir.join("model.safetensors.index.json");
    let index = match ferroseek_core::safetensors::parse_index(&index_path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{}: {e}", index_path.display());
            return ExitCode::FAILURE;
        }
    };

    let expected = ferroseek_core::weights::expected_tensors(&cfg);
    let (missing, unexpected) = ferroseek_core::weights::diff_index(&cfg, &index);

    let shards: std::collections::BTreeSet<&String> = index.values().collect();
    println!("tensors         {} in {} shards", index.len(), shards.len());
    println!(
        "census          {} expected (text model, DS4.1F)",
        expected.len()
    );
    println!(
        "coverage        {}/{}",
        expected.len() - missing.len(),
        expected.len()
    );
    if !missing.is_empty() {
        println!("missing         {} (first 10):", missing.len());
        for n in missing.iter().take(10) {
            println!("  - {n}");
        }
    }
    if !unexpected.is_empty() {
        println!("not in census   {} (first 10):", unexpected.len());
        for n in unexpected.iter().take(10) {
            println!("  ? {n}");
        }
    }
    if missing.is_empty() {
        println!("ok              every serving tensor is present");
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// A decode engine behind the CLI: the CPU reference (default) or the GPU
/// engine with cuda-gated construction.
trait Decoder {
    fn decode(&mut self, token: u32, position: u64) -> std::io::Result<Vec<f32>>;
    fn set_token_map(&mut self, map: Vec<i64>);
    /// Re-zero per-request state for the next request (serving). The CPU
    /// reference rebuilds cheaply; the GPU engine zeroes its caches.
    fn reset(&mut self) -> std::io::Result<()> {
        Ok(())
    }
    /// Transport accounting for run reports ("" when there is none).
    fn coll_stats(&self) -> String {
        String::new()
    }
}

impl Decoder for ferroseek_ref::RefModel<'_> {
    fn decode(&mut self, token: u32, position: u64) -> std::io::Result<Vec<f32>> {
        self.decode(token, position)
    }
    fn set_token_map(&mut self, map: Vec<i64>) {
        self.set_token_map(map)
    }
}

#[cfg(feature = "cuda")]
impl Decoder for ferroseek_gpu::engine::GpuModel<'static> {
    // the CLI leaks the checkpoint for process lifetime, so the streaming
    // model is always 'static here
    fn decode(&mut self, token: u32, position: u64) -> std::io::Result<Vec<f32>> {
        self.decode(token, position)
    }
    fn set_token_map(&mut self, map: Vec<i64>) {
        self.set_token_map(map)
    }
    fn reset(&mut self) -> std::io::Result<()> {
        ferroseek_gpu::engine::GpuModel::reset(self)
    }
    fn coll_stats(&self) -> String {
        let coll = self.comm.as_ref().map(|c| c.stats()).unwrap_or_default();
        let prof = self.prof.summary();
        match (coll.is_empty(), prof.is_empty()) {
            (false, false) => format!("{coll}\n{prof}"),
            (false, true) => coll,
            (_, prof_is_empty) => {
                if prof_is_empty {
                    String::new()
                } else {
                    prof
                }
            }
        }
    }
}
fn ferroseek_gpu_stream_env() -> bool {
    std::env::var("FERROSEEK_STREAM")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn build_engine(
    engine: &str,
    ckpt: ferroseek_core::Checkpoint,
    stream: bool,
    max_seq: u64,
) -> std::io::Result<Box<dyn Decoder>> {
    match engine {
        "gpu" => {
            if stream {
                // Streaming weights live in the checkpoint's shard mmaps,
                // so the model borrows it: leak one box for the process
                // lifetime of this CLI run (mappings stay registered until
                // exit).
                let ckpt: &'static mut ferroseek_core::Checkpoint = Box::leak(Box::new(ckpt));
                let model = ferroseek_gpu::engine::GpuModel::from_checkpoint_streaming(ckpt)?
                    .with_max_seq_len(max_seq);
                Ok(Box::new(model))
            } else {
                let ckpt: &'static mut ferroseek_core::Checkpoint = Box::leak(Box::new(ckpt));
                let model = ferroseek_gpu::engine::GpuModel::from_checkpoint(ckpt)?
                    .with_max_seq_len(max_seq);
                Ok(Box::new(model))
            }
        }
        _ => {
            let ckpt: &'static mut ferroseek_core::Checkpoint = Box::leak(Box::new(ckpt));
            let model = ferroseek_ref::RefModel::from_checkpoint(ckpt)?;
            Ok(Box::new(model))
        }
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn build_engine_tp4(
    ckpt: ferroseek_core::Checkpoint,
    ring_path: &str,
    rank: Option<u8>,
    port: u16,
    max_seq: u64,
    wait_ms: u64,
    stream: bool,
) -> std::io::Result<(Box<dyn Decoder>, ferroseek_gpu::tp4::Session)> {
    // largest collective payload the engine issues: attention/moe partials
    // (dim), the head's vocab shard (vocab/4), index scores (max_seq/ratio)
    let cfg = ferroseek_ref::config::RefConfig::from_model_config(&ckpt.config)?;
    let mut elems = cfg.dim as usize;
    elems = elems.max(cfg.vocab as usize / 4);
    for &src in &cfg.kv_sources {
        let ratio = cfg.compress_ratios[src as usize] as u64;
        if ratio > 0 {
            elems = elems.max((max_seq / ratio + 1) as usize);
        }
    }
    let mut session = ferroseek_gpu::tp4::join(ring_path, rank, port, elems, wait_ms)?;
    let sharding = ferroseek_gpu::engine::Sharding {
        world_size: 4,
        rank: session.rank as u32,
    };
    let comm = session.take_comm();
    let ckpt: &'static mut ferroseek_core::Checkpoint = Box::leak(Box::new(ckpt));
    let model = if stream {
        ferroseek_gpu::engine::GpuModel::from_checkpoint_streaming_tp4(ckpt, sharding, comm)?
            .with_max_seq_len(max_seq)
    } else {
        ferroseek_gpu::engine::GpuModel::from_checkpoint_tp4(ckpt, sharding, comm)?
            .with_max_seq_len(max_seq)
    };
    Ok((Box::new(model), session))
}

#[cfg(not(feature = "cuda"))]
fn build_engine(
    engine: &str,
    ckpt: ferroseek_core::Checkpoint,
    _stream: bool,
    _max_seq: u64,
) -> std::io::Result<Box<dyn Decoder>> {
    if engine == "gpu" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "built without --features cuda; rebuild with `cargo build --features ferroseek/cuda`",
        ));
    }
    // the CPU ref engine borrows the checkpoint (mmap-backed weights); the
    // CLI runs one decode and exits, so leak one box for process lifetime
    let ckpt: &'static mut ferroseek_core::Checkpoint = Box::leak(Box::new(ckpt));
    let model = ferroseek_ref::RefModel::from_checkpoint(ckpt)?;
    Ok(Box::new(model))
}

/// `decode <model-dir> <prompt> [max-tokens] [--force-ids "1,2,3"] [--logits-out path]`
fn cmd_decode(args: &[String]) -> ExitCode {
    let (Some(dir), Some(prompt)) = (args.get(1), args.get(2)) else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let max_tokens: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let mut force_ids: Vec<u32> = Vec::new();
    let mut logits_out: Option<String> = None;
    let mut logits_raw: Option<String> = None;
    let mut engine = "cpu".to_string();
    let mut stream = false;
    let mut max_seq: u64 = 4096;
    let mut tp4 = false;
    let mut tp_rank: Option<u8> = None;
    let mut tp_ring = String::new();
    let mut tp_port: u16 = 29871;
    let mut prompts_file: Option<String> = None;
    let mut tp_wait_ms: u64 = 60_000;
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--force-ids" if i + 1 < args.len() => {
                force_ids = args[i + 1]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                i += 2;
            }
            "--prompts-file" if i + 1 < args.len() => {
                prompts_file = Some(args[i + 1].clone());
                i += 2;
            }
            "--logits-out" if i + 1 < args.len() => {
                logits_out = Some(args[i + 1].clone());
                i += 2;
            }
            "--logits-raw" if i + 1 < args.len() => {
                logits_raw = Some(args[i + 1].clone());
                i += 2;
            }
            "--engine" if i + 1 < args.len() => {
                engine = args[i + 1].clone();
                i += 2;
            }
            "--stream" => {
                stream = true;
                i += 1;
            }
            "--max-seq" if i + 1 < args.len() => {
                max_seq = args[i + 1].parse().unwrap_or(max_seq);
                i += 2;
            }
            "--tp4" => {
                tp4 = true;
                i += 1;
            }
            "--rank" if i + 1 < args.len() => {
                tp_rank = args[i + 1].parse().ok();
                i += 2;
            }
            "--ring" if i + 1 < args.len() => {
                tp_ring = args[i + 1].clone();
                i += 2;
            }
            "--tp-port" if i + 1 < args.len() => {
                tp_port = args[i + 1].parse().unwrap_or(tp_port);
                i += 2;
            }
            "--tp-wait-ms" if i + 1 < args.len() => {
                tp_wait_ms = args[i + 1].parse().unwrap_or(tp_wait_ms);
                i += 2;
            }
            _ => i += 1,
        }
    }
    if engine != "cpu" && engine != "gpu" {
        eprintln!("--engine must be cpu or gpu (got {engine})");
        return ExitCode::from(2);
    }
    // FERROSEEK_STREAM=1 selects the unified-memory streaming path without
    // the flag (docs/streaming.md)
    let stream = stream || ferroseek_gpu_stream_env();
    let dir = Path::new(dir);

    let tk = match ferroseek_tokenizer::Tokenizer::from_model_dir(dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ckpt = match ferroseek_core::Checkpoint::open(dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("checkpoint: {e}");
            return ExitCode::FAILURE;
        }
    };
    let t0_load = std::time::Instant::now();
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    let mut tp4_session: Option<ferroseek_gpu::tp4::Session> = None;
    let mut model = if tp4 {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            if tp_ring.is_empty() {
                eprintln!("--tp4 needs --ring <nodes-ring4.local.json>");
                return ExitCode::from(2);
            }
            match build_engine_tp4(
                ckpt, &tp_ring, tp_rank, tp_port, max_seq, tp_wait_ms, stream,
            ) {
                Ok((m, mut sess)) => {
                    if let Err(e) = sess.start_barrier() {
                        eprintln!("tp4 start barrier: {e}");
                        return ExitCode::FAILURE;
                    }
                    eprintln!("tp4 rank {}: all ranks loaded, decode starting", sess.rank);
                    tp4_session = Some(sess);
                    m
                }
                Err(e) => {
                    eprintln!("tp4: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            let _ = (tp_ring.clone(), tp_rank, tp_port, tp_wait_ms);
            eprintln!("--tp4 needs the cuda feature on linux");
            return ExitCode::FAILURE;
        }
    } else {
        match build_engine(&engine, ckpt, stream, max_seq) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("model ({engine}): {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    eprintln!("model loaded in {:?}", t0_load.elapsed());
    let t_map = std::time::Instant::now();
    match ferroseek_tokenizer::engram_map::build_compressed_token_map(&tk) {
        Ok(map) => {
            let size = 1 + map.iter().max().copied().unwrap_or(0);
            eprintln!(
                "engram token map: {} compressed ids in {:?}",
                size,
                t_map.elapsed()
            );
            model.set_token_map(map);
        }
        Err(e) => eprintln!("warning: engram token map not built ({e}); identity fallback"),
    }

    // Per-position gate stats: logprob of the token actually consumed,
    // its rank, and the top-1 id at that position (fp32 log-softmax).
    let mut records: Vec<String> = Vec::new();
    let record = |logits: &[f32], pos: u64, consumed: u32, records: &mut Vec<String>| {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let (mut top1, mut top2, mut top1_idx) = (f32::NEG_INFINITY, f32::NEG_INFINITY, 0u32);
        let mut sum_exp = 0.0f64;
        for (i, &l) in logits.iter().enumerate() {
            sum_exp += ((l - max) as f64).exp();
            if l > top1 {
                top2 = top1;
                top1 = l;
                top1_idx = i as u32;
            } else if l > top2 {
                top2 = l;
            }
        }
        let logprob = logits[consumed as usize] - max - (sum_exp.ln() as f32);
        let rank = logits
            .iter()
            .filter(|&&l| l > logits[consumed as usize])
            .count()
            + 1;
        let norm = sum_exp.ln() as f32;
        records.push(format!(
            "{{\"pos\":{pos},\"consumed\":{consumed},\"logprob\":{logprob:.6},\"rank\":{rank},\"top1\":{top1_idx},\"top1_logprob\":{:.6},\"top1_top2_gap\":{:.6}}}",
            top1 - max - norm,
            top1 - top2
        ));
    };

    if let Some(dir) = &logits_raw {
        let _ = std::fs::create_dir_all(dir);
    }
    let dump_raw = |logits: &[f32], pos: u64, dir: &Option<String>| {
        if let Some(d) = dir {
            let path = std::path::Path::new(d).join(format!("pos{pos}.f32"));
            let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Err(e) = std::fs::write(&path, bytes) {
                eprintln!("logits-raw {path:?}: {e}");
            }
        }
    };

    // One request: prefill + greedy/forced generation. Every rank runs the
    // same request in lockstep (the collectives require it); results print
    // on each rank.
    let mut run_request = |model: &mut Box<dyn Decoder>, prompt: &str| -> Option<Vec<u32>> {
        let ids = tk.encode(prompt);
        if ids.is_empty() {
            eprintln!("empty prompt");
            return None;
        }
        let t0 = std::time::Instant::now();
        let mut logits = Vec::new();
        let mut pos = 0u64;
        for (i, &t) in ids.iter().enumerate() {
            match model.decode(t, pos) {
                Ok(l) => {
                    logits = l;
                    dump_raw(&logits, pos, &logits_raw);
                }
                Err(e) => {
                    eprintln!("decode@{pos}: {e}");
                    return None;
                }
            }
            pos += 1;
            if logits_out.is_some() {
                if let Some(&consumed) = ids.get(i + 1) {
                    record(&logits, pos, consumed, &mut records);
                }
            }
        }
        eprintln!("prefill: {} tokens in {:?}", ids.len(), t0.elapsed());

        let mut generated = Vec::new();
        let mut forced = force_ids.iter();
        for _ in 0..max_tokens {
            let next = match forced.next() {
                Some(&id) => id, // teacher-forced (gate mode)
                None => ferroseek_ref::sample::argmax(&logits),
            };
            generated.push(next);
            record(&logits, pos, next, &mut records);
            let t1 = std::time::Instant::now();
            let hit_eos = force_ids.is_empty() && Some(next) == tk.eos_token_id;
            if hit_eos {
                break;
            }
            match model.decode(next, pos) {
                Ok(l) => {
                    logits = l;
                    dump_raw(&logits, pos, &logits_raw);
                }
                Err(e) => {
                    eprintln!("decode@{pos}: {e}");
                    return None;
                }
            }
            pos += 1;
            eprintln!("  [{}] {} ( {:?} )", pos, tk.decode(&[next]), t1.elapsed());
        }
        Some(generated)
    };

    // Requests: the CLI prompt first, then --prompts-file lines. With tp4
    // the head distributes each next request over the control channel and
    // every rank re-zeros its per-request state (GpuModel::reset); the
    // 5-minute load happens once for all of them.
    let prompts: Vec<String> = match &prompts_file {
        Some(path) => std::fs::read_to_string(path)
            .map(|text| {
                text.lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect()
            })
            .unwrap_or_default(),
        None => Vec::new(),
    };

    let mut overall = ExitCode::SUCCESS;

    // Request stream, protocol-driven: the head owns the list (CLI prompt
    // + its prompts-file) and broadcasts each next request over the tp4
    // control channel; workers never count requests locally — they decode,
    // reset, and wait for the head's next Next (or Bye, which ends the
    // run). Single-node (no tp4) just walks the head's local list.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    let is_worker = tp4_session.as_ref().map(|s| s.rank != 0).unwrap_or(false);
    #[cfg(not(all(feature = "cuda", target_os = "linux")))]
    let is_worker = false;

    if is_worker {
        // worker: request 1 is the CLI prompt (already broadcast at Start
        // by convention — every rank got it in argv), then wait for each
        // next request from the head
        match run_request(&mut model, prompt) {
            Some(generated) => println!("---\n{}", tk.decode(&generated)),
            None => overall = ExitCode::FAILURE,
        }
        while overall == ExitCode::SUCCESS {
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            if let Some(sess) = tp4_session.as_mut() {
                match sess.next_request(None) {
                    Ok(Some(cmd)) => {
                        let next_prompt = cmd
                            .get("prompt")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if next_prompt.is_empty() {
                            overall = ExitCode::FAILURE;
                            break;
                        }
                        if let Err(e) = model.reset() {
                            eprintln!("reset: {e}");
                            overall = ExitCode::FAILURE;
                            break;
                        }
                        match run_request(&mut model, &next_prompt) {
                            Some(generated) => {
                                println!("---\n{}", tk.decode(&generated))
                            }
                            None => {
                                overall = ExitCode::FAILURE;
                                break;
                            }
                        }
                    }
                    Ok(None) => break, // head said Bye: done
                    Err(e) => {
                        eprintln!("tp4 next_request: {e}");
                        overall = ExitCode::FAILURE;
                        break;
                    }
                }
            }
        }
    } else {
        // head (or single-node): decode the CLI prompt, then each
        // prompts-file line, resetting between requests
        match run_request(&mut model, prompt) {
            Some(generated) => println!("---\n{}", tk.decode(&generated)),
            None => overall = ExitCode::FAILURE,
        }
        for p in prompts.iter() {
            if overall != ExitCode::SUCCESS {
                break;
            }
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            if let Some(sess) = tp4_session.as_mut() {
                match sess.next_request(Some(serde_json::json!({
                    "prompt": p,
                    "max_tokens": max_tokens,
                }))) {
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(e) => {
                        eprintln!("tp4 next_request: {e}");
                        overall = ExitCode::FAILURE;
                        break;
                    }
                }
            }
            if let Err(e) = model.reset() {
                eprintln!("reset: {e}");
                overall = ExitCode::FAILURE;
                break;
            }
            match run_request(&mut model, p) {
                Some(generated) => println!("---\n{}", tk.decode(&generated)),
                None => {
                    overall = ExitCode::FAILURE;
                    break;
                }
            }
        }
    }

    let cs = model.coll_stats();
    if !cs.is_empty() {
        eprintln!("{cs}");
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    if let Some(sess) = tp4_session.as_mut() {
        sess.finish();
    }

    if let Some(path) = logits_out {
        let json = format!("[{}]", records.join(","));
        if let Err(e) = std::fs::write(&path, json) {
            eprintln!("logits-out {path}: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("logits written to {path}");
    }

    overall
}

/// `serve <model-dir> [--tp4 --rank N --ring ...] [--port N]` — the
/// OpenAI-compatible endpoint (docs/transport.md milestone 4). The head
/// rank owns the engine and the HTTP listener; workers loop on the tp4
/// control channel (`Session::next_request`), decoding whatever the head
/// broadcasts in lockstep. One request at a time — the ring's SPMD
/// contract serializes requests by design; batching is a later milestone.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cmd_serve(args: &[String]) -> ExitCode {
    let Some(dir) = args.get(1) else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let mut port: u16 = 8000;
    let mut tp_ring = String::new();
    let mut tp_rank: Option<u8> = None;
    let mut tp_port: u16 = 29871;
    let mut max_seq: u64 = 4096;
    let mut tp_wait_ms: u64 = 60_000;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--port" if i + 1 < args.len() => {
                port = args[i + 1].parse().unwrap_or(port);
                i += 2;
            }
            "--tp4" => {
                i += 1;
            }
            "--rank" if i + 1 < args.len() => {
                tp_rank = args[i + 1].parse().ok();
                i += 2;
            }
            "--ring" if i + 1 < args.len() => {
                tp_ring = args[i + 1].clone();
                i += 2;
            }
            "--tp-port" if i + 1 < args.len() => {
                tp_port = args[i + 1].parse().unwrap_or(tp_port);
                i += 2;
            }
            "--max-seq" if i + 1 < args.len() => {
                max_seq = args[i + 1].parse().unwrap_or(max_seq);
                i += 2;
            }
            "--tp-wait-ms" if i + 1 < args.len() => {
                tp_wait_ms = args[i + 1].parse().unwrap_or(tp_wait_ms);
                i += 2;
            }
            _ => i += 1,
        }
    }
    if tp_ring.is_empty() {
        eprintln!("serve --tp4 needs --ring <nodes-ring4.local.json>");
        return ExitCode::from(2);
    }

    let tk = match ferroseek_tokenizer::Tokenizer::from_model_dir(Path::new(dir)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ckpt = match ferroseek_core::Checkpoint::open(Path::new(dir)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("checkpoint: {e}");
            return ExitCode::FAILURE;
        }
    };
    let t0_load = std::time::Instant::now();
    let (mut model, mut session) = match build_engine_tp4(
        ckpt, &tp_ring, tp_rank, tp_port, max_seq, tp_wait_ms,
        false, // upload mode — the production path (docs/transport.md)
    ) {
        Ok((m, s)) => (m, s),
        Err(e) => {
            eprintln!("tp4: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = session.start_barrier() {
        eprintln!("tp4 start barrier: {e}");
        return ExitCode::FAILURE;
    }
    eprintln!(
        "tp4 rank {}: all ranks loaded in {:?}",
        session.rank,
        t0_load.elapsed()
    );
    match ferroseek_tokenizer::engram_map::build_compressed_token_map(&tk) {
        Ok(map) => model.set_token_map(map),
        Err(e) => eprintln!("warning: engram token map not built ({e})"),
    }

    let is_head = session.rank == 0;
    if !is_head {
        // worker: decode whatever the head broadcasts, forever
        loop {
            match session.next_request(None) {
                Ok(Some(cmd)) => {
                    let prompt = cmd
                        .get("prompt")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let max_tokens =
                        cmd.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(64) as u32;
                    if prompt.is_empty()
                        || serve_decode(&mut model, &tk, &prompt, max_tokens, None).is_err()
                    {
                        break;
                    }
                    if model.reset().is_err() {
                        break;
                    }
                }
                _ => break, // Bye or error: done
            }
        }
        return ExitCode::SUCCESS;
    }

    // head: the submit queue carries (request, token-sender) pairs; the
    // HTTP thread creates the per-request channel, the decode owner
    // replies on it. One request at a time — the ring is lockstep SPMD.
    let (tx, rx) = std::sync::mpsc::channel::<(
        serve::ServeRequest,
        std::sync::mpsc::Sender<serve::TokenEvent>,
    )>();
    let http = std::thread::spawn(move || {
        let _ = serve::serve_forever(
            &format!("0.0.0.0:{port}"),
            move |req: serve::ServeRequest| {
                let (tok_tx, tok_rx) = std::sync::mpsc::channel::<serve::TokenEvent>();
                tx.send((req, tok_tx)).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, e.to_string())
                })?;
                Ok(tok_rx)
            },
        );
    });

    // decode owner: one request at a time, lockstep with the workers
    while let Ok((req, tok_tx)) = rx.recv() {
        // broadcast to the workers (they mirror the decode in lockstep)
        if let Err(e) = session.next_request(Some(serde_json::json!({
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
        }))) {
            eprintln!("tp4 next_request: {e}");
            let _ = tok_tx.send(serve::TokenEvent::Done);
            break;
        }
        let _ = serve_decode(&mut model, &tk, &req.prompt, req.max_tokens, Some(tok_tx));
        let _ = model.reset();
    }
    let _ = http.join();
    session.finish();
    ExitCode::SUCCESS
}

/// Decode `prompt` streaming each generated token to `sink` (when Some).
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn serve_decode(
    model: &mut Box<dyn Decoder>,
    tk: &ferroseek_tokenizer::Tokenizer,
    prompt: &str,
    max_tokens: u32,
    mut sink: Option<std::sync::mpsc::Sender<serve::TokenEvent>>,
) -> std::io::Result<()> {
    let ids = tk.encode(prompt);
    if ids.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "empty prompt",
        ));
    }
    let mut logits = Vec::new();
    let mut pos = 0u64;
    for &t in ids.iter() {
        logits = model.decode(t, pos)?;
        pos += 1;
    }
    for _ in 0..max_tokens {
        let next = ferroseek_ref::sample::argmax(&logits);
        if let Some(tx) = sink.as_ref() {
            let _ = tx.send(serve::TokenEvent::Token(tk.decode(&[next])));
        }
        if Some(next) == tk.eos_token_id {
            break;
        }
        logits = model.decode(next, pos)?;
        pos += 1;
    }
    if let Some(tx) = sink.as_ref() {
        let _ = tx.send(serve::TokenEvent::Done);
    }
    Ok(())
}
fn cmd_tokens(dir: Option<&String>, text: Option<&String>) -> ExitCode {
    let (Some(dir), Some(text)) = (dir, text) else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let tk = match ferroseek_tokenizer::Tokenizer::from_model_dir(Path::new(dir)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let ids = tk.encode(text);
    println!("{:?}", ids);
    println!("{}", ids.len());
    ExitCode::SUCCESS
}
