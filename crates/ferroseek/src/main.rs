use ferroseek_core::{kv, ModelConfig};
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "ferroseek — DGX Spark-native inference for DeepSeek V4.1 Flash

USAGE:
    ferroseek info <model-dir>     Parse config.json, print shape and KV sizing
    ferroseek weights <model-dir>  Validate shard coverage against the DS4.1F census
    ferroseek tokens <model-dir> <text>  Encode text, print token ids
    ferroseek decode <model-dir> <prompt> [max-tokens]  Greedy decode on the CPU reference engine
    ferroseek serve <model-dir>    OpenAI-compatible endpoint (phase 3)
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
        Some("serve") | Some("bench") => {
            eprintln!(
                "{}: not yet — phases and kernel strategy in docs/architecture.md",
                args[0]
            );
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
    println!("census          {} expected (text model, DS4.1F)", expected.len());
    println!("coverage        {}/{}", expected.len() - missing.len(), expected.len());
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

/// `decode <model-dir> <prompt> [max-tokens] [--force-ids "1,2,3"] [--logits-out path]`
fn cmd_decode(args: &[String]) -> ExitCode {
    let (Some(dir), Some(prompt)) = (args.get(1), args.get(2)) else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let max_tokens: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let mut force_ids: Vec<u32> = Vec::new();
    let mut logits_out: Option<String> = None;
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
            "--logits-out" if i + 1 < args.len() => {
                logits_out = Some(args[i + 1].clone());
                i += 2;
            }
            _ => i += 1,
        }
    }
    let dir = Path::new(dir);

    let tk = match ferroseek_tokenizer::Tokenizer::from_model_dir(dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut ckpt = match ferroseek_core::Checkpoint::open(dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("checkpoint: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("loading model (mmap, dequant on demand)…");
    let mut model = match ferroseek_ref::RefModel::from_checkpoint(&mut ckpt) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("model: {e}");
            return ExitCode::FAILURE;
        }
    };
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

    let ids = tk.encode(prompt);
    if ids.is_empty() {
        eprintln!("empty prompt");
        return ExitCode::from(2);
    }

    // Per-position gate stats: logprob of the token actually consumed,
    // its rank, and the top-1 id at that position (fp32 log-softmax).
    let mut records: Vec<String> = Vec::new();
    let mut record = |logits: &[f32], pos: u64, consumed: u32, records: &mut Vec<String>| {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let (mut top1, mut top1_idx) = (f32::NEG_INFINITY, 0u32);
        let mut sum_exp = 0.0f64;
        for (i, &l) in logits.iter().enumerate() {
            sum_exp += ((l - max) as f64).exp();
            if l > top1 {
                top1 = l;
                top1_idx = i as u32;
            }
        }
        let logprob = logits[consumed as usize] - max - (sum_exp.ln() as f32);
        let rank = logits.iter().filter(|&&l| l > logits[consumed as usize]).count() + 1;
        records.push(format!(
            "{{\"pos\":{pos},\"consumed\":{consumed},\"logprob\":{logprob:.6},\"rank\":{rank},\"top1\":{top1_idx},\"top1_logprob\":{:.6}}}",
            top1 - max - (sum_exp.ln() as f32)
        ));
    };

    let t0 = std::time::Instant::now();
    let mut logits = Vec::new();
    let mut pos = 0u64;
    for (i, &t) in ids.iter().enumerate() {
        match model.decode(t, pos) {
            Ok(l) => logits = l,
            Err(e) => {
                eprintln!("decode@{pos}: {e}");
                return ExitCode::FAILURE;
            }
        }
        pos += 1;
        // logits now predict the next token; record it when teacher-forced
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
            Ok(l) => logits = l,
            Err(e) => {
                eprintln!("decode@{pos}: {e}");
                return ExitCode::FAILURE;
            }
        }
        pos += 1;
        eprintln!("  [{}] {} ( {:?} )", pos, tk.decode(&[next]), t1.elapsed());
    }

    if let Some(path) = logits_out {
        let json = format!("[{}]", records.join(","));
        if let Err(e) = std::fs::write(&path, json) {
            eprintln!("logits-out {path}: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("logits written to {path}");
    }

    println!("---\n{}", tk.decode(&generated));
    ExitCode::SUCCESS
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
