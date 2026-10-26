use ferroseek_core::{kv, ModelConfig};
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "ferroseek — DGX Spark-native inference for DeepSeek V4.1 Flash

USAGE:
    ferroseek info <model-dir>   Parse config.json, print shape and KV sizing
    ferroseek serve <model-dir>  OpenAI-compatible endpoint (phase 3)
    ferroseek bench <model-dir>  Decode/prefill microbench (phase 4)

Docs: docs/architecture.md
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") => cmd_info(args.get(1)),
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
            let ratio_str = if cfg.compress_ratios.is_empty() {
                String::new()
            } else {
                format!(
                    ", CSA ratios {} layers @1/2",
                    cfg.compress_ratios.iter().filter(|&&r| r == 2).count()
                )
            };
            println!("mla             {w}-wide latent{ratio_str}");
            if let (Some(fp8), Some(bf16)) = (
                kv::kv_bytes_per_token(&cfg, kv::FP8_BYTES),
                kv::kv_bytes_per_token(&cfg, kv::BF16_BYTES),
            ) {
                println!("kv/token        {fp8} B (fp8+scales) / {bf16} B (bf16)");
                for ctx in [262_144u64, 524_288, 1_048_576] {
                    let gib = ctx as f64 * fp8 as f64 / (1u64 << 30) as f64;
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
