//! Debug tool: dump the engram hash ids for a token sequence under a REAL
//! config + compressed token map, for comparison against the reference
//! `engram.py:NgramHashState` (layout primes/offsets/multipliers at release
//! scale — no weights involved).
//!
//! ```text
//! cargo run -p weightstream-ref --bin dump-engram-hash -- \
//!     <config.json> <tokens.csv> <token-map.i32> <out-dir>
//! ```

use weightstream_core::ModelConfig;
use weightstream_ref::engram::{hash_ids, EngramLayout};
use weightstream_ref::RefConfig;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(cfg_path), Some(tokens), Some(map_path), Some(out)) =
        (args.first(), args.get(1), args.get(2), args.get(3))
    else {
        eprintln!("usage: dump-engram-hash <config.json> <tokens.csv> <token-map.i32> <out-dir>");
        std::process::exit(2);
    };
    let cfg = ModelConfig::from_file(Path::new(cfg_path)).expect("config");
    let rcfg = RefConfig::from_model_config(&cfg).expect("ref config");
    let layout = EngramLayout::build(&rcfg).expect("engram layout");

    let tokens: Vec<u32> = tokens
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let map_bytes = std::fs::read(map_path).expect("token map");
    let token_map: Vec<i64> = map_bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
        .collect();

    let out = Path::new(out);
    std::fs::create_dir_all(out).expect("mkdir");
    let mut cache: Vec<i64> = Vec::new();
    for (p, &t) in tokens.iter().enumerate() {
        let per_layer = hash_ids(&rcfg, &layout, &mut cache, Some(&token_map), t, p as u64);
        for (l, hashes) in per_layer.iter().enumerate() {
            let bytes: Vec<u8> = hashes.iter().flat_map(|&v| v.to_le_bytes()).collect();
            std::fs::write(out.join(format!("hash_pos{p}_l{l}.i64")), bytes).expect("write");
        }
    }
    eprintln!(
        "dumped hashes for {} positions x {} engram layers to {}",
        tokens.len(),
        layout.layer_ids.len(),
        out.display()
    );
}
