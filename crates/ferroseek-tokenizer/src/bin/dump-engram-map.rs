//! Debug tool: dump the engram compressed token map for a tokenizer dir,
//! as little-endian i32 (one per token id), for comparison against the
//! reference `engram.py:build_compressed_token_map`.
//!
//! ```text
//! cargo run -p ferroseek-tokenizer --bin dump-engram-map -- <model-dir> <out.i32>
//! ```

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(dir), Some(out)) = (args.first(), args.get(1)) else {
        eprintln!("usage: dump-engram-map <model-dir> <out.i32>");
        std::process::exit(2);
    };
    let tk = match ferroseek_tokenizer::Tokenizer::from_model_dir(Path::new(dir)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer: {e}");
            std::process::exit(1);
        }
    };
    let map = ferroseek_tokenizer::engram_map::build_compressed_token_map(&tk).unwrap();
    let size = 1 + map.iter().max().unwrap();
    eprintln!("{} tokens, {} compressed ids", map.len(), size);
    let bytes: Vec<u8> = map.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect();
    std::fs::write(out, bytes).expect("write");
}
