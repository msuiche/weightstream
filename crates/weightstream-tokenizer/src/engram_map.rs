//! Engram compressed token map (engram.py:17-61, model-spec §8/§11).
//!
//! Every token id maps onto a smaller id space where tokens that normalize
//! alike collapse together (" The" / "the" / "THE" hash the same). The map
//! size matters beyond bounds checking: the hash multipliers are derived
//! from it. For the 1017 checkpoint the compressed size must be 99092.

use crate::Tokenizer;
use tokenizers::normalizers::replace::ReplacePattern;
use tokenizers::normalizers::{Lowercase, Replace, Sequence, Strip, StripAccents, NFD, NFKC};
use tokenizers::{NormalizedString, Normalizer};

/// The expected compressed vocab size on DS4.1F-1017 (config:
/// engram_compressed_vocab_size).
pub const EXPECTED_COMPRESSED_VOCAB: usize = 99092;

fn build_normalizer() -> Sequence {
    // Mirrors engram.py exactly: a private-use sentinel keeps a token that
    // is exactly one space from collapsing into the empty string.
    let sentinel = "\u{e000}";
    Sequence::new(vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        Replace::new(ReplacePattern::Regex(r"[ \t\r\n]+".into()), " ")
            .unwrap()
            .into(),
        Replace::new(ReplacePattern::Regex(r"^ $".into()), sentinel)
            .unwrap()
            .into(),
        Strip::new(true, true).into(),
        Replace::new(sentinel, " ").unwrap().into(),
    ])
}

fn normalize_str(normalizer: &Sequence, text: &str) -> String {
    let mut ns = NormalizedString::from(text);
    let _ = normalizer.normalize(&mut ns);
    ns.get().to_string()
}

/// Compressed id per token id (index = token id). Panics never; returns
/// Err on tokenizer inconsistencies.
pub fn build_compressed_token_map(tk: &Tokenizer) -> Result<Vec<i64>, String> {
    let normalizer = build_normalizer();
    let inner = tk.inner();
    let n = inner.get_vocab_size(true);
    let mut key_to_new: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    let mut lookup = vec![0i64; n];
    for id in 0..n as u32 {
        let text = inner
            .decode(&[id], false)
            .map_err(|e| format!("decode {id}: {e}"))?;
        let key = if text.contains('\u{fffd}') {
            // partial UTF-8 byte token: keyed by its raw form
            inner
                .id_to_token(id)
                .ok_or_else(|| format!("id_to_token {id}"))?
        } else {
            let normalized = normalize_str(&normalizer, &text);
            if normalized.is_empty() {
                text
            } else {
                normalized
            }
        };
        let next = key_to_new.len() as i64;
        lookup[id as usize] = *key_to_new.entry(key).or_insert(next);
    }
    Ok(lookup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn compressed_map_size_matches_checkpoint() {
        let Ok(tk) = Tokenizer::from_model_dir(Path::new("/tmp/ds41f-ref")) else {
            eprintln!("tokenizer.json not present; skipping");
            return;
        };
        let map = build_compressed_token_map(&tk).unwrap();
        let size = 1 + map.iter().max().unwrap();
        assert_eq!(size as usize, EXPECTED_COMPRESSED_VOCAB);
        // pad token (id 2) must exist in map
        assert_eq!(map.len(), tk.vocab_size());
    }
}
