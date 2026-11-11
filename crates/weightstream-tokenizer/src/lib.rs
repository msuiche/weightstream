//! Tokenizer wrapper: HF `tokenizer.json` plus the special-token contract
//! from `tokenizer_config.json` (no auto BOS/EOS insertion — chat framing
//! adds them explicitly, per `add_bos_token: false`).

use std::path::Path;

pub mod engram_map;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

#[derive(serde::Deserialize)]
struct TokenizerConfig {
    bos_token: Option<TokenEntry>,
    eos_token: Option<TokenEntry>,
    pad_token: Option<TokenEntry>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum TokenEntry {
    Plain(String),
    Structured { content: String },
}

impl TokenEntry {
    fn content(&self) -> &str {
        match self {
            TokenEntry::Plain(s) => s,
            TokenEntry::Structured { content } => content,
        }
    }
}

impl Tokenizer {
    pub fn from_model_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let config: Option<TokenizerConfig> =
            std::fs::read_to_string(dir.join("tokenizer_config.json"))
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok());
        let id_of = |entry: Option<&TokenEntry>, inner: &tokenizers::Tokenizer| {
            entry.and_then(|e| inner.token_to_id(e.content()))
        };
        Ok(Self {
            bos_token_id: config
                .as_ref()
                .and_then(|c| id_of(c.bos_token.as_ref(), &inner)),
            eos_token_id: config
                .as_ref()
                .and_then(|c| id_of(c.eos_token.as_ref(), &inner)),
            pad_token_id: config
                .as_ref()
                .and_then(|c| id_of(c.pad_token.as_ref(), &inner)),
            inner,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, false)
            .expect("encode")
            .get_ids()
            .to_vec()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).expect("decode")
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Raw fast tokenizer (engram.py's `backend_tokenizer` equivalent).
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Option<Tokenizer> {
        let dir = Path::new("/tmp/ds41f-ref");
        Tokenizer::from_model_dir(dir).ok()
    }

    #[test]
    fn loads_real_tokenizer() {
        let Some(tk) = tokenizer() else {
            eprintln!("tokenizer.json not present; skipping");
            return;
        };
        assert!(tk.vocab_size() >= 129280 - 100);
        assert_eq!(tk.eos_token_id, Some(1));
        assert!(tk.bos_token_id.is_some());
    }

    #[test]
    fn roundtrip() {
        let Some(tk) = tokenizer() else { return };
        let text = "DeepSeek V4.1 Flash runs on four DGX Sparks.";
        let ids = tk.encode(text);
        assert!(!ids.is_empty());
        assert_eq!(tk.decode(&ids), text);
        // BOS/EOS are never added implicitly
        assert_ne!(ids[0], tk.bos_token_id.unwrap());
    }
}
