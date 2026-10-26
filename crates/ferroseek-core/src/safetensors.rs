//! safetensors header and shard-index parsing.
//!
//! Only metadata is read here — tensor payloads stay on disk until the
//! loader mmaps them. Format: 8-byte little-endian header length, then a
//! JSON object mapping tensor names to {dtype, shape, data_offsets}.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}

impl TensorMeta {
    pub fn nbytes(&self) -> u64 {
        self.data_offsets[1] - self.data_offsets[0]
    }
}

#[derive(Debug, Clone)]
pub struct ShardHeader {
    pub tensors: BTreeMap<String, TensorMeta>,
    /// Byte offset where the data section starts (8 + header_len).
    pub data_start: u64,
}

pub fn parse_header(reader: &mut impl Read) -> std::io::Result<ShardHeader> {
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut json = vec![0u8; header_len];
    reader.read_exact(&mut json)?;
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut tensors = BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let meta: TensorMeta = serde_json::from_value(value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        tensors.insert(name, meta);
    }
    Ok(ShardHeader {
        tensors,
        data_start: 8 + header_len as u64,
    })
}

pub fn parse_header_file(path: &Path) -> std::io::Result<ShardHeader> {
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(0))?;
    parse_header(&mut file)
}

/// `model.safetensors.index.json` for sharded checkpoints:
/// tensor name → shard filename.
pub fn parse_index(path: &Path) -> std::io::Result<BTreeMap<String, String>> {
    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }
    let text = std::fs::read_to_string(path)?;
    let idx: Index = serde_json::from_str(&text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(idx.weight_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_shard() -> Vec<u8> {
        let hdr = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "embed.weight": {"dtype": "BF16", "shape": [129280, 5120], "data_offsets": [0, 1323827200]},
            "layers.0.attn.attn_sink": {"dtype": "F32", "shape": [64], "data_offsets": [1323827200, 1323827456]}
        });
        let json = serde_json::to_vec(&hdr).unwrap();
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&json);
        out
    }

    #[test]
    fn parses_header() {
        let bytes = synthetic_shard();
        let hdr = parse_header(&mut &bytes[..]).unwrap();
        assert_eq!(hdr.tensors.len(), 2);
        let emb = &hdr.tensors["embed.weight"];
        assert_eq!(emb.dtype, "BF16");
        assert_eq!(emb.shape, vec![129280, 5120]);
        assert_eq!(emb.nbytes(), 129280 * 5120 * 2);
        assert_eq!(hdr.data_start as usize, bytes.len());
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_header(&mut &[0u8; 8][..]).is_err());
    }
}
