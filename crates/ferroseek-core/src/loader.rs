//! Checkpoint loader: config + index + lazily-mmapped shards.
//!
//! Shards stay on disk until a tensor is requested; each shard is mmapped
//! on first touch and its header parsed once. Payload slices are zero-copy
//! borrows of the mapping — the GPU staging path reads straight from page
//! cache.

use crate::config::ModelConfig;
use crate::safetensors::{self, ShardHeader, TensorMeta};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

struct Shard {
    header: ShardHeader,
    mmap: Mmap,
}

pub struct Checkpoint {
    pub dir: PathBuf,
    pub config: ModelConfig,
    index: BTreeMap<String, String>,
    shards: BTreeMap<String, Shard>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        let config = ModelConfig::from_file(&dir.join("config.json"))?;
        let index = safetensors::parse_index(&dir.join("model.safetensors.index.json"))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            index,
            shards: BTreeMap::new(),
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    fn shard(&mut self, name: &str) -> std::io::Result<&Shard> {
        if !self.shards.contains_key(name) {
            let path = self.dir.join(name);
            let file = std::fs::File::open(&path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let header = safetensors::parse_header(&mut &mmap[..])?;
            self.shards.insert(name.to_string(), Shard { header, mmap });
        }
        Ok(self.shards.get(name).unwrap())
    }

    pub fn meta(&mut self, tensor: &str) -> std::io::Result<TensorMeta> {
        let shard = self
            .index
            .get(tensor)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, tensor))?
            .clone();
        let shard = self.shard(&shard)?;
        shard.header.tensors.get(tensor).cloned().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{tensor} not in shard header"))
        })
    }

    /// Zero-copy payload bytes for `tensor` (raw on-disk dtype).
    pub fn bytes(&mut self, tensor: &str) -> std::io::Result<&[u8]> {
        let shard_name = self
            .index
            .get(tensor)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, tensor))?
            .clone();
        let shard = self.shard(&shard_name)?;
        let meta = shard.header.tensors.get(tensor).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{tensor} not in shard header"))
        })?;
        let start = shard.header.data_start + meta.data_offsets[0];
        let end = shard.header.data_start + meta.data_offsets[1];
        Ok(&shard.mmap[start as usize..end as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::tests::fixture_json;

    fn write_checkpoint(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), fixture_json()).unwrap();
        let hdr = serde_json::json!({
            "embed.weight": {"dtype": "BF16", "shape": [4, 2], "data_offsets": [0, 16]}
        });
        let json = serde_json::to_vec(&hdr).unwrap();
        let mut shard = (json.len() as u64).to_le_bytes().to_vec();
        shard.extend_from_slice(&json);
        shard.extend_from_slice(&[7u8; 16]);
        std::fs::write(dir.join("model-00001-of-00001.safetensors"), &shard).unwrap();
        let idx = serde_json::json!({
            "metadata": {"total_size": 16},
            "weight_map": {"embed.weight": "model-00001-of-00001.safetensors"}
        });
        std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&idx).unwrap())
            .unwrap();
    }

    #[test]
    fn roundtrips_a_tensor() {
        let dir = std::env::temp_dir().join(format!("ferroseek-test-{}", std::process::id()));
        write_checkpoint(&dir);
        let mut ckpt = Checkpoint::open(&dir).unwrap();
        let meta = ckpt.meta("embed.weight").unwrap();
        assert_eq!(meta.dtype, "BF16");
        assert_eq!(meta.shape, vec![4, 2]);
        let bytes = ckpt.bytes("embed.weight").unwrap();
        assert_eq!(bytes, &[7u8; 16]);
        assert!(ckpt.meta("nope").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
