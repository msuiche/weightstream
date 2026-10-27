//! Engram: n-gram hash lookups written into the residual stream (spec §8,
//! engram.py). Runs on the hc stream *before* the block.
//!
//! The hash multipliers reproduce `np.random.default_rng(10007 * layer_id)
//! .integers(0, bound, 4)` bit-exactly (see [`crate::rng::NumpyPcg64`]), and
//! the primes are drawn from a global never-reuse pool starting at
//! `engram_vocab_size - 1` — both deterministic given the config, exactly as
//! in training (spec §8).

use crate::config::RefConfig;
use crate::model::Runtime;
use crate::num;
use crate::rng::NumpyPcg64;
use crate::store::{fp8_row_scaled, WeightStore};
use std::collections::HashSet;
use std::io;

/// Bucket layout of the n-gram hash tables (engram.py `EngramLayout`).
pub struct EngramLayout {
    pub layer_ids: Vec<u32>,
    pub max_ngram: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// `primes[layer][ngram_size-1][head]` — bucket modulus.
    pub primes: Vec<Vec<Vec<u64>>>,
    /// `offsets[layer][col]` — prefix sum of the layer's primes, in column
    /// order (i-gram heads first).
    pub offsets: Vec<Vec<u64>>,
    /// `multipliers[layer][shift]` — odd, bounded, numpy-replicated.
    pub multipliers: Vec<Vec<u64>>,
    /// (max_ngram - 1) * n_heads hash columns per position.
    pub n_hash_cols: usize,
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    if n % 3 == 0 {
        return n == 3;
    }
    let mut i = 5u64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// The smallest prime above `start` not in `seen` (engram.py:9-13).
fn find_next_prime(start: u64, seen: &mut HashSet<u64>) -> u64 {
    let mut c = start + 1;
    while !is_prime(c) || !seen.insert(c) {
        c += 1;
    }
    c
}

impl EngramLayout {
    /// Build the layout from config. Does NOT check the table row counts —
    /// call [`EngramLayout::required_rows`] and compare against
    /// `engram_num_embeddings` (spec §12.5).
    pub fn build(cfg: &RefConfig) -> io::Result<Self> {
        if cfg.engram_layers.is_empty() {
            return Ok(EngramLayout {
                layer_ids: vec![],
                max_ngram: 0,
                n_heads: 0,
                head_dim: 0,
                primes: vec![],
                offsets: vec![],
                multipliers: vec![],
                n_hash_cols: 0,
            });
        }
        let max_ngram = cfg.engram_max_ngram as usize;
        let n_heads = cfg.engram_n_heads as usize;
        if max_ngram < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "engram_max_ngram_size >= 2 required",
            ));
        }
        let mut seen: HashSet<u64> = HashSet::new();
        let mut primes = Vec::new();
        let mut offsets = Vec::new();
        let bound = (i64::MAX as u64 / cfg.engram_compressed_vocab as u64) / 2;
        let bound = bound.max(1);
        let mut multipliers = Vec::new();
        for &layer in &cfg.engram_layers {
            let mut per_layer: Vec<Vec<u64>> = Vec::new();
            for _ in 1..max_ngram {
                let mut per_ngram = Vec::with_capacity(n_heads);
                for _ in 0..n_heads {
                    per_ngram.push(find_next_prime(
                        cfg.engram_prime_start.saturating_sub(1),
                        &mut seen,
                    ));
                }
                per_layer.push(per_ngram);
            }
            let flat: Vec<u64> = per_layer.iter().flatten().copied().collect();
            let mut offs = Vec::with_capacity(flat.len());
            let mut acc = 0u64;
            for &p in &flat {
                offs.push(acc);
                acc += p;
            }
            primes.push(per_layer);
            offsets.push(offs);
            let mut rng = NumpyPcg64::new(10007 * layer as u64);
            multipliers.push(
                rng.integers(bound, max_ngram)
                    .into_iter()
                    .map(|v| v * 2 + 1)
                    .collect(),
            );
        }
        Ok(EngramLayout {
            layer_ids: cfg.engram_layers.clone(),
            max_ngram,
            n_heads,
            head_dim: cfg.engram_head_dim as usize,
            primes,
            offsets,
            multipliers,
            n_hash_cols: (max_ngram - 1) * n_heads,
        })
    }

    /// Minimum table rows each layer's embed table must have (spec §12.5).
    pub fn required_rows(&self) -> Vec<u64> {
        self.offsets
            .iter()
            .zip(self.primes.iter())
            .map(|(offs, pr)| {
                offs.last().copied().unwrap_or(0)
                    + pr.last().and_then(|v| v.last()).copied().unwrap_or(0)
            })
            .collect()
    }
}

/// Compressed-id lookup for a raw token. Without a tokenizer-derived map
/// (host-side, spec §11), identity is used — synthetic tests and any
/// checkpoint load start there; call `RefModel::set_token_map` to install
/// the real one.
fn map_token(token_map: Option<&[i64]>, token: i64) -> i64 {
    match token_map {
        Some(m) => {
            let i = token.clamp(0, m.len() as i64 - 1) as usize;
            m[i]
        }
        None => token,
    }
}

/// Hash the current position into per-layer hash-id rows
/// (`NgramHashState.forward`, engram.py:159-184). Text-only: no DEAD/image
/// masking. Look-back stops at the sequence start; missing history takes the
/// compressed pad id.
///
/// Public (runtime-free form): the GPU engine hashes host-side on the same
/// compressed-id cache and uploads the ids.
pub fn hash_ids(
    cfg: &RefConfig,
    layout: &EngramLayout,
    eng_cache: &mut Vec<i64>,
    token_map: Option<&[i64]>,
    token: u32,
    pos: u64,
) -> Vec<Vec<u64>> {
    let cid = map_token(token_map, token as i64);
    debug_assert_eq!(eng_cache.len(), pos as usize);
    eng_cache.push(cid);
    let pad = map_token(token_map, cfg.engram_pad_id as i64);

    let mut tokens = vec![0i64; layout.max_ngram];
    for shift in 0..layout.max_ngram {
        tokens[shift] = if pos >= shift as u64 {
            eng_cache[(pos - shift as u64) as usize]
        } else {
            pad
        };
    }

    let mut out = Vec::with_capacity(layout.layer_ids.len());
    for l in 0..layout.layer_ids.len() {
        let mult = &layout.multipliers[l];
        // int64 products (no overflow: the bound guarantees it), XOR-rolled
        // one lookback at a time so step i hashes the (i+1)-gram
        let mut rolling = tokens[0].wrapping_mul(mult[0] as i64);
        let mut col = 0usize;
        let mut hashes = vec![0u64; layout.n_hash_cols];
        for i in 1..layout.max_ngram {
            rolling ^= tokens[i].wrapping_mul(mult[i] as i64);
            for h in 0..layout.n_heads {
                let prime = layout.primes[l][i - 1][h] as i64;
                hashes[col + h] = (rolling % prime) as u64 + layout.offsets[l][col + h];
            }
            col += layout.n_heads;
        }
        out.push(hashes);
    }
    out
}

/// `hash_ids` over the ref engine's runtime state.
pub fn hash_step(
    cfg: &RefConfig,
    layout: &EngramLayout,
    rt: &mut Runtime,
    token: u32,
    pos: u64,
) -> Vec<Vec<u64>> {
    let token_map = rt.token_map.clone();
    hash_ids(
        cfg,
        layout,
        &mut rt.engram_ids,
        token_map.as_deref(),
        token,
        pos,
    )
}

/// Merge the n-gram embeddings into the stream `h` (model.py:328-365):
/// one key per hc copy plus a shared value, gated by a normalized dot
/// product of the stream against the key.
pub fn merge(
    store: &mut dyn WeightStore,
    cfg: &RefConfig,
    layout: &EngramLayout,
    _layer_idx: usize,
    layer: u32,
    h: &mut [f32],
    hash_ids: &[u64],
) -> io::Result<()> {
    let hc = cfg.hc_mult as usize;
    let dim = cfg.dim as usize;
    let ehd = layout.head_dim;
    let cols = layout.n_hash_cols;

    // gather table rows and dequantize (fp8 rows, ue8m0 per-row scales)
    let (emb_w, emb_s) = store.get2(
        &format!("layers.{layer}.engram.embed.weight"),
        &format!("layers.{layer}.engram.embed.scale"),
    )?;
    let mut flat = vec![0.0f32; cols * ehd];
    for (c, id) in hash_ids.iter().take(cols).enumerate() {
        let r = *id as usize;
        if r >= emb_w.shape[0] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("engram hash row {r} out of table ({})", emb_w.shape[0]),
            ));
        }
        let row = fp8_row_scaled(emb_w.bytes, emb_s.bytes, r, ehd);
        flat[c * ehd..(c + 1) * ehd].copy_from_slice(&row);
    }
    crate::dump::f32s(
        format!("d_pos{}_eflat{layer}.f32", crate::dump::pos()),
        &flat,
    );
    // wkv: fp8 Linear over the flattened rows -> key per copy + shared value
    let (wkv_w, wkv_s) = store.get2(
        &format!("layers.{layer}.engram.wkv.weight"),
        &format!("layers.{layer}.engram.wkv.scale"),
    )?;
    let kv = crate::actquant::fp8_gemv(wkv_w.bytes, wkv_s.bytes, dim * (hc + 1), cols * ehd, &flat);

    // w = q_weight * k_weight, per copy (only ever used as a product)
    let qwv = {
        let qw = store.get(&format!("layers.{layer}.engram.q_weight"))?;
        crate::store::bytes_to_f32(&qw)?
    };
    let kwv = {
        let kw = store.get(&format!("layers.{layer}.engram.k_weight"))?;
        crate::store::bytes_to_f32(&kw)?
    };

    let eps = cfg.norm_eps;
    let dim_scale = num::rsqrt(dim as f32);
    for j in 0..hc {
        let key = &kv[j * dim..(j + 1) * dim];
        let value = &kv[hc * dim..(hc + 1) * dim];
        let mut dot = 0.0f32;
        let mut hs = 0.0f32;
        let mut ks = 0.0f32;
        for d in 0..dim {
            let w = qwv[j * dim + d] * kwv[j * dim + d];
            dot += h[j * dim + d] * w * key[d];
            hs += h[j * dim + d] * h[j * dim + d];
            ks += key[d] * key[d];
        }
        // normalized per (token, copy) over dim; eps = norm_eps (model.py:359-360)
        let rstd = num::rsqrt(hs / dim as f32 + eps) * num::rsqrt(ks / dim as f32 + eps);
        dot *= rstd * dim_scale;
        // signed sqrt before the sigmoid (model.py:362)
        let mag = dot.abs().max(1e-6).sqrt();
        let gate = num::sigmoid(if dot >= 0.0 { mag } else { -mag });
        let base = j * dim;
        for d in 0..dim {
            h[base + d] += gate * value[d];
        }
    }
    Ok(())
}
