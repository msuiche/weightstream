//! Weight sourcing. The engine reads every tensor through [`WeightStore`] as
//! raw on-disk bytes; quantized tensors are never densified. Two
//! implementations exist — an mmap-backed checkpoint adapter and
//! [`SyntheticWeights`] for tests — and both serve the exact same byte
//! layouts, so tests exercise the same dequantization code path as real
//! checkpoints (gate honesty requirement).

use crate::config::RefConfig;
use crate::layout::{self, TKind, TSpec};
use crate::rng::splitmix64;
use ferroseek_core::loader::Checkpoint;
use ferroseek_core::quant::{
    bf16_to_f32, e4m3_to_f32, e8m0_to_f32, f32_to_bf16, f32_to_e2m1, f32_to_e4m3, log2_ceil, pow2i,
};
use std::collections::{BTreeMap, HashMap};
use std::io;

/// On-disk (or synthetic) tensor element type.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DType {
    Bf16,
    F32,
    Fp8E4m3,
    /// UE8M0 scale byte (bare power-of-two exponent).
    E8m0,
    /// Two E2M1 values per byte, low nibble = even element.
    Fp4Packed,
    U8,
}

impl DType {
    /// safetensors dtype string (as written by convert.py / the HF export).
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "BF16" | "BFLOAT16" => Some(DType::Bf16),
            "F32" | "FP32" | "FLOAT32" => Some(DType::F32),
            "F8_E4M3" | "FLOAT8_E4M3" | "FP8" => Some(DType::Fp8E4m3),
            "F8_E8M0" | "FLOAT8_E8M0" => Some(DType::E8m0),
            "F4_E2M1" | "FLOAT4_E2M1" => Some(DType::Fp4Packed),
            "I8" | "U8" | "INT8" | "UINT8" => Some(DType::U8),
            _ => None,
        }
    }
}

/// Zero-copy view of one tensor's payload in its stored layout.
pub struct TensorView<'s> {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: &'s [u8],
}

/// Raw tensor access. `get2` exists because a quantized weight and its
/// scale must be borrowed at the same time.
pub trait WeightStore {
    fn get(&mut self, name: &str) -> io::Result<TensorView<'_>>;
    fn get2(&mut self, a: &str, b: &str) -> io::Result<(TensorView<'_>, TensorView<'_>)>;
}

/// mmap-backed adapter over a ferroseek-core [`Checkpoint`].
pub struct CkptStore<'a> {
    ckpt: &'a mut Checkpoint,
}

impl<'a> CkptStore<'a> {
    pub fn new(ckpt: &'a mut Checkpoint) -> Self {
        Self { ckpt }
    }
}

impl WeightStore for CkptStore<'_> {
    fn get(&mut self, name: &str) -> io::Result<TensorView<'_>> {
        let meta = self.ckpt.meta(name)?;
        let bytes = self.ckpt.bytes(name)?;
        let dtype = map_ckpt_dtype(name, &meta.dtype)?;
        Ok(TensorView {
            dtype,
            shape: meta.shape.iter().map(|&d| d as usize).collect(),
            bytes,
        })
    }

    fn get2(&mut self, a: &str, b: &str) -> io::Result<(TensorView<'_>, TensorView<'_>)> {
        let meta_a = self.ckpt.meta(a)?;
        let meta_b = self.ckpt.meta(b)?;
        let (bytes_a, bytes_b) = self.ckpt.bytes2(a, b)?;
        let dtype_a = map_ckpt_dtype(a, &meta_a.dtype)?;
        let dtype_b = map_ckpt_dtype(b, &meta_b.dtype)?;
        Ok((
            TensorView {
                dtype: dtype_a,
                shape: meta_a.shape.iter().map(|&d| d as usize).collect(),
                bytes: bytes_a,
            },
            TensorView {
                dtype: dtype_b,
                shape: meta_b.shape.iter().map(|&d| d as usize).collect(),
                bytes: bytes_b,
            },
        ))
    }
}

/// Scale tensors and packed-FP4 expert weights may be stored under a raw U8
/// dtype string depending on the safetensors writer; the name tells them
/// apart (convert.py keeps packed int8 bytes and e8m0 scale bytes).
fn map_ckpt_dtype(name: &str, raw: &str) -> io::Result<DType> {
    if let Some(d) = DType::from_str(raw) {
        if d == DType::U8 {
            if name.ends_with(".scale") {
                return Ok(DType::E8m0);
            }
            if name.contains(".experts.") && name.ends_with(".weight") {
                return Ok(DType::Fp4Packed);
            }
        }
        return Ok(d);
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{name}: unsupported dtype {raw}"),
    ))
}

struct StoredTensor {
    dtype: DType,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Deterministic synthetic checkpoint-shaped weights for tests and fuzzing:
/// every tensor the layout inventory names, generated from a seeded
/// splitmix64 stream and encoded into the real on-disk formats (bf16, f32,
/// fp8 + ue8m0 scales, packed fp4 + per-1x32-K ue8m0 scales). Served through
/// the same [`WeightStore`] interface, so the engine cannot tell it from a
/// real checkpoint.
pub struct SyntheticWeights {
    tensors: BTreeMap<String, StoredTensor>,
}

impl SyntheticWeights {
    /// Generate every tensor `layout::layout(cfg)` names, deterministically
    /// from `seed`. A quantized weight and its `.scale` tensor are encoded
    /// together from one value stream, so the pair is self-consistent.
    pub fn generate(cfg: &ferroseek_core::ModelConfig, seed: u64) -> io::Result<Self> {
        let rcfg = RefConfig::from_model_config(cfg)?;
        Self::generate_ref(&rcfg, seed)
    }

    pub fn generate_ref(cfg: &RefConfig, seed: u64) -> io::Result<Self> {
        let specs = layout::layout(cfg);
        let mut tensors = BTreeMap::new();
        for (name, spec) in &specs {
            if name.ends_with(".scale") {
                continue; // emitted together with its weight tensor
            }
            let scale_name = match spec.kind {
                TKind::Fp8 | TKind::Fp4 => {
                    Some(format!("{}.scale", name.trim_end_matches(".weight")))
                }
                _ => None,
            };
            if let Some(sn) = &scale_name {
                if !specs.contains_key(sn) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("layout missing scale tensor {sn}"),
                    ));
                }
            }
            let (weight, scale) = encode_pair(name, spec, seed)?;
            let (dtype, data) = weight;
            tensors.insert(
                name.clone(),
                StoredTensor {
                    dtype,
                    shape: spec.shape.clone(),
                    data,
                },
            );
            if let (Some(sn), Some((sdtype, sdata))) = (scale_name, scale) {
                let shape = specs[&sn].shape.clone();
                tensors.insert(
                    sn,
                    StoredTensor {
                        dtype: sdtype,
                        shape,
                        data: sdata,
                    },
                );
            }
        }
        Ok(Self { tensors })
    }

    /// All tensor names, sorted (debugging aid).
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Serialize as a real on-disk safetensors checkpoint (config.json +
    /// index + one shard), so gate tests can exercise the mmap/loader path
    /// real checkpoints take — including unified-memory streaming.
    pub fn write_checkpoint(&self, config_json: &str, dir: &std::path::Path) -> io::Result<()> {
        use std::collections::BTreeMap;
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("config.json"), config_json)?;
        let dt = |d: DType| match d {
            DType::Bf16 => "BF16",
            DType::F32 => "F32",
            DType::Fp8E4m3 => "F8_E4M3",
            DType::E8m0 => "F8_E8M0",
            DType::Fp4Packed => "I8", // packed int8, like convert.py
            DType::U8 => "U8",
        };
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, t) in &self.tensors {
            let start = payload.len() as u64;
            payload.extend_from_slice(&t.data);
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": dt(t.dtype), "shape": t.shape, "data_offsets": [start, payload.len() as u64]}),
            );
        }
        let hdr = serde_json::to_vec(&serde_json::Value::Object(header))?;
        let mut shard = (hdr.len() as u64).to_le_bytes().to_vec();
        shard.extend_from_slice(&hdr);
        shard.extend_from_slice(&payload);
        let shard_name = "model-00001-of-00001.safetensors";
        std::fs::write(dir.join(shard_name), &shard)?;
        let idx = serde_json::json!({
            "metadata": {"total_size": payload.len()},
            "weight_map": self
                .tensors
                .keys()
                .map(|n| (n.clone(), shard_name.to_string()))
                .collect::<BTreeMap<_, _>>()
        });
        std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&idx).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?)?;
        Ok(())
    }
}

impl WeightStore for SyntheticWeights {
    fn get(&mut self, name: &str) -> io::Result<TensorView<'_>> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, name.to_string()))?;
        Ok(TensorView {
            dtype: t.dtype,
            shape: t.shape.clone(),
            bytes: &t.data[..],
        })
    }

    fn get2(&mut self, a: &str, b: &str) -> io::Result<(TensorView<'_>, TensorView<'_>)> {
        let ta = self
            .tensors
            .get(a)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, a.to_string()))?;
        let tb = self
            .tensors
            .get(b)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, b.to_string()))?;
        Ok((
            TensorView {
                dtype: ta.dtype,
                shape: ta.shape.clone(),
                bytes: &ta.data[..],
            },
            TensorView {
                dtype: tb.dtype,
                shape: tb.shape.clone(),
                bytes: &tb.data[..],
            },
        ))
    }
}

/// fnv-1a over the name, mixed with the global seed: every tensor gets its
/// own deterministic value stream.
fn tensor_seed(seed: u64, name: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^ seed
}

struct Rng(u64);

impl Rng {
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        // 24-bit uniform from splitmix64
        let u = (splitmix64(&mut self.0) >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }
}

/// Deterministic f32 values for one tensor, BEFORE encoding. Value scale
/// rules keep the synthetic model numerically sane: matrices get
/// fan-in-scaled variance, norms stay near 1, hc bases/sinks/biases small.
fn gen_values(name: &str, spec: &TSpec, seed: u64) -> Vec<f32> {
    let numel: usize = spec.shape.iter().product();
    let cols = spec.shape.get(1).copied().unwrap_or(1);
    let mut rng = Rng(tensor_seed(seed, name));

    let norm_like =
        name.contains("norm.") || name.ends_with("q_weight") || name.ends_with("k_weight");
    if norm_like {
        return (0..numel).map(|_| rng.uniform(0.5, 1.5)).collect();
    }
    if name.ends_with("base") || name.ends_with("attn_sink") {
        return (0..numel).map(|_| rng.uniform(-0.5, 0.5)).collect();
    }
    if name.contains("gate.bias") {
        return (0..numel).map(|_| rng.uniform(-0.1, 0.1)).collect();
    }
    if spec.shape.len() == 1 {
        return (0..numel).map(|_| rng.uniform(-0.1, 0.1)).collect();
    }
    // matrix: uniform with std ~ 1/sqrt(cols)
    let bound = (3.0 / cols as f32).sqrt();
    (0..numel).map(|_| rng.uniform(-bound, bound)).collect()
}

/// Encode one tensor, and for quantized weights also its scale tensor.
/// Returns `((dtype, weight bytes), Option<(dtype, scale bytes)>)`.
fn encode_pair(
    name: &str,
    spec: &TSpec,
    seed: u64,
) -> io::Result<((DType, Vec<u8>), Option<(DType, Vec<u8>)>)> {
    let vals = gen_values(name, spec, seed);
    match spec.kind {
        TKind::Bf16 => Ok((
            (
                DType::Bf16,
                vals.iter()
                    .map(|&v| f32_to_bf16(v).to_le_bytes())
                    .flatten()
                    .collect(),
            ),
            None,
        )),
        TKind::F32 => Ok((
            (
                DType::F32,
                vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            None,
        )),
        TKind::Fp8 => {
            // engram embed rows carry per-row scales; everything else 32x32
            let per_row = name.ends_with("engram.embed.weight");
            let rows = spec.shape[0];
            let cols = spec.shape[1];
            let mut w = vec![0u8; rows * cols];
            let s_cols = cols.div_ceil(32);
            let mut scales = if per_row {
                vec![127u8; rows * s_cols]
            } else {
                vec![127u8; rows.div_ceil(32) * s_cols]
            };
            for r in 0..rows {
                for kb in 0..s_cols {
                    let mut amax = 1e-4f32;
                    for c in kb * 32..(kb * 32 + 32).min(cols) {
                        amax = amax.max(vals[r * cols + c].abs());
                    }
                    let e = log2_ceil(amax / 448.0);
                    let s = pow2i(e);
                    if per_row {
                        scales[r * s_cols + kb] = (e + 127) as u8;
                    } else {
                        scales[(r / 32) * s_cols + kb] = (e + 127) as u8;
                    }
                    for c in kb * 32..(kb * 32 + 32).min(cols) {
                        w[r * cols + c] =
                            f32_to_e4m3((vals[r * cols + c] / s).clamp(-448.0, 448.0));
                    }
                }
            }
            Ok(((DType::Fp8E4m3, w), Some((DType::E8m0, scales))))
        }
        TKind::Fp4 => {
            let rows = spec.shape[0];
            let cols = spec.shape[1];
            let s_cols = cols.div_ceil(32);
            let mut w = vec![0u8; rows * cols / 2];
            let mut scales = vec![127u8; rows * s_cols];
            for r in 0..rows {
                for kb in 0..s_cols {
                    let mut amax = 6.0 * pow2i(-126);
                    for c in kb * 32..(kb * 32 + 32).min(cols) {
                        amax = amax.max(vals[r * cols + c].abs());
                    }
                    let e = log2_ceil(amax / 6.0);
                    let s = pow2i(e);
                    scales[r * s_cols + kb] = (e + 127) as u8;
                    for c in kb * 32..(kb * 32 + 32).min(cols) {
                        let q = f32_to_e2m1((vals[r * cols + c] / s).clamp(-6.0, 6.0));
                        let idx = r * (cols / 2) + c / 2;
                        if c % 2 == 0 {
                            w[idx] = (w[idx] & 0xF0) | (q & 0x0F);
                        } else {
                            w[idx] = (w[idx] & 0x0F) | (q << 4);
                        }
                    }
                }
            }
            Ok(((DType::Fp4Packed, w), Some((DType::E8m0, scales))))
        }
        TKind::E8m0 => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "standalone e8m0 tensors are generated with their weight",
        )),
    }
}

/// The engine's view of one dense (bf16/f32) tensor, converted to f32 and
/// kept in a bounded LRU cache. Quantized tensors bypass this entirely.
pub struct DenseLru {
    map: HashMap<String, DenseEntry>,
    bytes: usize,
    budget: usize,
    clock: u64,
}

struct DenseEntry {
    data: Vec<f32>,
    clock: u64,
}

impl DenseLru {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            map: HashMap::new(),
            bytes: 0,
            budget: budget_bytes,
            clock: 0,
        }
    }

    /// f32 values of a dense tensor, converting on first use. Raw-export
    /// fp8+ue8m0 tensors (wo_a in the 1017 export) are dequantized here —
    /// convert.py does the same host-side for the reference stack.
    pub fn get(&mut self, store: &mut dyn WeightStore, name: &str) -> io::Result<&[f32]> {
        if !self.map.contains_key(name) {
            let data = match name.strip_suffix(".weight") {
                Some(prefix) => {
                    let scale_name = format!("{prefix}.scale");
                    match store.get2(name, &scale_name) {
                        Ok((w, s))
                            if w.dtype == DType::Fp8E4m3 && s.dtype == DType::E8m0 =>
                        {
                            let (rows, cols) = match w.shape.as_slice() {
                                [r, c] => (*r, *c),
                                _ => {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        format!("{name}: fp8 weight not 2-D: {:?}", w.shape),
                                    ))
                                }
                            };
                            ferroseek_core::quant::dequant_fp8_blockwise(
                                w.bytes, s.bytes, rows, cols,
                            )
                        }
                        _ => {
                            let view = store.get(name)?;
                            bytes_to_f32(&view)?
                        }
                    }
                }
                None => {
                    let view = store.get(name)?;
                    bytes_to_f32(&view)?
                }
            };
            let sz = data.len() * 4;
            if self.bytes + sz > self.budget {
                self.evict(self.bytes + sz - self.budget);
            }
            self.bytes += sz;
            self.map
                .insert(name.to_string(), DenseEntry { data, clock: 0 });
        }
        self.clock += 1;
        let clock = self.clock;
        let entry = self.map.get_mut(name).unwrap();
        entry.clock = clock;
        Ok(&entry.data[..])
    }

    fn evict(&mut self, mut needed: usize) {
        while needed > 0 && self.map.len() > 1 {
            let victim = self
                .map
                .iter()
                .min_by_key(|(_, e)| e.clock)
                .map(|(k, _)| k.clone())
                .unwrap();
            if let Some(e) = self.map.remove(&victim) {
                let sz = e.data.len() * 4;
                self.bytes -= sz;
                needed = needed.saturating_sub(sz);
            }
        }
    }
}

/// Decode bf16/f32 bytes into f32 (f32 path is a byte-identical copy).
pub fn bytes_to_f32(view: &TensorView) -> io::Result<Vec<f32>> {
    let n = view.shape.iter().product::<usize>();
    match view.dtype {
        DType::Bf16 => {
            if view.bytes.len() != n * 2 {
                return Err(bad_shape(view));
            }
            Ok(view
                .bytes
                .chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect())
        }
        DType::F32 => {
            if view.bytes.len() != n * 4 {
                return Err(bad_shape(view));
            }
            Ok(view
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "dense path cannot decode {:?} ({:?})",
                view.dtype, view.shape
            ),
        )),
    }
}

fn bad_shape(view: &TensorView) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "byte length {} does not match shape {:?}",
            view.bytes.len(),
            view.shape
        ),
    )
}

/// Read one bf16 row `r` of a `[rows, cols]` bf16 tensor as f32.
pub fn bf16_row(view: &TensorView, r: usize, cols: usize) -> io::Result<Vec<f32>> {
    debug_assert_eq!(view.dtype, DType::Bf16);
    let start = r * cols * 2;
    if start + cols * 2 > view.bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "row out of range",
        ));
    }
    Ok(view.bytes[start..start + cols * 2]
        .chunks_exact(2)
        .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect())
}

/// Dequantize one fp8 row with its ue8m0 scale row (engram embedding layout:
/// `[rows, dim]` fp8 + `[rows, dim/32]` ue8m0; spec §11).
pub fn fp8_row_scaled(w: &[u8], scales: &[u8], r: usize, cols: usize) -> Vec<f32> {
    let s_cols = cols.div_ceil(32);
    let mut out = vec![0.0f32; cols];
    for (i, o) in out.iter_mut().enumerate() {
        let s = e8m0_to_f32(scales[r * s_cols + i / 32]);
        *o = e4m3_to_f32(w[r * cols + i]) * s;
    }
    out
}
