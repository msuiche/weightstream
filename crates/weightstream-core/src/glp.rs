//! GLP (GGUF Layer Projection) spec_version 3 vectors and the fold-site op.
//!
//! `hc_fold_pre_engram`: after the fold consuming a layer's FFN write and
//! before Engram injection. The companion attention writer (`o_proj`) is
//! armed with the same stack and alpha. Other hooks, rank > 1, scaling keys,
//! or `direction.0` are refused. No WEIGHTLESS_STEER_PATH = no-op.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io;

#[derive(Debug, Clone)]
pub struct GlpVector {
    pub alpha: f32,
    pub dirs: BTreeMap<u32, Vec<f32>>,
    pub hook: String,
}

fn invalid<T>(what: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, what.into()))
}

impl GlpVector {
    pub fn from_env(
        num_layers: usize,
        hidden: usize,
        base_hint: Option<&str>,
    ) -> io::Result<Option<Self>> {
        let path = std::env::var("WEIGHTLESS_STEER_PATH").unwrap_or_default();
        let path = path.trim();
        if path.is_empty() {
            return Ok(None);
        }
        let hook =
            std::env::var("WEIGHTLESS_STEER_HOOK").unwrap_or_else(|_| "hc_fold_pre_engram".into());
        if hook != "hc_fold_pre_engram" {
            return invalid(format!(
                "WEIGHTLESS_STEER_HOOK={hook:?} is not implemented; the only site here is hc_fold_pre_engram"
            ));
        }
        let spec = std::env::var("WEIGHTLESS_STEER_LAYERS").unwrap_or_default();
        let selected: Option<std::collections::BTreeSet<u32>> = if spec.trim().is_empty() {
            None
        } else {
            Some(
                spec.replace(' ', "")
                    .split(',')
                    .filter_map(|t| t.parse().ok())
                    .collect(),
            )
        };
        let is_gguf = path.to_ascii_lowercase().ends_with(".gguf");
        let (alpha_default, dirs, hook) = if is_gguf {
            load_gguf(path, num_layers, hidden, base_hint)?
        } else {
            (1.0, load_torch_zip(path, num_layers, hidden)?, hook)
        };
        let alpha = std::env::var("WEIGHTLESS_STEER_ALPHA")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .unwrap_or(alpha_default);
        if !alpha.is_finite() {
            return invalid("steering alpha must be finite");
        }
        let mut out = BTreeMap::new();
        for (layer, mut v) in dirs {
            if let Some(sel) = &selected {
                if !sel.contains(&layer) {
                    continue;
                }
            }
            let norm = v
                .iter()
                .map(|x| (*x as f64) * (*x as f64))
                .sum::<f64>()
                .sqrt();
            if !norm.is_finite() {
                return invalid(format!("direction layer {layer} must be finite"));
            }
            for x in &mut v {
                *x = (*x as f64 / (norm + 1e-9)) as f32;
            }
            if v.iter().any(|x| !x.is_finite()) {
                return invalid(format!(
                    "direction layer {layer} is not representable in f32"
                ));
            }
            out.insert(layer, v);
        }
        if out.is_empty() {
            return invalid(format!(
                "{path}: matched no layers; refusing to run unsteered"
            ));
        }
        Ok(Some(Self {
            alpha,
            dirs: out,
            hook,
        }))
    }

    pub fn fold_dir(&self, layer_idx: usize) -> Option<&[f32]> {
        self.dirs.get(&(layer_idx as u32)).map(Vec::as_slice)
    }

    pub fn writer_dir(&self, layer_idx: usize) -> Option<&[f32]> {
        self.dirs.get(&(layer_idx as u32 + 1)).map(Vec::as_slice)
    }
}

/// CPU form of glp_project: the mean over copies is projected and the
/// rank-1 term subtracted from every copy, shifting the mean by exactly
/// `alpha * (mean . d) d`.
pub fn fold_project_cpu(stream: &mut [f32], d: &[f32], alpha: f32, hc: usize) {
    assert_eq!(stream.len(), hc * d.len());
    let hidden = d.len();
    let mut mean = vec![0.0f32; hidden];
    for c in 0..hc {
        for j in 0..hidden {
            mean[j] += stream[c * hidden + j] / hc as f32;
        }
    }
    let dot = mean.iter().zip(d).map(|(h, dv)| h * dv).sum::<f32>();
    let coef = alpha * dot;
    for c in 0..hc {
        for j in 0..hidden {
            stream[c * hidden + j] -= coef * d[j];
        }
    }
}
/// CPU form of the companion attention-writer projection. This is the same
/// rank-1 operation as the fold, applied to one already-collapsed vector.
pub fn writer_project_cpu(stream: &mut [f32], d: &[f32], alpha: f32) {
    assert_eq!(stream.len(), d.len());
    let dot = stream.iter().zip(d).map(|(x, dv)| x * dv).sum::<f32>();
    let coef = alpha * dot;
    for (x, dv) in stream.iter_mut().zip(d) {
        *x -= coef * dv;
    }
}

const HOOK: &str = "hc_fold_pre_engram";

fn le_u32(b: &[u8], p: usize) -> io::Result<u32> {
    let s = b
        .get(p..p + 4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "u32"))?;
    Ok(u32::from_le_bytes(s.try_into().unwrap()))
}
fn le_u64(b: &[u8], p: usize) -> io::Result<u64> {
    let s = b
        .get(p..p + 8)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "u64"))?;
    Ok(u64::from_le_bytes(s.try_into().unwrap()))
}
fn le_i64(b: &[u8], p: usize) -> io::Result<i64> {
    Ok(le_u64(b, p)? as i64)
}
fn le_f32(b: &[u8], p: usize) -> io::Result<f32> {
    let s = b
        .get(p..p + 4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "f32"))?;
    Ok(f32::from_le_bytes(s.try_into().unwrap()))
}
fn le_f64(b: &[u8], p: usize) -> io::Result<f64> {
    let s = b
        .get(p..p + 8)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "f64"))?;
    Ok(f64::from_le_bytes(s.try_into().unwrap()))
}
fn read_str(b: &[u8], p: &mut usize) -> io::Result<String> {
    let n = le_u64(b, *p)? as usize;
    *p += 8;
    let s = b
        .get(*p..*p + n)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "string"))?;
    *p += n;
    String::from_utf8(s.to_vec()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn load_gguf(
    path: &str,
    num_layers: usize,
    hidden: usize,
    base_hint: Option<&str>,
) -> io::Result<(f32, BTreeMap<u32, Vec<f32>>, String)> {
    let b = std::fs::read(path)?;
    if b.len() < 24 || &b[..4] != b"GGUF" {
        return invalid(format!("{path}: not a GGUF file"));
    }
    let version = le_u32(&b, 4)?;
    if version > 3 {
        return invalid(format!("{path}: GGUF version {version} unsupported"));
    }
    let n_tensors = le_u64(&b, 8)? as usize;
    let n_meta = le_u64(&b, 16)? as usize;
    let mut p = 24usize;
    let mut meta = BTreeMap::new();
    for _ in 0..n_meta {
        let key = read_str(&b, &mut p)?;
        let ty = le_u32(&b, p)?;
        p += 4;
        match ty {
            3 => {
                let v = le_u32(&b, p)?;
                p += 4;
                meta.insert(key, v.to_string());
            }
            4 => {
                let v = le_u32(&b, p)? as i32;
                p += 4;
                meta.insert(key, v.to_string());
            }
            5 => {
                let v = le_f32(&b, p)?;
                p += 4;
                meta.insert(key, v.to_string());
            }
            7 => {
                let v = read_str(&b, &mut p)?;
                meta.insert(key, v);
            }
            9 => {
                let v = le_u64(&b, p)?;
                p += 8;
                meta.insert(key, v.to_string());
            }
            10 => {
                let v = le_i64(&b, p)?;
                p += 8;
                meta.insert(key, v.to_string());
            }
            11 => {
                let v = le_f64(&b, p)?;
                p += 8;
                meta.insert(key, v.to_string());
            }
            0 | 1 | 2 | 6 => {
                let v = b[p];
                p += 1;
                meta.insert(key, v.to_string());
            }
            8 => {
                let inner = le_u32(&b, p)?;
                p += 4;
                let n = le_u64(&b, p)? as usize;
                p += 8;
                let mut vals = Vec::with_capacity(n);
                for _ in 0..n {
                    match inner {
                        7 => vals.push(read_str(&b, &mut p)?),
                        3 => {
                            let v = le_u32(&b, p)?;
                            p += 4;
                            vals.push(v.to_string());
                        }
                        9 => {
                            let v = le_u64(&b, p)?;
                            p += 8;
                            vals.push(v.to_string());
                        }
                        10 => {
                            let v = le_i64(&b, p)?;
                            p += 8;
                            vals.push(v.to_string());
                        }
                        11 => {
                            let v = le_f64(&b, p)?;
                            p += 8;
                            vals.push(v.to_string());
                        }
                        5 => {
                            let v = le_f32(&b, p)?;
                            p += 4;
                            vals.push(v.to_string());
                        }
                        _ => return invalid("unsupported metadata array element type"),
                    }
                }
                meta.insert(key, vals.join(","));
            }
            _ => return invalid(format!("{path}: unsupported metadata type {ty}")),
        }
    }
    if meta.get("glp.mode").map(String::as_str) != Some("project") {
        return invalid(format!(
            "{path}: glp.mode must be 'project' (got {:?})",
            meta.get("glp.mode")
        ));
    }
    let hook = meta.get("glp.hook_point").cloned().unwrap_or_default();
    if hook != HOOK {
        return invalid(format!(
            "{path}: glp.hook_point={hook:?} does not match {HOOK}; refusing to apply at the wrong site"
        ));
    }
    let spec = meta
        .get("glp.spec_version")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1);
    if spec < 3 {
        return invalid(format!(
            "{path}: hc fold site requires glp.spec_version 3 (file declares {spec})"
        ));
    }
    if meta.contains_key("glp.dir_scales") || meta.contains_key("glp.layer_scales") {
        return invalid(format!(
            "{path}: alpha multipliers (dir_scales/layer_scales) are not implemented by this lane"
        ));
    }
    let rank = meta
        .get("glp.rank")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1);
    if rank != 1 {
        return invalid(format!(
            "{path}: rank-k (subspace) vectors are refused; this lane implements rank 1 only"
        ));
    }
    if let Some(hint) = base_hint {
        if let Some(rev) = meta.get("general.base_model.0.version") {
            if !hint.contains(rev) {
                return invalid(format!(
                    "{path}: base checkpoint revision {rev} does not match the loaded checkpoint ({hint})"
                ));
            }
        }
    }
    let mut infos = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = read_str(&b, &mut p)?;
        let nd = le_u32(&b, p)? as usize;
        p += 4;
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(le_u64(&b, p)? as usize);
            p += 8;
        }
        let dtype = le_u32(&b, p)?;
        p += 4;
        let off = le_u64(&b, p)? as usize;
        p += 8;
        infos.push((name, dims, dtype, off));
    }
    let data_start = (p + 31) & !31;
    let mut hash = Sha256::new();
    let mut dirs = BTreeMap::new();
    for (name, dims, dtype, off) in infos {
        if !name.starts_with("direction.") {
            continue;
        }
        let layer: u32 = name["direction.".len()..].parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad direction name {name}"),
            )
        })?;
        if layer == 0 {
            return invalid(format!("{path}: direction.0 is invalid at an hc fold site"));
        }
        if layer as usize >= num_layers {
            return invalid(format!(
                "{path}: direction layer {layer} out of range for this model ({num_layers} layers)"
            ));
        }
        if dtype != 0 {
            return invalid(format!("{path}: {name} must be F32 (dtype {dtype})"));
        }
        if dims.len() != 1 || dims[0] as u64 != hidden as u64 {
            return invalid(format!("{path}: {name} shape {dims:?} != [{hidden}]"));
        }
        let start = data_start + off;
        let end = start + hidden * 4;
        let bytes = b
            .get(start..end)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, name.clone()))?;
        hash.update(bytes);
        let v = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>();
        dirs.insert(layer, v);
    }
    if dirs.is_empty() {
        return invalid(format!("{path}: no direction tensors"));
    }
    if let Some(expected) = meta.get("glp.content_sha256") {
        let actual = format!("{:x}", hash.finalize());
        if &actual != expected {
            return invalid(format!(
                "{path}: content_sha256 mismatch (computed {actual}, file declares {expected})"
            ));
        }
    }
    let alpha = meta
        .get("glp.alpha_default")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    Ok((alpha, dirs, HOOK.to_string()))
}

fn zip_entry_bytes<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    suffix: &str,
) -> io::Result<Vec<u8>> {
    let name = archive
        .file_names()
        .find(|n| n.ends_with(suffix))
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("zip entry {suffix}")))?
        .to_string();
    let mut f = archive
        .by_name(&name)
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e))?;
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut f, &mut out)?;
    Ok(out)
}

fn z_u16(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes(b[p..p + 2].try_into().unwrap())
}
fn z_u32(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}

fn p_global(b: &[u8], i: &mut usize) -> io::Result<String> {
    if b[*i] == b'h' {
        *i += 2;
        return Ok(String::new());
    }
    if b[*i] == b'j' {
        *i += 5;
        return Ok(String::new());
    }
    if b[*i] != b'c' {
        return invalid(format!(
            "pickle: expected GLOBAL or BINGET at {}, got {:#x}",
            *i, b[*i]
        ));
    }
    let m_start = *i + 1;
    let m_end = b[m_start..]
        .iter()
        .position(|&c| c == b'\n')
        .map(|p| m_start + p)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "pickle: unterminated module"))?;
    let module = String::from_utf8(b[m_start..m_end].to_vec())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let n_start = m_end + 1;
    let n_end = b[n_start..]
        .iter()
        .position(|&c| c == b'\n')
        .map(|p| n_start + p)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "pickle: unterminated name"))?;
    let name = String::from_utf8(b[n_start..n_end].to_vec())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    *i = n_end + 1;
    Ok(format!("{module} {name}"))
}
fn p_unicode(b: &[u8], i: &mut usize) -> io::Result<String> {
    if b[*i] == b'h' {
        *i += 2;
        return Ok(String::new());
    }
    if b[*i] == b'j' {
        *i += 5;
        return Ok(String::new());
    }
    if b[*i] != b'X' {
        return invalid("pickle: expected BINUNICODE or BINGET");
    }
    let n = z_u32(b, *i + 1) as usize;
    let s = String::from_utf8(b[*i + 5..*i + 5 + n].to_vec())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    *i += 5 + n;
    Ok(s)
}
fn p_binint2(b: &[u8], i: &mut usize) -> io::Result<u32> {
    if b[*i] != b'M' {
        return invalid(format!(
            "pickle: expected BININT2 at {}, got {:#x}",
            *i, b[*i]
        ));
    }
    let v = z_u16(b, *i + 1) as u32;
    *i += 3;
    Ok(v)
}
fn p_binput1(b: &[u8], i: &mut usize) -> io::Result<()> {
    match b[*i] {
        b'q' => *i += 2,
        b'r' => *i += 5,
        _ => {}
    }
    Ok(())
}
fn p_memo_any(b: &[u8], i: &mut usize) -> io::Result<()> {
    match b[*i] {
        b'q' | b'h' => *i += 2,
        b'j' | b'r' => *i += 5,
        _ => {}
    }
    Ok(())
}
fn p_expect(b: &[u8], i: &mut usize, op: u8) -> io::Result<()> {
    if b[*i] != op {
        return invalid(format!(
            "pickle: expected opcode {op:#x} at {}, got {:#x}",
            *i, b[*i]
        ));
    }
    *i += 1;
    Ok(())
}

fn load_torch_zip(
    path: &str,
    num_layers: usize,
    hidden: usize,
) -> io::Result<BTreeMap<u32, Vec<f32>>> {
    let b = std::fs::read(path)?;
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(b))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let pkl = zip_entry_bytes(&mut archive, "data.pkl")?;
    let pkl = &pkl;
    let mut dirs = BTreeMap::new();
    let mut i = 6usize;
    while i < pkl.len() {
        if pkl[i] == b'u' || pkl[i] == b'.' {
            break;
        }
        let layer = match pkl[i] {
            b'K' => {
                let v = pkl[i + 1] as u32;
                i += 2;
                v
            }
            b'M' => {
                let v = z_u16(pkl, i + 1) as u32;
                i += 3;
                v
            }
            op => return invalid(format!("pickle: expected layer id, got {op:#x}")),
        };
        let rebuild = p_global(pkl, &mut i)?;
        if !rebuild.is_empty() && rebuild != "torch._utils _rebuild_tensor_v2" {
            return invalid("pickle: unexpected tensor rebuild symbol");
        }
        p_binput1(pkl, &mut i)?;
        p_expect(pkl, &mut i, b'(')?;
        p_expect(pkl, &mut i, b'(')?;
        let storage = p_unicode(pkl, &mut i)?;
        if !storage.is_empty() && storage != "storage" {
            return invalid("pickle: expected storage key");
        }
        p_binput1(pkl, &mut i)?;
        let float_storage = p_global(pkl, &mut i)?;
        if !float_storage.is_empty() && float_storage != "torch FloatStorage" {
            return invalid("pickle: expected FloatStorage");
        }
        p_binput1(pkl, &mut i)?;
        let sid = p_unicode(pkl, &mut i)?;
        p_binput1(pkl, &mut i)?;
        let cpu = p_unicode(pkl, &mut i)?;
        if !cpu.is_empty() && cpu != "cpu" {
            return invalid("pickle: expected cpu storage");
        }
        p_binput1(pkl, &mut i)?;
        let n1 = p_binint2(pkl, &mut i)? as usize;
        p_expect(pkl, &mut i, b't')?;
        p_binput1(pkl, &mut i)?;
        p_expect(pkl, &mut i, b'Q')?;
        p_expect(pkl, &mut i, b'K')?;
        p_expect(pkl, &mut i, 0)?;
        let n2 = p_binint2(pkl, &mut i)? as usize;
        if n1 != n2 || n1 != hidden {
            return invalid(format!(
                "{path}: direction layer {layer} width {n2} != {hidden}"
            ));
        }
        p_expect(pkl, &mut i, 0x85)?;
        p_binput1(pkl, &mut i)?;
        p_expect(pkl, &mut i, b'K')?;
        p_expect(pkl, &mut i, 1)?;
        p_expect(pkl, &mut i, 0x85)?;
        p_binput1(pkl, &mut i)?;
        p_expect(pkl, &mut i, 0x89)?;
        let ordered_dict = p_global(pkl, &mut i)?;
        if !ordered_dict.is_empty() && ordered_dict != "collections OrderedDict" {
            return invalid("pickle: expected OrderedDict");
        }
        p_binput1(pkl, &mut i)?;
        p_expect(pkl, &mut i, b')')?;
        p_expect(pkl, &mut i, b'R')?;
        p_memo_any(pkl, &mut i)?;
        p_expect(pkl, &mut i, b't')?;
        p_memo_any(pkl, &mut i)?;
        p_expect(pkl, &mut i, b'R')?;
        p_memo_any(pkl, &mut i)?;
        if layer == 0 {
            return invalid(format!("{path}: direction.0 is invalid at an hc fold site"));
        }
        if layer as usize >= num_layers {
            return invalid(format!(
                "{path}: direction layer {layer} out of range ({num_layers})"
            ));
        }
        let v = zip_entry_bytes(&mut archive, &format!("/data/{sid}"))?;
        if v.len() != hidden * 4 {
            return invalid(format!(
                "{path}: storage {sid} has {} bytes, expected {}",
                v.len(),
                hidden * 4
            ));
        }
        let v = v
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>();
        dirs.insert(layer, v);
    }
    if dirs.is_empty() {
        return invalid(format!("{path}: no direction tensors"));
    }
    Ok(dirs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zip_store(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, data) in entries {
            w.start_file(name, opts).unwrap();
            std::io::Write::write_all(&mut w, data).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    fn one_tensor_pt() -> Vec<u8> {
        // The first tensor record of the real GLP-27 .pt file, verbatim.
        let pkl: &[u8] = &[
            0x80, 0x02, 0x7d, 0x71, 0x00, 0x28, 0x4b, 0x01, 0x63, 0x74, 0x6f, 0x72, 0x63, 0x68,
            0x2e, 0x5f, 0x75, 0x74, 0x69, 0x6c, 0x73, 0x0a, 0x5f, 0x72, 0x65, 0x62, 0x75, 0x69,
            0x6c, 0x64, 0x5f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x5f, 0x76, 0x32, 0x0a, 0x71,
            0x01, 0x28, 0x28, 0x58, 0x07, 0x00, 0x00, 0x00, 0x73, 0x74, 0x6f, 0x72, 0x61, 0x67,
            0x65, 0x71, 0x02, 0x63, 0x74, 0x6f, 0x72, 0x63, 0x68, 0x0a, 0x46, 0x6c, 0x6f, 0x61,
            0x74, 0x53, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x0a, 0x71, 0x03, 0x58, 0x01, 0x00,
            0x00, 0x00, 0x30, 0x71, 0x04, 0x58, 0x03, 0x00, 0x00, 0x00, 0x63, 0x70, 0x75, 0x71,
            0x05, 0x4d, 0x00, 0x14, 0x74, 0x71, 0x06, 0x51, 0x4b, 0x00, 0x4d, 0x00, 0x14, 0x85,
            0x71, 0x07, 0x4b, 0x01, 0x85, 0x71, 0x08, 0x89, 0x63, 0x63, 0x6f, 0x6c, 0x6c, 0x65,
            0x63, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x0a, 0x4f, 0x72, 0x64, 0x65, 0x72, 0x65, 0x64,
            0x44, 0x69, 0x63, 0x74, 0x0a, 0x71, 0x09, 0x29, 0x52, 0x71, 0x0a, 0x74, 0x71, 0x0b,
            0x52, 0x71, 0x0c,
        ];
        let bytes = vec![0.25f32; 5120]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        zip_store(&[
            ("dsv41-perlayer-dirs-ffn/data.pkl", pkl),
            ("dsv41-perlayer-dirs-ffn/data/0", &bytes),
        ])
    }

    #[test]
    fn torch_zip_loads() {
        let dir = std::env::temp_dir().join("weightstream-glp-test.pt");
        std::fs::write(&dir, one_tensor_pt()).unwrap();
        let dirs = load_torch_zip(dir.to_str().unwrap(), 40, 5120).unwrap();
        let v = &dirs[&1];
        assert_eq!(v.len(), 5120);
        assert!(v.iter().all(|&x| (x - 0.25).abs() < 1e-6));
    }

    #[test]
    fn fold_mean_shifts_exactly() {
        let hc = 4;
        let hidden = 3;
        let d = [0.5, -1.0, 2.0];
        let mut stream = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let before = stream.clone();
        let alpha = 0.75;
        fold_project_cpu(&mut stream, &d, alpha, hc);
        let mean_before: Vec<f32> = (0..hidden)
            .map(|j| (0..hc).map(|c| before[c * hidden + j]).sum::<f32>() / hc as f32)
            .collect();
        let dot = mean_before.iter().zip(d).map(|(h, dv)| h * dv).sum::<f32>();
        for j in 0..hidden {
            let expected = mean_before[j] - alpha * dot * d[j];
            let got = (0..hc).map(|c| stream[c * hidden + j]).sum::<f32>() / hc as f32;
            assert!(
                (got - expected).abs() < 1e-6,
                "mean at {j}: {got} != {expected}"
            );
        }
    }

    #[test]
    fn glp27_pt_loads_if_configured() {
        let Ok(path) = std::env::var("WEIGHTSTREAM_GLP27_PT") else {
            return;
        };
        let dirs = load_torch_zip(&path, 40, 5120).unwrap();
        assert_eq!(dirs.len(), 39);
        assert!(dirs.contains_key(&1) && dirs.contains_key(&39));
        assert!(dirs.values().all(|v| v.len() == 5120));
    }
}
