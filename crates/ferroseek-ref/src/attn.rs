//! MLA attention over window + compressed KV (spec §3, §4, §5;
//! model.py:614-791), including the DSA indexer, the CSA compressor and the
//! sparse online-softmax attention kernel semantics (kernel.py:311-403).

use crate::actquant::{fp4_snap_e4m3, fp4_snap_e8m0, fp8_gemv, fp8_snap};
use crate::config::RefConfig;
use crate::model::{Ctx, LayerInfo, Runtime, SourceState};
use crate::num;
use crate::store::bytes_to_f32;
use std::io;

/// Sliding-window indices for one decode query (model.py:409-426): the whole
/// ring, oldest first, with `-1` for slots that hold nothing yet.
///
/// Public: the GPU engine's host glue selects indices for the same ring
/// layout.
pub fn window_topk_idxs(window: usize, pos: u64) -> Vec<i32> {
    let oldest = (pos % window as u64 + 1) as usize;
    let mut idxs: Vec<i32> = (oldest..window)
        .chain(0..oldest)
        .map(|i| i as i32)
        .collect();
    for i in idxs.iter_mut() {
        if *i as u64 > pos {
            *i = -1;
        }
    }
    idxs
}

/// Sparse attention with the reference kernel's exact semantics
/// (kernel.py:311-403): online softmax in blocks of 64 over gathered rows,
/// finite `-1e30` max lower bound, the attention sink added as one extra
/// key with a zero value, invalid indices contributing zero. Divergences
/// are documented in the crate docs (f32 instead of the kernel's bf16
/// P·V operands).
pub fn sparse_attn(
    q: &[f32],
    n_heads: usize,
    head_dim: usize,
    kv: &[f32],
    idxs: &[i32],
    sink: &[f32],
    scale: f32,
) -> Vec<f32> {
    let row = |i: i32| -> &[f32] {
        debug_assert!(i >= 0);
        &kv[i as usize * head_dim..(i as usize + 1) * head_dim]
    };
    let mut out = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut m = -1e30f32;
        let mut sum = 0.0f32;
        let mut acc = vec![0.0f32; head_dim];
        for chunk in idxs.chunks(64) {
            // logits for this chunk; -1 indices are invalid
            let mut chunk_max = m;
            let mut logits = vec![f32::NEG_INFINITY; chunk.len()];
            for (j, &i) in chunk.iter().enumerate() {
                if i >= 0 {
                    let l = num::dot(qh, row(i)) * scale;
                    logits[j] = l;
                    if l > chunk_max {
                        chunk_max = l;
                    }
                }
            }
            let rescale = (m - chunk_max).exp();
            m = chunk_max;
            // rescale the running accumulator FIRST, then add this block
            // (kernel.py:377-380)
            for v in acc.iter_mut() {
                *v *= rescale;
            }
            let mut chunk_sum = 0.0f32;
            for (j, &l) in logits.iter().enumerate() {
                let p = (l - chunk_max).exp();
                chunk_sum += p;
                if p != 0.0 && chunk[j] >= 0 {
                    let r = row(chunk[j]);
                    for d in 0..head_dim {
                        acc[d] += p * r[d];
                    }
                }
            }
            sum = sum * rescale + chunk_sum;
        }
        // sink: one extra key with logit attn_sink[h] and a zero value
        sum += (sink[h] - m).exp();
        for d in 0..head_dim {
            out[h * head_dim + d] = acc[d] / sum;
        }
    }
    out
}

/// One attention sublayer for layer `li` at position `pos`. `x` is the
/// normed sublayer input. Returns the projected output `[dim]`.
pub fn attention(
    cfg: &RefConfig,
    li: &LayerInfo,
    ctx: &mut Ctx<'_>,
    rt: &mut Runtime,
    x: &[f32],
    pos: u64,
) -> io::Result<Vec<f32>> {
    let l = li.layer;
    let hd = cfg.head_dim as usize;
    let heads = cfg.n_heads as usize;
    let dim = cfg.dim as usize;
    let win = cfg.window_size as usize;
    let freqs = &li.freqs;
    if rt.capture_x.is_some() {
        rt.capture_x.as_mut().unwrap().push((l, x.to_vec()));
    }

    // query path: q_a -> q_norm -> q_b, rope tail (spec §3, model.py:771-773)
    let qr = {
        let (w, s) = ctx.get2(
            &format!("layers.{l}.attn.wq_a.weight"),
            &format!("layers.{l}.attn.wq_a.scale"),
        )?;
        let q0 = fp8_gemv(w.bytes, s.bytes, cfg.q_lora as usize, dim, x);
        let nrm = ctx.dense(&format!("layers.{l}.attn.q_norm.weight"))?;
        num::rmsnorm(&q0, nrm, cfg.norm_eps)
    };
    let mut q = {
        let (w, s) = ctx.get2(
            &format!("layers.{l}.attn.wq_b.weight"),
            &format!("layers.{l}.attn.wq_b.scale"),
        )?;
        fp8_gemv(w.bytes, s.bytes, heads * hd, cfg.q_lora as usize, &qr)
    };
    freqs.apply_heads(&mut q, hd, pos, false);
    crate::dump::f32s(format!("d_pos{pos}_qr{l}.f32"), &qr);
    crate::dump::f32s(format!("d_pos{pos}_q{l}.f32"), &q);

    // window KV: one shared latent per token, rope tail, fp8 snap, ring write
    let mut kv = {
        let (w, s) = ctx.get2(
            &format!("layers.{l}.attn.wkv.weight"),
            &format!("layers.{l}.attn.wkv.scale"),
        )?;
        let k0 = fp8_gemv(w.bytes, s.bytes, hd, dim, x);
        let nrm = ctx.dense(&format!("layers.{l}.attn.kv_norm.weight"))?;
        num::rmsnorm(&k0, nrm, cfg.norm_eps)
    };
    freqs.apply_tail(&mut kv, pos, false);
    fp8_snap(&mut kv, 32);
    {
        let ring = &mut rt.window[l as usize];
        let slot = (pos % win as u64) as usize;
        ring[slot * hd..(slot + 1) * hd].copy_from_slice(&kv);
    }

    let mut kv_all = rt.window[l as usize].clone();
    let mut idxs = window_topk_idxs(win, pos);

    if li.ratio > 0 {
        let ratio = li.ratio as u64;
        let compress_len = ((pos + 1) / ratio) as usize;

        // CSA compressor (spec §5): kv sources produce the pre-RoPE latent
        // (None while a group is still open)
        let latent = match li.kv_src {
            Some(src) => {
                rt.compress_slot = src;
                rt.index_k_slot = src;
                let state = &mut rt.sources[src];
                compressor(cfg, ctx, state, x, pos, l, li.ratio)?
            }
            None => None,
        };

        // top-k selection (spec §4)
        let cmp_idxs = if li.idx_src {
            let idxs = indexer(cfg, li, ctx, rt, &qr, x, latent.as_deref(), pos)?;
            rt.topk_idxs = idxs.clone();
            idxs
        } else {
            rt.topk_idxs.clone()
        };

        // post-RoPE fp4(16, e4m3) cache write (model.py:752-762)
        if let (Some(src), Some(lat)) = (li.kv_src, &latent) {
            let mut row = lat.clone();
            freqs.apply_tail(&mut row, pos + 1 - ratio, false);
            fp4_snap_e4m3(&mut row, 16);
            rt.sources[src].compress.extend_from_slice(&row);
        }

        // read the shared published cache (model.py:764)
        let src = rt.compress_slot;
        let rows = &rt.sources[src].compress[..compress_len * hd];
        kv_all.extend_from_slice(rows);
        if let Some(lat) = &latent {
            crate::dump::f32s(format!("d_pos{pos}_latent{l}.f32"), lat);
        }
        crate::dump::i32s(format!("d_pos{pos}_cmpidxs{l}.i32"), &cmp_idxs);
        idxs.extend(cmp_idxs);
    }

    // attention itself (kernel.py:311-403)
    let sink = ctx.dense(&format!("layers.{l}.attn.attn_sink"))?.to_vec();
    crate::dump::f32s(format!("d_pos{pos}_kvall{l}.f32"), &kv_all);
    crate::dump::i32s(format!("d_pos{pos}_idxs{l}.i32"), &idxs);
    let mut o = sparse_attn(&q, heads, hd, &kv_all, &idxs, &sink, num::rsqrt(hd as f32));
    crate::dump::f32s(format!("d_pos{pos}_attnraw{l}.f32"), &o);

    // inverse rope on the output tail at the query position (spec §3)
    freqs.apply_heads(&mut o, hd, pos, true);
    crate::dump::f32s(format!("d_pos{pos}_osnap{l}.f32"), &o);

    // grouped output projection: fp8 snap per group row, block-diagonal wo_a,
    // then wo_b (model.py:785-791)
    let og = cfg.o_groups as usize;
    let ol = cfg.o_lora as usize;
    let gdim = heads * hd / og;
    debug_assert_eq!(gdim % 32, 0, "wo_a group row must be 32-divisible");
    for g in 0..og {
        fp8_snap(&mut o[g * gdim..(g + 1) * gdim], 32);
    }
    crate::dump::f32s(format!("d_pos{pos}_osnapS{l}.f32"), &o);
    let mut fl = vec![0.0f32; og * ol];
    {
        // block-diagonal einsum: group g projects only its own heads
        let wo_a = ctx.dense(&format!("layers.{l}.attn.wo_a.weight"))?;
        for g in 0..og {
            let orow = &o[g * gdim..(g + 1) * gdim];
            for r in 0..ol {
                let wrow = &wo_a[(g * ol + r) * gdim..(g * ol + r + 1) * gdim];
                fl[g * ol + r] = num::dot(orow, wrow);
            }
        }
    }
    crate::dump::f32s(format!("d_pos{pos}_fl{l}.f32"), &fl);
    let out = {
        let (w, s) = ctx.get2(
            &format!("layers.{l}.attn.wo_b.weight"),
            &format!("layers.{l}.attn.wo_b.scale"),
        )?;
        if std::env::var_os("FERROSEEK_DUMP").is_some() {
            let qa = crate::actquant::fp8_act_values(&fl, 32);
            crate::dump::f32s(format!("d_pos{pos}_actflQA{l}.f32"), &qa);
        }
        fp8_gemv(w.bytes, s.bytes, dim, og * ol, &fl)
    };
    crate::dump::f32s(format!("d_pos{pos}_attnout{l}.f32"), &out);
    Ok(out)
}

/// CSA compressor (spec §5, model.py:429-485). Returns the pre-RoPE latent
/// when a compression group just completed, parking partial groups in the
/// fp32 open-pair state buffers; `None` while a group is still open. Ratio 1
/// is a plain projection (one latent per token).
fn compressor(
    cfg: &RefConfig,
    ctx: &mut Ctx<'_>,
    state: &mut SourceState,
    x: &[f32],
    pos: u64,
    layer: u32,
    ratio: u32,
) -> io::Result<Option<Vec<f32>>> {
    let hd = cfg.head_dim as usize;
    let dim = cfg.dim as usize;
    if ratio == 1 {
        // plain projection, bf16 weights, no gate, no state (model.py:461-462)
        let norm = ctx
            .dense(&format!("layers.{layer}.attn.compressor.norm.weight"))?
            .to_vec();
        let lat = {
            let w = {
                let t = ctx
                    .store
                    .get(&format!("layers.{layer}.attn.compressor.wkv.weight"))?;
                bytes_to_f32(&t)?
            };
            num::gemv(&w, hd, dim, x)
        };
        return Ok(Some(num::rmsnorm(&lat, &norm, cfg.norm_eps)));
    }
    // ratio > 1: fp32 weights, softmax pooling over the open group
    let slot = (pos % ratio as u64) as usize;
    {
        let kv = {
            let w = {
                let t = ctx
                    .store
                    .get(&format!("layers.{layer}.attn.compressor.wkv.weight"))?;
                bytes_to_f32(&t)?
            };
            num::gemv(&w, hd, dim, x)
        };
        state.kv_state.as_mut().unwrap()[slot * hd..(slot + 1) * hd].copy_from_slice(&kv);
    }
    {
        let score = {
            let g = {
                let t = ctx
                    .store
                    .get(&format!("layers.{layer}.attn.compressor.wgate.weight"))?;
                bytes_to_f32(&t)?
            };
            num::gemv(&g, hd, dim, x)
        };
        state.score_state.as_mut().unwrap()[slot * hd..(slot + 1) * hd].copy_from_slice(&score);
    }
    if (pos + 1) % ratio as u64 != 0 {
        return Ok(None); // open pair: nothing appended this step
    }
    // softmax over the slots, per dim (model.py:482); latent stands for the
    // group's first token
    let norm = ctx
        .dense(&format!("layers.{layer}.attn.compressor.norm.weight"))?
        .to_vec();
    let kv_state = state.kv_state.as_ref().unwrap();
    let score_state = state.score_state.as_mut().unwrap();
    let mut pooled = vec![0.0f32; hd];
    for d in 0..hd {
        let mut scores = vec![0.0f32; ratio as usize];
        for s in 0..ratio as usize {
            scores[s] = score_state[s * hd + d];
        }
        num::softmax(&mut scores);
        for s in 0..ratio as usize {
            pooled[d] += scores[s] * kv_state[s * hd + d];
        }
    }
    Ok(Some(num::rmsnorm(&pooled, &norm, cfg.norm_eps)))
}

/// DSA indexer (spec §4, model.py:488-611): fp4-snapped q/k, relu +
/// weights_proj scoring, two-level top-k. Returns the compressed indices
/// (offset by the window size; -1 unreachable) and publishes `index_k`
/// and, on the candidate source, the candidate mask.
fn indexer(
    cfg: &RefConfig,
    li: &LayerInfo,
    ctx: &mut Ctx<'_>,
    rt: &mut Runtime,
    qr: &[f32],
    x: &[f32],
    latent: Option<&[f32]>,
    pos: u64,
) -> io::Result<Vec<i32>> {
    let l = li.layer;
    let inh = cfg.index_n_heads as usize;
    let ihd = cfg.index_head_dim as usize;
    let ratio = li.ratio as u64;
    let freqs = &li.freqs;
    let hd = cfg.head_dim as usize;

    // key side: on group completion, k = k_norm(wk(latent)), rope tail at the
    // group's first-token position, fp4(32, e8m0) snap (model.py:536-547)
    if li.owns_k {
        if let Some(lat) = latent {
            let k = {
                let w = {
                    let t = ctx
                        .store
                        .get(&format!("layers.{l}.attn.indexer.wk.weight"))?;
                    bytes_to_f32(&t)?
                };
                let nrm = {
                    let t = ctx
                        .store
                        .get(&format!("layers.{l}.attn.indexer.k_norm.weight"))?;
                    bytes_to_f32(&t)?
                };
                let k0 = num::gemv(&w, ihd, hd, lat);
                num::rmsnorm(&k0, &nrm, cfg.norm_eps)
            };
            let mut k = k;
            freqs.apply_tail(&mut k, pos + 1 - ratio, false);
            fp4_snap_e8m0(&mut k, 32);
            rt.sources[li.kv_src.unwrap()].index_k.extend_from_slice(&k);
        }
    }

    // query side: wq_b(qr), rope tail at pos, fp4(32, e8m0) snap
    let mut q = {
        let (w, s) = ctx.get2(
            &format!("layers.{l}.attn.indexer.wq_b.weight"),
            &format!("layers.{l}.attn.indexer.wq_b.scale"),
        )?;
        fp8_gemv(w.bytes, s.bytes, inh * ihd, cfg.q_lora as usize, qr)
    };
    freqs.apply_heads(&mut q, ihd, pos, false);
    fp4_snap_e8m0(&mut q, 32);

    // scoring: relu then per-head weights (model.py:555-560)
    let compress_len = ((pos + 1) / ratio) as usize;
    let weights = {
        let w = {
            let t = ctx
                .store
                .get(&format!("layers.{l}.attn.indexer.weights_proj.weight"))?;
            bytes_to_f32(&t)?
        };
        let mut wv = num::gemv(&w, inh, cfg.dim as usize, x);
        let s = num::rsqrt(ihd as f32) * num::rsqrt(inh as f32);
        for v in wv.iter_mut() {
            *v *= s;
        }
        wv
    };
    let mut score = vec![0.0f32; compress_len];
    {
        let kcache = &rt.sources[rt.index_k_slot].index_k;
        debug_assert_eq!(kcache.len(), compress_len * ihd);
        for t in 0..compress_len {
            let krow = &kcache[t * ihd..(t + 1) * ihd];
            for h in 0..inh {
                let s = num::dot(&q[h * ihd..(h + 1) * ihd], krow);
                if s > 0.0 {
                    score[t] += s * weights[h];
                }
            }
        }
    }

    // two-level top-k (model.py:570-581)
    if li.cand_src {
        rt.candidates = select_candidate_blocks(
            &score,
            cfg.candidate_topk_blocks as usize,
            cfg.candidate_block_size as usize,
        );
    } else if li.uses_cand {
        for (t, s) in score.iter_mut().enumerate() {
            if !rt.candidates.get(t).copied().unwrap_or(false) {
                *s = f32::NEG_INFINITY;
            }
        }
    }

    let topk = (cfg.index_topk as usize).min(compress_len);
    Ok(select_topk(
        &score,
        topk,
        compress_len,
        cfg.window_size as i32,
    ))
}

/// Decode-path top-k over indexer scores (model.py:579-581): deterministic
/// (value desc, index asc) selection (spec §12.6), re-sorted into position
/// order, shifted by the window offset with `-1` for unreachable entries.
/// Public: the GPU engine runs this selection host-side for v1.
pub fn select_topk(score: &[f32], topk: usize, compress_len: usize, offset: i32) -> Vec<i32> {
    let mut order: Vec<usize> = (0..compress_len).collect();
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
    order.truncate(topk);
    order.sort_unstable(); // re-sorted into position order (model.py:580)
    order
        .into_iter()
        .map(|i| {
            if i < compress_len {
                (i as i32) + offset
            } else {
                -1
            }
        })
        .collect()
}

/// Level-one top-k: keep the `topk_blocks` best candidate blocks, pinning the
/// block holding the newest position (select_candidate_blocks,
/// model.py:584-611). Public: the GPU engine runs level-one host-side.
pub fn select_candidate_blocks(score: &[f32], topk_blocks: usize, block_size: usize) -> Vec<bool> {
    let width = score.len();
    if width == 0 || topk_blocks == 0 || block_size == 0 {
        return vec![false; width];
    }
    let n_blocks = width.div_ceil(block_size);
    let mut block_score = vec![f32::NEG_INFINITY; n_blocks];
    for (b, s) in block_score.iter_mut().enumerate() {
        let lo = b * block_size;
        let hi = (lo + block_size).min(width);
        *s = score[lo..hi]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
    }
    // pin the block containing the newest position
    let last = (width - 1) / block_size;
    block_score[last] = f32::INFINITY;
    let mut order: Vec<usize> = (0..n_blocks).collect();
    order.sort_by(|&a, &b| block_score[b].total_cmp(&block_score[a]).then(a.cmp(&b)));
    order.truncate(topk_blocks.min(n_blocks));
    let mut keep = vec![false; n_blocks];
    for b in order {
        if block_score[b] > f32::NEG_INFINITY {
            keep[b] = true;
        }
    }
    let mut mask = vec![false; width];
    for (t, m) in mask.iter_mut().enumerate() {
        *m = keep[t / block_size];
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_idxs_wrap_and_mark_empty() {
        // p=3, win=8: slots 4..7 empty (-1), oldest-first then 0..3
        let want: Vec<i32> = [4, 5, 6, 7, 0, 1, 2, 3]
            .iter()
            .map(|&v| if v > 3 { -1 } else { v })
            .collect();
        assert_eq!(window_topk_idxs(8, 3), want);
        // first token: only slot 0 valid
        assert_eq!(window_topk_idxs(8, 0), vec![-1, -1, -1, -1, -1, -1, -1, 0]);
        // after wrap: p=9, win=8 -> slot 1 holds pos 9, all slots valid
        assert_eq!(window_topk_idxs(8, 9), vec![2, 3, 4, 5, 6, 7, 0, 1]);
    }

    #[test]
    fn sink_adds_extra_key() {
        // single valid key identical to q: with a very negative sink its
        // weight vanishes and the output equals the key; with a huge sink the
        // zero "sink value" dominates and the output collapses to zero
        let q = vec![1.0f32; 4];
        let kv = vec![1.0f32; 4];
        let no_sink = sparse_attn(&q, 1, 4, &kv, &[0], &[-100.0], 0.5);
        assert!(
            no_sink.iter().all(|&v| (v - 1.0).abs() < 1e-6),
            "{no_sink:?}"
        );
        let big_sink = sparse_attn(&q, 1, 4, &kv, &[0], &[100.0], 0.5);
        assert!(big_sink.iter().all(|&v| v.abs() < 1e-6), "{big_sink:?}");
    }

    #[test]
    fn all_invalid_rows_yield_zero_output() {
        let q = vec![1.0f32; 4];
        let kv = vec![1.0f32; 4];
        let o = sparse_attn(&q, 1, 4, &kv, &[-1, -1], &[0.0], 0.5);
        assert!(o.iter().all(|&v| v == 0.0), "{o:?}");
    }

    #[test]
    fn candidate_blocks_pin_newest() {
        // 8 positions, block 2, topk 2: newest block pinned + best other
        let score = vec![9.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mask = select_candidate_blocks(&score, 2, 2);
        // block 3 (newest, pos 6-7) pinned; block 0 (score 9) kept
        assert_eq!(
            mask,
            vec![true, true, false, false, false, false, true, true]
        );
        // topk 1: only the pinned block
        let mask = select_candidate_blocks(&score, 1, 2);
        assert_eq!(
            mask,
            vec![false, false, false, false, false, false, true, true]
        );
    }
}
