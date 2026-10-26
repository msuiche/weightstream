//! Tensor inventory: every text-side tensor name the engine reads, with its
//! on-disk dtype and shape. One source of truth used by (a) the engine, (b)
//! checkpoint validation in `RefModel::from_checkpoint`, and (c)
//! [`crate::store::SyntheticWeights`] generation. Naming follows the
//! checkpoint census (`ferroseek_core::weights`) minus the mtp.* drafter
//! (stubbed, `TODO(spec-decode)`) and vision tensors (out of scope).

use crate::config::RefConfig;
use std::collections::BTreeMap;

/// How a tensor is stored (spec §11 load plan).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TKind {
    Bf16,
    F32,
    /// FP8 E4M3 with a `.scale` UE8M0 companion.
    Fp8,
    /// Packed FP4 E2M1 with a per-1x32-K `.scale` UE8M0 companion.
    Fp4,
    /// UE8M0 scale bytes.
    E8m0,
}

#[derive(Clone, Debug)]
pub struct TSpec {
    pub kind: TKind,
    pub shape: Vec<usize>,
}

fn spec(kind: TKind, shape: Vec<usize>) -> TSpec {
    TSpec { kind, shape }
}

/// The full inventory for a config (spec §11).
pub fn layout(cfg: &RefConfig) -> BTreeMap<String, TSpec> {
    let mut m: BTreeMap<String, TSpec> = BTreeMap::new();
    let dim = cfg.dim as usize;
    let vocab = cfg.vocab as usize;
    let hd = cfg.head_dim as usize;
    let ql = cfg.q_lora as usize;
    let ol = cfg.o_lora as usize;
    let og = cfg.o_groups as usize;
    let heads = cfg.n_heads as usize;
    let inter = cfg.inter_dim as usize;
    let nrouted = cfg.n_routed as usize;
    let mix_hc = cfg.mix_hc();
    let hc_dim = cfg.hc_dim();

    m.insert("embed.weight".into(), spec(TKind::Bf16, vec![vocab, dim]));
    m.insert("head.weight".into(), spec(TKind::Bf16, vec![vocab, dim]));
    m.insert("norm.weight".into(), spec(TKind::Bf16, vec![dim]));

    for l in 0..cfg.n_layers as usize {
        let p = format!("layers.{l}");
        m.insert(
            format!("{p}.attn_norm.weight"),
            spec(TKind::Bf16, vec![dim]),
        );
        m.insert(format!("{p}.ffn_norm.weight"), spec(TKind::Bf16, vec![dim]));
        m.insert(
            format!("{p}.hc_attn_fn"),
            spec(TKind::F32, vec![mix_hc, hc_dim]),
        );
        m.insert(
            format!("{p}.hc_ffn_fn"),
            spec(TKind::F32, vec![mix_hc, hc_dim]),
        );
        m.insert(format!("{p}.hc_attn_base"), spec(TKind::F32, vec![mix_hc]));
        m.insert(format!("{p}.hc_ffn_base"), spec(TKind::F32, vec![mix_hc]));
        m.insert(format!("{p}.hc_attn_scale"), spec(TKind::F32, vec![3]));
        m.insert(format!("{p}.hc_ffn_scale"), spec(TKind::F32, vec![3]));

        let fp8 = |m: &mut BTreeMap<String, TSpec>, name: String, rows: usize, cols: usize| {
            m.insert(format!("{name}.weight"), spec(TKind::Fp8, vec![rows, cols]));
            m.insert(
                format!("{name}.scale"),
                spec(TKind::E8m0, vec![rows.div_ceil(32), cols.div_ceil(32)]),
            );
        };
        fp8(&mut m, format!("{p}.attn.wq_a"), ql, dim);
        m.insert(
            format!("{p}.attn.q_norm.weight"),
            spec(TKind::Bf16, vec![ql]),
        );
        fp8(&mut m, format!("{p}.attn.wq_b"), heads * hd, ql);
        fp8(&mut m, format!("{p}.attn.wkv"), hd, dim);
        m.insert(
            format!("{p}.attn.kv_norm.weight"),
            spec(TKind::Bf16, vec![hd]),
        );
        m.insert(format!("{p}.attn.attn_sink"), spec(TKind::F32, vec![heads]));
        // convert.py dequantizes wo_a to bf16 on the host (spec §11)
        m.insert(
            format!("{p}.attn.wo_a.weight"),
            spec(TKind::Bf16, vec![og * ol, heads * hd / og]),
        );
        fp8(&mut m, format!("{p}.attn.wo_b"), dim, og * ol);

        m.insert(
            format!("{p}.ffn.gate.weight"),
            spec(TKind::Bf16, vec![nrouted, dim]),
        );
        m.insert(
            format!("{p}.ffn.gate.bias"),
            spec(TKind::F32, vec![nrouted]),
        );
        m.insert(
            format!("{p}.ffn.gate.bias_vl"),
            spec(TKind::F32, vec![nrouted]),
        );
        for e in 0..nrouted {
            for (w, rows, cols) in [("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)] {
                m.insert(
                    format!("{p}.ffn.experts.{e}.{w}.weight"),
                    spec(TKind::Fp4, vec![rows, cols]),
                );
                m.insert(
                    format!("{p}.ffn.experts.{e}.{w}.scale"),
                    spec(TKind::E8m0, vec![rows, cols.div_ceil(32)]),
                );
            }
        }
        fp8(&mut m, format!("{p}.ffn.shared_experts.w1"), inter, dim);
        fp8(&mut m, format!("{p}.ffn.shared_experts.w2"), dim, inter);
        fp8(&mut m, format!("{p}.ffn.shared_experts.w3"), inter, dim);

        let ratio = cfg.compress_ratios[l] as usize;
        if cfg.kv_sources.contains(&(l as u32)) {
            let kind = if ratio > 1 { TKind::F32 } else { TKind::Bf16 };
            m.insert(
                format!("{p}.attn.compressor.wkv.weight"),
                spec(kind, vec![hd, dim]),
            );
            if ratio > 1 {
                m.insert(
                    format!("{p}.attn.compressor.wgate.weight"),
                    spec(TKind::F32, vec![hd, dim]),
                );
            }
            m.insert(
                format!("{p}.attn.compressor.norm.weight"),
                spec(TKind::Bf16, vec![hd]),
            );
            // index keys are owned by kv-source layers (spec §4)
            m.insert(
                format!("{p}.attn.indexer.wk.weight"),
                spec(TKind::Bf16, vec![cfg.index_head_dim as usize, hd]),
            );
            m.insert(
                format!("{p}.attn.indexer.k_norm.weight"),
                spec(TKind::Bf16, vec![cfg.index_head_dim as usize]),
            );
        }
        if cfg.index_sources.contains(&(l as u32)) {
            let inh = cfg.index_n_heads as usize;
            let ihd = cfg.index_head_dim as usize;
            fp8(&mut m, format!("{p}.attn.indexer.wq_b"), inh * ihd, ql);
            m.insert(
                format!("{p}.attn.indexer.weights_proj.weight"),
                spec(TKind::Bf16, vec![inh, dim]),
            );
        }
        if let Some(ei) = cfg.engram_layers.iter().position(|&x| x == l as u32) {
            let rows = cfg.engram_num_embeddings[ei] as usize;
            let ehd = cfg.engram_head_dim as usize;
            let ncols = ((cfg.engram_max_ngram - 1) * cfg.engram_n_heads) as usize;
            m.insert(
                format!("{p}.engram.embed.weight"),
                spec(TKind::Fp8, vec![rows, ehd]),
            );
            m.insert(
                format!("{p}.engram.embed.scale"),
                spec(TKind::E8m0, vec![rows, ehd.div_ceil(32)]),
            );
            fp8(
                &mut m,
                format!("{p}.engram.wkv"),
                dim * (cfg.hc_mult as usize + 1),
                ncols * ehd,
            );
            m.insert(
                format!("{p}.engram.q_weight"),
                spec(TKind::Bf16, vec![cfg.hc_mult as usize, dim]),
            );
            m.insert(
                format!("{p}.engram.k_weight"),
                spec(TKind::Bf16, vec![cfg.hc_mult as usize, dim]),
            );
        }
    }
    m
}
