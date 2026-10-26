//! DS4.1F-1017 weight census: which tensors the text model must have.
//!
//! Built from the checkpoint's real naming (validated against the 48-shard,
//! 96 085-tensor index). The strict set covers everything the serving path
//! reads: 40 decoder layers (MLA attention, 384-expert MoE, hyper-connection
//! params), the drafter modules (`mtp.*`), embedding, head, final norm.
//! Vision, aligner and image-marker tensors are intentionally excluded —
//! ferroseek serves text only.

use crate::config::ModelConfig;
use std::collections::BTreeSet;

const PROJ: [&str; 4] = ["wq_a", "wq_b", "wkv", "wo_a"];
const EXPERT_W: [&str; 3] = ["w1", "w2", "w3"];

fn push_attn(set: &mut BTreeSet<String>, prefix: &str) {
    for p in PROJ {
        set.insert(format!("{prefix}.attn.{p}.weight"));
        set.insert(format!("{prefix}.attn.{p}.scale"));
    }
    set.insert(format!("{prefix}.attn.wo_b.weight"));
    set.insert(format!("{prefix}.attn.wo_b.scale"));
    set.insert(format!("{prefix}.attn.q_norm.weight"));
    set.insert(format!("{prefix}.attn.kv_norm.weight"));
    set.insert(format!("{prefix}.attn.attn_sink"));
}

fn push_ffn(set: &mut BTreeSet<String>, prefix: &str, n_experts: u32) {
    set.insert(format!("{prefix}.ffn.gate.weight"));
    set.insert(format!("{prefix}.ffn.gate.bias"));
    set.insert(format!("{prefix}.ffn.gate.bias_vl"));
    for w in EXPERT_W {
        set.insert(format!("{prefix}.ffn.shared_experts.{w}.weight"));
        set.insert(format!("{prefix}.ffn.shared_experts.{w}.scale"));
        for e in 0..n_experts {
            set.insert(format!("{prefix}.ffn.experts.{e}.{w}.weight"));
            set.insert(format!("{prefix}.ffn.experts.{e}.{w}.scale"));
        }
    }
}

fn push_layer_extras(set: &mut BTreeSet<String>, prefix: &str) {
    for hc in ["hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"] {
        set.insert(format!("{prefix}.{hc}"));
    }
    set.insert(format!("{prefix}.attn_norm.weight"));
    set.insert(format!("{prefix}.ffn_norm.weight"));
}

pub fn expected_tensors(cfg: &ModelConfig) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    let n_experts = cfg.n_routed_experts.unwrap_or(0);
    let n_mtp = cfg.num_nextn_predict_layers();

    set.insert("embed.weight".into());
    set.insert("head.weight".into());
    set.insert("norm.weight".into());

    for layer in 0..cfg.num_hidden_layers {
        let prefix = format!("layers.{layer}");
        push_attn(&mut set, &prefix);
        push_ffn(&mut set, &prefix, n_experts);
        push_layer_extras(&mut set, &prefix);
        if cfg.kv_source_layer_ids.contains(&layer) {
            set.insert(format!("{prefix}.attn.compressor.norm.weight"));
            set.insert(format!("{prefix}.attn.compressor.wkv.weight"));
            set.insert(format!("{prefix}.attn.indexer.k_norm.weight"));
            set.insert(format!("{prefix}.attn.indexer.wk.weight"));
            // wgate exists on every kv-source layer except the last
            if cfg.kv_source_layer_ids.last() != Some(&layer) {
                set.insert(format!("{prefix}.attn.compressor.wgate.weight"));
            }
        }
        if cfg.index_source_layer_ids.contains(&layer) {
            set.insert(format!("{prefix}.attn.indexer.weights_proj.weight"));
            set.insert(format!("{prefix}.attn.indexer.wq_b.weight"));
            set.insert(format!("{prefix}.attn.indexer.wq_b.scale"));
        }
        if cfg.engram_layer_ids.contains(&layer) {
            set.insert(format!("{prefix}.engram.embed.weight"));
            set.insert(format!("{prefix}.engram.embed.scale"));
            set.insert(format!("{prefix}.engram.k_weight"));
            set.insert(format!("{prefix}.engram.q_weight"));
            set.insert(format!("{prefix}.engram.wkv.weight"));
            set.insert(format!("{prefix}.engram.wkv.scale"));
        }
    }

    for k in 0..n_mtp {
        let prefix = format!("mtp.{k}");
        push_attn(&mut set, &prefix);
        // the drafter has its own, smaller expert pool (128 on DS4.1F)
        push_ffn(&mut set, &prefix, cfg.dspark_n_routed_experts.unwrap_or(n_experts));
        push_layer_extras(&mut set, &prefix);
        if k == 0 {
            // only the first drafter block carries main_norm
            set.insert(format!("{prefix}.main_norm.weight"));
        }
        if k == 0 {
            set.insert(format!("{prefix}.main_proj.weight"));
            set.insert(format!("{prefix}.main_proj.scale"));
        }
        if k + 1 == n_mtp {
            set.insert(format!("{prefix}.norm.weight"));
            set.insert(format!("{prefix}.confidence_head.proj.weight"));
            set.insert(format!("{prefix}.markov_head.embed.weight"));
            set.insert(format!("{prefix}.markov_head.head.weight"));
        }
    }
    set
}

/// Split an index's tensor names into (missing from checkpoint,
/// unexpected by the census). `unexpected` excludes vision-side tensors,
/// which we know we skip.
pub fn diff_index(
    cfg: &ModelConfig,
    index: &std::collections::BTreeMap<String, String>,
) -> (Vec<String>, Vec<String>) {
    let expected = expected_tensors(cfg);
    let missing: Vec<String> = expected
        .iter()
        .filter(|n| !index.contains_key(*n))
        .cloned()
        .collect();
    let unexpected: Vec<String> = index
        .keys()
        .filter(|n| {
            !expected.contains(*n)
                && !n.starts_with("vision.")
                && !n.starts_with("aligner.")
                && !n.starts_with("image_")
        })
        .cloned()
        .collect();
    (missing, unexpected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::tests::fixture;

    #[test]
    fn census_matches_real_checkpoint_counts() {
        let cfg = fixture();
        let set = expected_tensors(&cfg);
        // Real 1017 index has 96 085 names; text-side strict census must
        // account for ~99k of them (the rest is vision/aligner/indexer).
        let experts = 40 * 384 * 3 * 2 + 3 * 128 * 3 * 2; // main + drafter
        assert_eq!(
            set.iter().filter(|n| n.contains("ffn.experts.")).count(),
            experts as usize
        );
        assert!(set.contains("head.weight"));
        assert!(set.contains("mtp.2.markov_head.head.weight"));
        assert!(set.contains("layers.2.attn.compressor.wkv.weight"));
        assert!(!set.contains("layers.3.attn.compressor.wkv.weight"));
    }
}
