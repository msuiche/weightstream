//! MoE: sqrtsoftplus gate, top-k routed fp4 experts, shared fp8 expert
//! (spec §6, model.py:794-906).

use crate::actquant::fp4_gemv;
use crate::actquant::fp8_gemv;
use crate::config::RefConfig;
use crate::model::Ctx;
use crate::num;
use std::io;

/// One MoE sublayer: gate scoring, top-`n_activated` routed experts, the
/// shared expert on every token. Returns the fp32-assembled output `[dim]`.
pub fn moe(cfg: &RefConfig, ctx: &mut Ctx<'_>, layer: u32, x: &[f32]) -> io::Result<Vec<f32>> {
    let dim = cfg.dim as usize;
    let nrouted = cfg.n_routed as usize;
    let topk = cfg.n_activated as usize;
    let p = format!("layers.{layer}.ffn");

    // gate (model.py:794-829): fp32 scoring, bias steers selection only
    let scores = {
        let w = ctx.dense(&format!("{p}.gate.weight"))?;
        let mut s = num::gemv(w, nrouted, dim, x);
        if cfg.gate_temp != 1.0 {
            for v in s.iter_mut() {
                *v /= cfg.gate_temp;
            }
        }
        match cfg.score_func.as_str() {
            "softmax" => num::softmax(&mut s),
            "sigmoid" => {
                for v in s.iter_mut() {
                    *v = num::sigmoid(*v);
                }
            }
            // "sqrtsoftplus" (1017) and anything else fall back per spec §6
            _ => {
                for v in s.iter_mut() {
                    *v = num::softplus(*v).sqrt();
                }
            }
        }
        s
    };
    let bias = ctx.dense(&format!("{p}.gate.bias"))?.to_vec(); // text tokens use `bias` (spec §6)

    // top-k over scores + bias; weights come from the UNBIASED scores
    let mut order: Vec<usize> = (0..nrouted).collect();
    let biased: Vec<f32> = (0..nrouted).map(|i| scores[i] + bias[i]).collect();
    order.sort_by(|&a, &b| biased[b].total_cmp(&biased[a]).then(a.cmp(&b)));
    order.truncate(topk);
    let mut weights: Vec<f32> = order.iter().map(|&i| scores[i]).collect();
    if cfg.norm_topk_prob && topk > 1 {
        let sum: f32 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= sum + 1e-20; // literal 1e-20, NOT norm_eps (model.py:827)
        }
    }
    for w in weights.iter_mut() {
        *w *= cfg.route_scale;
    }

    // fp32 accumulator, experts added in expert-id order (model.py:895)
    let mut y = vec![0.0f32; dim];
    let mut sel: Vec<(usize, f32)> = order.iter().cloned().zip(weights.iter().copied()).collect();
    sel.sort_by_key(|&(e, _)| e); // ascending expert id, matching the reference loop
    for (e, w) in sel {
        let h = expert_hidden(cfg, ctx, &p, e, x)?;
        for d in 0..dim {
            y[d] += w * h[d];
        }
    }

    // shared expert on every token, no routing weight (model.py:905)
    let shared = shared_expert_hidden(cfg, ctx, &p, x)?;
    for d in 0..dim {
        y[d] += shared[d];
    }
    Ok(y)
}

/// One routed expert's hidden state: gate/up fp4 GEMMs, asymmetric
/// swiglu_limit clamps, silu·up (spec §6, model.py:843-853).
fn expert_hidden(
    cfg: &RefConfig,
    ctx: &mut Ctx<'_>,
    p: &str,
    expert: usize,
    x: &[f32],
) -> io::Result<Vec<f32>> {
    let inter = cfg.inter_dim as usize;
    let dim = cfg.dim as usize;
    let (gw, gs) = ctx.get2(
        &format!("{p}.experts.{expert}.w1.weight"),
        &format!("{p}.experts.{expert}.w1.scale"),
    )?;
    let gate = fp4_gemv(gw.bytes, gs.bytes, inter, dim, x);
    let (uw, us) = ctx.get2(
        &format!("{p}.experts.{expert}.w3.weight"),
        &format!("{p}.experts.{expert}.w3.scale"),
    )?;
    let up = fp4_gemv(uw.bytes, us.bytes, inter, dim, x);

    let limit = cfg.swiglu_limit;
    let mut h = vec![0.0f32; inter];
    for i in 0..inter {
        // asymmetric clamps: up on both sides, gate only from above
        let u = if limit > 0.0 {
            up[i].clamp(-limit, limit)
        } else {
            up[i]
        };
        let g = if limit > 0.0 {
            gate[i].min(limit)
        } else {
            gate[i]
        };
        h[i] = num::silu(g) * u;
    }
    let (w2, s2) = ctx.get2(
        &format!("{p}.experts.{expert}.w2.weight"),
        &format!("{p}.experts.{expert}.w2.scale"),
    )?;
    Ok(fp4_gemv(w2.bytes, s2.bytes, dim, inter, &h))
}

/// Shared expert: identical structure, fp8 weights (model.py:891-906).
fn shared_expert_hidden(
    cfg: &RefConfig,
    ctx: &mut Ctx<'_>,
    p: &str,
    x: &[f32],
) -> io::Result<Vec<f32>> {
    let inter = cfg.inter_dim as usize;
    let dim = cfg.dim as usize;
    let (gw, gs) = ctx.get2(
        &format!("{p}.shared_experts.w1.weight"),
        &format!("{p}.shared_experts.w1.scale"),
    )?;
    let gate = fp8_gemv(gw.bytes, gs.bytes, inter, dim, x);
    let (uw, us) = ctx.get2(
        &format!("{p}.shared_experts.w3.weight"),
        &format!("{p}.shared_experts.w3.scale"),
    )?;
    let up = fp8_gemv(uw.bytes, us.bytes, inter, dim, x);

    let limit = cfg.swiglu_limit;
    let mut h = vec![0.0f32; inter];
    for i in 0..inter {
        let u = if limit > 0.0 {
            up[i].clamp(-limit, limit)
        } else {
            up[i]
        };
        let g = if limit > 0.0 {
            gate[i].min(limit)
        } else {
            gate[i]
        };
        h[i] = num::silu(g) * u;
    }
    let (w2, s2) = ctx.get2(
        &format!("{p}.shared_experts.w2.weight"),
        &format!("{p}.shared_experts.w2.scale"),
    )?;
    Ok(fp8_gemv(w2.bytes, s2.bytes, dim, inter, &h))
}
