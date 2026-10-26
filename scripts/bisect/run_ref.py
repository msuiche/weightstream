"""Synthetic bisection: run the reference model.py forward on the synthetic
mini-model dumped by ferroseek-ref's dump-synth binary, then write the same
per-layer per-position stream files with a `py_` prefix for comparison.

Usage (inside the reference container, model.py reachable):
    REF_DIR=/bisect/ref OUT=/bisect/out python3 run_ref.py

Assumptions baked in (see crates/ferroseek-ref/src/bin/dump-synth.rs):
- TP1, bf16 torch default dtype, args.dtype "fp8" / expert_dtype "fp4"
- identity compressed token map (synthetic vocab == engram compressed vocab)
- one prefill chunk over the whole token list; per-token outputs equal
  chunked prefill because attention is causal and the window/compressor
  state evolves identically (spec §1). Window rows map ring-slot j == token
  j while the prompt is shorter than the window, so gathered-KV rows pair
  1:1 with the rust decode dumps.
"""

import json
import os
import sys

import numpy as np
import torch

REF_DIR = os.environ.get("REF_DIR", "/bisect/ref")
OUT = os.environ.get("OUT", "/bisect/out")

sys.path.insert(0, REF_DIR)

import engram as engram_mod  # noqa: E402

with open(os.path.join(OUT, "model_args.json")) as f:
    ARGS = json.load(f)

# Synthetic weights use an identity compressed map; the real 99092-entry map
# is built from a tokenizer we do not have here (spec §11).
V = ARGS["engram_compressed_vocab_size"]
assert V == ARGS["vocab_size"]
engram_mod.build_compressed_token_map = lambda tok: (list(range(V)), V)

import model as model_mod  # noqa: E402
from model import Transformer, ModelArgs  # noqa: E402
import kernel as kernel_mod  # noqa: E402

tokens = json.load(open(os.path.join(OUT, "tokens.json")))
S = len(tokens)

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(0)
args = ModelArgs(**ARGS)
with torch.device("cuda"):
    model = Transformer(args, tokenizer=None)

from safetensors.torch import load_file  # noqa: E402

sd = load_file(os.path.join(OUT, "synth.safetensors"))
mkeys = set(model.state_dict().keys())
missing = mkeys - set(sd)
assert not missing, f"model params absent from the dump: {sorted(missing)}"
extra = set(sd) - mkeys
if extra:
    print("ignoring dump keys the model does not carry:", sorted(extra))
sd = {k: v for k, v in sd.items() if k in mkeys}
model.load_state_dict(sd, strict=True)
model.eval()
# generate.py:134 — runtime index tensors (window topk, candidates) follow the
# default device from here on
torch.set_default_device("cuda")

# ---- recording wrappers (all thin, no reimplementation) -------------------

sparse_rec = []  # per attention call, in layer order


def wrap_sparse_attn(orig):
    def w(q, kv, attn_sink, topk_idxs, scale):
        out = orig(q, kv, attn_sink, topk_idxs, scale)
        sparse_rec.append(
            dict(
                q=q.detach().float().cpu(),
                kv=kv.detach().float().cpu(),
                sink=attn_sink.detach().float().cpu(),
                idxs=topk_idxs.detach().cpu(),
                out=out.detach().float().cpu(),
            )
        )
        return out

    return w


model_mod.sparse_attn = wrap_sparse_attn(model_mod.sparse_attn)

# SHADOW=1: perturb the embedding by a bf16-scale relative error and dump to
# the same layout — a control run measuring how much pure transport noise
# amplifies through the layers (py-vs-py, no engine difference).
if os.environ.get("SHADOW"):
    orig_embed_fwd = model_mod.ParallelEmbedding.forward

    def shadow_embed_fwd(self, x):
        out = orig_embed_fwd(self, x)
        g = torch.Generator(device="cuda").manual_seed(777)
        u = (torch.rand(out.shape, device="cuda", generator=g) - 0.5) * (2.0**-7)
        # perturb in f32, THEN round to bf16 — computing 1+u in bf16 would
        # round back to exactly 1.0 and cancel the perturbation entirely
        return (out.float() * (1.0 + u)).to(out.dtype)

    model_mod.ParallelEmbedding.forward = shadow_embed_fwd

hc_rec = []  # per hc_split_sinkhorn call: 2 per block, layer order


def wrap_hc_split(orig):
    def w(mixes, hc_scale, hc_base, *a, **k):
        pre, post, comb = orig(mixes, hc_scale, hc_base, *a, **k)
        hc_rec.append(
            dict(
                pre=pre.detach().float().cpu(),
                post=post.detach().float().cpu(),
                comb=comb.detach().float().cpu(),
            )
        )
        return pre, post, comb

    return w


model_mod.hc_split_sinkhorn = wrap_hc_split(model_mod.hc_split_sinkhorn)

caps = {}


def cap_output(name, pick=lambda out: out):
    def hook(mod, inp, out):
        caps[name] = pick(out).detach().float().cpu()

    return hook


def cap_input(name):
    def hook(mod, args):
        caps[name] = args[0].detach().float().cpu()

    return hook


for l, layer in enumerate(model.layers):
    layer.register_forward_pre_hook(cap_input(f"in{l}"))
    layer.register_forward_hook(cap_output(f"blockout{l}", pick=lambda out: out[0]))
    layer.attn.register_forward_hook(cap_output(f"attn{l}"))
    layer.ffn.register_forward_hook(cap_output(f"ffn{l}"))
    layer.attn.q_norm.register_forward_hook(cap_output(f"qr{l}"))
    layer.attn.wo_b.register_forward_pre_hook(cap_input(f"fl{l}"))

    def gate_hook(mod, inp, out, l=l):
        caps[f"gweights{l}"] = out[0].detach().float().cpu()
        caps[f"gorder{l}"] = out[1].detach().cpu()

    layer.ffn.gate.register_forward_hook(gate_hook)

    if layer.attn.compressor is not None:
        def comp_hook(mod, inp, out, l=l):
            caps[f"latent{l}"] = (
                out.detach().float().cpu() if out is not None else None
            )

        layer.attn.compressor.register_forward_hook(comp_hook)

    if layer.attn.indexer is not None:
        def idx_hook(mod, args, out, l=l):
            caps[f"cmpidxs{l}"] = out.detach().cpu()

        layer.attn.indexer.register_forward_hook(idx_hook)

    if layer.engram is not None:
        layer.engram.register_forward_pre_hook(cap_input(f"pre{l}"))
    else:
        caps[f"pre{l}"] = None  # filled from in{l} after the forward


def cap_last_block(mod, inp, out):
    caps["blockout_last"] = out[0].detach().float().cpu()
    caps["ffnpre_last"] = out[1].detach().float().cpu()


model.layers[-1].register_forward_hook(cap_last_block)

x = torch.tensor([tokens], dtype=torch.long, device="cuda")
with torch.inference_mode():
    model(x, start_pos=0)

for l, layer in enumerate(model.layers):
    if layer.engram is None:
        caps[f"pre{l}"] = caps[f"in{l}"]

assert len(sparse_rec) == len(model.layers), len(sparse_rec)
assert len(hc_rec) == 2 * len(model.layers), len(hc_rec)

# per-position logits, replicating the final collapse exactly
# (model.py:1271-1272: hc_pre with the last block's ffn_pre, norm, head)
with torch.inference_mode():
    h_last = caps["blockout_last"].to("cuda").to(torch.bfloat16)
    ffn_pre = caps["ffnpre_last"].to("cuda")
    xc = model.layers[-1].hc_pre(h_last, ffn_pre)
    xn = model.norm(xc)
    logits = model.head(xn, full_logits=True).float().cpu()  # [1, S, V]
caps["logits"] = logits[0]


def write(name, p, arr):
    np.asarray(arr, dtype=np.float32).tofile(
        os.path.join(OUT, f"py_pos{p}_{name}.f32")
    )


def write_i32(name, p, arr):
    np.asarray(arr, dtype=np.int32).tofile(
        os.path.join(OUT, f"py_pos{p}_{name}.i32")
    )


NL = len(model.layers)
names = (
    [f"pre{l}" for l in range(NL)]
    + [f"in{l}" for l in range(NL)]
    + [f"attn{l}" for l in range(NL)]
    + [f"ffn{l}" for l in range(NL)]
    + ["blockout_last"]
)
for name in names:
    a = caps[name][0]
    assert a.shape[0] == S, (name, a.shape)
    for p in range(S):
        write(name, p, a[p])
for p in range(S):
    np.asarray(caps["logits"][p], dtype=np.float32).tofile(
        os.path.join(OUT, f"py_logits{p}.f32")
    )

# ---- detail dumps, paired with the rust d_pos{p}_*{l} files ---------------


def write_full(name, arr):
    np.asarray(arr, dtype=np.float32).tofile(os.path.join(OUT, "py_" + name))


def write_full_i32(name, arr):
    np.asarray(arr, dtype=np.int32).tofile(os.path.join(OUT, "py_" + name))


WIN = args.window_size
for l in range(NL):
    layer = model.layers[l]
    rec = sparse_rec[l]
    ratio = args.compress_ratios[l]
    # q post-rope / raw attention output / window+compress indices
    for p in range(S):
        write_full(f"d_pos{p}_q{l}.f32", rec["q"][0, p])
        write_full(f"d_pos{p}_attnraw{l}.f32", rec["out"][0, p])
        write_full_i32(f"d_pos{p}_idxs{l}.i32", rec["idxs"][0, p])
        write_full(f"d_pos{p}_qr{l}.f32", caps[f"qr{l}"][0, p])
        write_full(f"d_pos{p}_fl{l}.f32", caps[f"fl{l}"][0, p])
        write_full(f"d_pos{p}_gweights{l}.f32", caps[f"gweights{l}"][p])
        write_full_i32(f"d_pos{p}_gorder{l}.i32", caps[f"gorder{l}"][p])
    # o post inverse-rope + fp8 snap, reconstructed with the model's own ops
    o = rec["out"].to("cuda").to(torch.bfloat16).clone()
    rd = layer.attn.rope_head_dim
    with torch.inference_mode():
        model_mod.apply_rotary_emb(o[..., -rd:], layer.attn.freqs_cis[:S], True)
        kernel_mod.act_quant(
            o, model_mod.fp8_block_size, model_mod.scale_fmt, model_mod.scale_dtype, True
        )
    o = o.float().cpu()
    for p in range(S):
        write_full(f"d_pos{p}_osnap{l}.f32", o[0, p])
    # gathered attended rows, sorted window-by-token then compress-by-group
    nwin = rec["kv"].shape[1] - (S // ratio if ratio else 0)
    for p in range(S):
        rows = []
        for i in rec["idxs"][0, p].tolist():
            if i < 0:
                continue
            key = (0, i) if i < nwin else (1, i - nwin)
            rows.append((key, rec["kv"][0, i]))
        rows.sort(key=lambda r: r[0])
        write_full(f"d_pos{p}_gather{l}.f32", np.stack([r[1] for r in rows]))
    # latent rows: group t completes at token t*ratio + (ratio-1)
    lat = caps.get(f"latent{l}")
    if ratio and lat is not None:
        for t in range(lat.shape[1]):
            p = t * ratio + (ratio - 1)
            if p < S:
                write_full(f"d_pos{p}_latent{l}.f32", lat[0, t])
    if f"cmpidxs{l}" in caps:
        for p in range(S):
            write_full_i32(f"d_pos{p}_cmpidxs{l}.i32", caps[f"cmpidxs{l}"][0, p])

# hc mixes: 2 calls per block — attn first, ffn second
for i, rec in enumerate(hc_rec):
    l, kind = divmod(i, 2)
    tag = "hca" if kind == 0 else "hcf"
    for p in range(S):
        write_full(f"d_pos{p}_{tag}_pre{l}.f32", rec["pre"][0, p])
        write_full(f"d_pos{p}_{tag}_post{l}.f32", rec["post"][0, p])
        write_full(f"d_pos{p}_{tag}_comb{l}.f32", rec["comb"][0, p])

print(f"wrote py_ dumps for {len(names)} streams + details to {OUT}")