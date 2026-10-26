# DeepSeek-V4.1-Flash-1017 — decode compute spec

Derived from the reference inference stack at `/tmp/ds41f-ref/inference`
(`model.py`, `kernel.py`, `generate.py`, `engram.py`, `convert.py`) and the
release `config.json` (text_config, `tie_word_embeddings: false`,
`quantization_config`: fp8 block `[32,32]` `scale_fmt ue8m0`, experts fp4,
dynamic activation). All citations `(model.py:Lxxx)` refer to that reference.
Single-rank (TP1) semantics are normative; TP4 notes are additive.

## 0. Configuration (1017 checkpoint)

| name | value | name | value |
|---|---|---|---|
| vocab_size | 129280 | dim | 5120 |
| n_layers | 40 | n_mtp_layers (dspark) | 3 |
| n_heads | 64 | head_dim | 512 |
| rope_head_dim (qk_rope) | 64 | nope_head_dim | 448 |
| q_lora_rank | 1280 | o_lora_rank / o_groups | 1024 / 8 |
| window_size | 128 | norm_eps | 1e-20 |
| moe_inter_dim | 2304 | n_routed / activated | 384 / 6 |
| score_func | sqrtsoftplus | gate_temp | 1.0 |
| route_scale | 1.5 | swiglu_limit | 10.0 |
| n_shared_experts | 1 | norm_topk_prob | true |
| kv_source_layers | [2,8,14,20] | index_source_layers | [2,8,14,20,24,28,32,36] |
| candidate_source_layer | 20 | candidate_topk_blocks / block_size | 2048 / 8 |
| index_n_heads | 32 | index_head_dim | 128 |
| index_topk | 512 | hc_mult | 4 |
| hc_sinkhorn_iters | 20 | hc_eps | 1e-6 |
| rope_theta | 10000 | compress_rope_theta | 160000 |
| original_seq_len | 65536 | rope_factor / beta_fast / beta_slow | 16 / 32 / 1 |
| engram_layer_ids | [1,14] | engram_num_embeddings | [384006168, 384016682] |
| engram_max_ngram_size | 4 | engram_n_heads / head_dim | 8 / 256 |
| engram_pad_id | 2 | engram_compressed_vocab_size | 99092 |
| dspark_block_size | 5 | dspark_noise_token_id | 128799 |
| dspark_target_layer_ids | [37,38,39] | dspark_markov_rank | 256 |
| dspark routed / activated | 128 / 3 | image_token_id | 129264 |
| eos / bos / pad | 1 / 0 / 2 | dtype / expert_dtype | fp8 / fp4 |

`compress_ratios` (43 entries: 40 backbone + 3 MTP, `model.py:81`):

```
L0-1:   0   (sliding window only)
L2-19:  2   (CSA, compressed 2:1)
L20-39: 1   (plain projection "compressed" KV, 1:1)
MTP0-2: 0   (sliding window only)
```

Shared state across layers within one forward pass (`model.py:1169-1183`):
one global runtime slot each for `compress_kv`, `index_k`, `topk_idxs`,
`candidates`. A source layer writes before its consumers read; nothing is
reset between steps. Effective publishers:

- `compress_kv` (the KV cache tensor itself): layer 2 → consumers 3-7, 8 →
  9-13, 14 → 15-19, 20 → 21-39 (`model.py:747-749`).
- `index_k`: last kv_source layer to run so far; after L20 it is L20's cache
  for all later indexers (`model.py:548-549`).
- `topk_idxs`: layer 2 → 3-7, 8 → 9-13, 14 → 15-19, 20 → 21-23,
  24 → 25-27, 28 → 29-31, 32 → 33-35, 36 → 37-39 (`model.py:723-738`).
- `candidates`: L20's indexer → indexers at 24, 28, 32, 36 (`model.py:570-576`).

## 1. Token flow end to end

Decode step, one sequence, batch B, `start_pos = p` (0-based absolute
position of the token being consumed), `input_ids: [B,1]` (`model.py:1244-1275`):

```
h = embed(input_ids)                       # [B,1,5120] bf16 (model.py:1256)
# (VL only, prefill: image spans overwrite h rows — out of decode scope)
h = h.unsqueeze(2).repeat(1,1,4,1)         # [B,1,4,5120] hyper-connection copies (model.py:1261)
pre_mix = one-hot([1,0,0,0]) fp32          # [B,1,4] (model.py:1162-1166)
engram_hashes = NgramHashState(input_ids, p, text_mask)   # [B,1,2,24]
for i, layer in enumerate(layers):        # 40 blocks
    if i in engram_layer_ids: h = layer.engram(h, hashes_for_layer, mask)
    if i in [37,38,39]: main_hiddens.append(h.mean(dim=2))   # [B,1,5120] each
    h, pre_mix = layer(h, p, pre_mix, image_mask)
h = last_layer.hc_pre(h, pre_mix)          # collapse 4 copies → [B,1,5120] (model.py:1271)
x = norm(h)                                # final RMSNorm (model.py:1208)
logits = head(x)                           # [B,129280] fp32, last position only (model.py:1010-1019)
output_ids = sample(logits, temperature)   # model samples internally (model.py:1273)
```

- `main_hiddens` are captured from the block *input* stream (mean over the 4
  hc copies), before layers 37/38/39 run (`model.py:1267-1269`); concatenated
  to `[B,1,15360]` they feed the drafter (§9).
- `tie_word_embeddings: false` — `embed.weight` and `head.weight` are distinct
  tensors. The MTP modules, however, *tie* to the backbone: `mtp.K.embed` and
  `mtp.K.head` are assigned the backbone modules (`model.py:1213-1216`), and
  `convert.py` drops any `mtp.*.embed.weight` / `mtp.*.head.weight` from the
  export (`convert.py:109-110`).
- `head` is bf16 on disk, held as **fp32** in memory so logits come out fp32
  (`model.py:1007-1008`); final matmul is `F.linear(x.float(), weight)` fp32.
- Prefill is one chunk: `start_pos=0`, whole prompt, all positions; images
  only in that chunk. Decode is `seqlen=1` per call (`generate.py:67-78`).

## 2. Hyper-connection stream mechanics

Stream shape: `[B, S, hc_mult=4, dim=5120]`. Parameters per block, all fp32
(`model.py:940-948`):

```
hc_attn_fn / hc_ffn_fn : [24, 20480]    # mix_hc=24=(2+4)*4 rows, hc_dim=4*5120
hc_attn_base / hc_ffn_base : [24]
hc_attn_scale / hc_ffn_scale : [3]      # one scalar for pre, post, comb
```

Mix computation `hc_mixes` (`model.py:950-957`) — per token, on the full
flattened 20480-wide stream, fp32:

```
xf   = h.flatten(2).float()                        # [B,S,20480]
rstd = rsqrt(mean(xf^2, dim=-1) + 1e-20)           # ONE statistic per token over all 4 copies
mixes = (xf @ hc_fn^T) * rstd                      # [B,S,24]
```

Split + Sinkhorn (`kernel.py:406-474`), per token, fp32 (hc_eps = 1e-6):

```
pre[j]  = sigmoid(mixes[j]        * scale[0] + base[j])        + eps   # j=0..3
post[j] = 2 * sigmoid(mixes[4+j]  * scale[1] + base[4+j])              # no eps
comb_raw[j][k] = mixes[8 + j*4 + k] * scale[2] + base[8 + j*4 + k]     # j = SOURCE copy, k = DEST copy

# Sinkhorn on comb (row = source j, col = destination k):
comb = softmax_row(comb_raw) + eps          # entrywise +eps AFTER softmax (kernel.py:439-443)
comb = comb / (colsum + eps)                # colsum over sources (kernel.py:445-448)
repeat (hc_sinkhorn_iters - 1) = 19 times:  # (kernel.py:450-458)
    comb = comb / (rowsum + eps)
    comb = comb / (colsum + eps)
```

Exactly 20 iterations total (1 initial row-softmax/col-norm pair + 19
row/col pairs); final matrix is not re-softmaxed. All divides add `eps` in
the denominator; only the first softmax adds `eps` to the entries.

Folding / unfolding (`model.py:959-968`), fp32 then cast to stream dtype:

```
hc_pre(x[4], pre)      -> y[d]     = Σ_j pre[j] * x[j][d]                    # collapse
hc_post(x, res, post, C) -> y[j][d] = post[j] * x[d] + Σ_a C[a][j] * res[a][d]
                                                                       # C indexed [source, dest]
```

Sublayer wiring inside `Block.forward` (`model.py:970-996`) — a sublayer's
mixes are computed from the *current stream* and its `pre` is consumed by the
*next* sublayer:

```
residual = h
attn_pre, attn_post, attn_comb = hc_mixes(h, hc_attn_fn, hc_attn_scale, hc_attn_base)
x  = hc_pre(h, pre_mix_in)      # pre_mix_in from the PREVIOUS layer's FFN (identity at L0)
x  = attn_norm(x);  x = attn(x, start_pos)
h  = hc_post(x, residual, attn_post, attn_comb)

residual = h
ffn_pre, ffn_post, ffn_comb = hc_mixes(h, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
x  = hc_pre(h, attn_pre)       # this block's attention coefficients feed the FFN
x  = ffn_norm(x);  x = ffn(x, image_mask)
h  = hc_post(x, residual, ffn_post, ffn_comb)
return h, ffn_pre              # feeds the NEXT block's attention
```

After L39: `h = hc_pre(h, L39.ffn_pre)`, then final norm + head
(`model.py:1271-1272`). Residual ordering is therefore: each sublayer output
is scaled by `post` and *added into* a Sinkhorn-mixed copy of the stream —
there is no plain `x + residual` anywhere.

## 3. Attention per layer

Per-layer tensors (TP1) and shapes:

```
wq_a   Linear(5120→1280)  fp8, scale [40,160]  ue8m0
q_norm RMSNorm(1280, eps=1e-20)
wq_b   Linear(1280→64*512=32768) fp8, scale [1024,40]   (column-parallel)
wkv    Linear(5120→512)  fp8, scale [16,160]           # single shared KV head
kv_norm RMSNorm(512)
attn_sink [64] fp32
wo_a   Linear(4096→8192) bf16, scale None (dequantized by convert.py)
wo_b   Linear(8192→5120) fp8, scale [160,256]           (row-parallel)
```

Query path (`model.py:771-773`):

```
qr = q_norm(wq_a(x))                     # [B,S,1280]
q  = wq_b(qr).view(B,S,64,512)           # heads×head_dim, nope 448 | rope 64
apply_rotary_emb(q[..., -64:], freqs_cis[p:p+S])
```

`qr` is shared: the DSA indexer derives its query from the same 1280-d latent
(`model.py:777`, `model.py:551`). Q stays bf16 (not quantized) into the
attention kernel.

KV path (`model.py:701-721`) — one latent vector serves as both K and V:

```
kv = kv_norm(wkv(x))                     # [B,S,512]
apply_rotary_emb(kv[..., -64:], freqs_cis[p:p+S])
act_quant(kv, 32, "ue8m0", inplace=True) # fp8 snap: quant→dequant back to bf16
```

RoPE (`model.py:368-406`): complex rotation of adjacent pairs over the last
`rope_head_dim=64` dims (32 complex freqs), computed in fp32. Base frequency
table **depends on the layer** (`model.py:681-688`):

```
compress_ratio > 0 (L2-39): theta = compress_rope_theta = 160000, YaRN ON
                            (original_seq_len 65536, factor 16, beta_fast 32, beta_slow 1)
compress_ratio == 0 (L0,1, MTP): theta = rope_theta = 10000, YaRN OFF
```

YaRN (`model.py:376-386`): `freqs = 1/theta^(2i/d)`; `low =
floor(d·ln(orig/(32·2π))/(2·ln θ))`, `high = ceil(... rotations=1 ...)`;
`ramp = clamp((i-low)/(high-low), 0, 1)`, `smooth = 1-ramp`;
`freqs = freqs/factor·(1-smooth) + freqs·smooth`. No attention-scale
(mscale) correction is applied — this implementation omits the standard
YaRN `0.1·ln factor + 1` softmax temperature.

Sliding window (`model.py:409-426`, `model.py:701-721`): ring of 128 slots.
Decode writes `window_kv_cache[B, p % 128] = kv` and attends to
`min(128, p+1)` slots, listed oldest-first; slots with absolute position
`> p` are marked `-1` (empty). Window K stays bf16 storage with fp8-snapped
values (§7).

Compressed path (L2-39 only): see §4/§5. The final attention input is the
concatenation (`model.py:776-779`):

```
kv_all = cat([window_kv (≤128 rows), compress_kv (compress_len rows)], dim=1)  # [B, n, 512]
topk_idxs = cat([window_idxs (≤128), compress_idxs (≤512, offset by 128 in decode)], dim=-1)
# compress idx = compressed_position + offset (= window_kv.size(1)); unreachable = -1
```

Attention kernel `sparse_attn` (`kernel.py:311-403`), scale `512^-0.5`,
fp32 online softmax over gathered rows:

```
logit_i = (q · kv_i) * scale            # invalid idx (-1): logit = -inf, value row = 0
running max/sum per head, exp(logit - max)
sink: after all blocks, sum_exp += exp(attn_sink[h] - max)   # kernel.py:382-383
out = acc_o / sum_exp
```

`attn_sink` is exactly equivalent to one extra key with logit
`attn_sink[h]` and a zero value vector, added after the online-softmax
accumulation. Rows with all `-1` indices produce a finite lower bound
(`max = -1e30`) and an all-zero output — not NaN (`kernel.py:355`). Heads <
16 are zero-padded for the kernel and stripped after (`kernel.py:394-402`;
irrelevant at 64 heads).

Output projection (`model.py:781-791`):

```
apply_rotary_emb(o[..., -64:], freqs_cis[p:p+S], inverse=True)  # CONJUGATE rotation at the query position
o = o.view(B,S,8,4096)                 # 8 groups × (8 heads × 512)
act_quant(o, 32, "ue8m0", inplace=True)
wo_a viewed [8, 1024, 4096]             # block-diagonal: group g sees only its own 8 heads
o = einsum("bsgd,grd->bsgr", o, wo_a)   # [B,S,8,1024]
x = wo_b(o.flatten(2))                  # [B,S,5120], fp8 GEMM
```

The inverse RoPE on the output tail exists because V == the rotated KV
latent; the output's rope dims carry the per-key rotation and the kernel
removes the *query's* rotation so the cache holds one shared rotated form
(`model.py:392-395`). Replicate exactly: rotate `o[..., -64:]` by
`conj(freqs_cis[query position])`.

TP1 head counts: 64 heads, 8 wo groups, 32 index heads, 1 KV head.
**TP4 note** (`model.py:246-278`, `628`, `643`, `651`): heads are split 16
per rank; `wq_b`, `wo_a`, `weights_proj`, `head` are column-parallel (output
dim split), `wo_b` is row-parallel (8192→2048 per rank, fp32 all-reduce,
no bias); `wkv`, `wq_a`, `q_norm`, `kv_norm` are replicated; `attn_sink` is
per-local-head `[16]`; gate and shared expert are replicated (shared-expert
output is recomputed, not reduced); routed experts split 96 per rank with an
fp32 all-reduce over expert outputs; `index_score` is all-reduced
(`model.py:559-560`); embedding/head are vocab-sharded (32320 rows per rank)
with head logits all-gathered (`model.py:1015-1018`).

## 4. DSA indexer

Present at `index_source_layers = [2,8,14,20,24,28,32,36]`. Index *keys* are
owned only by `kv_source_layers` (all four are index sources); later
indexers read the shared `index_k` cache published by L20 (`model.py:500`,
`548-549`). Per indexer (TP1):

```
wq_b        Linear(1280 → 32*128 = 4096) fp8, scale [128,40]   (column-parallel)
weights_proj Linear(5120 → 32) bf16                              (column-parallel)
wk          Linear(512 → 128) bf16           # key owner only
k_norm      RMSNorm(128, eps=1e-20)
k_cache     [B, max_seq/ratio, 128] bf16     # values fp4-snapped
softmax_scale = index_head_dim^-0.5 = 128^-0.5
```

Key side, on each completed compression group (`model.py:536-547`):

```
k = k_norm(wk(latent))                  # latent = pre-RoPE compressor output (§5)
apply_rotary_emb(k[..., -64:], freqs_cis[group first-token position j*ratio])
fp4_act_quant(k, 32, inplace=True)       # E8M0 scales, fp4 snap → stays bf16
k_cache[B, j] = k
```

Query side per token (`model.py:551-553`):

```
q = wq_b(qr).view(B,S,32,128)           # qr = shared q latent from §3
apply_rotary_emb(q[..., -64:], freqs_cis[p:p+S])
fp4_act_quant(q, 32, inplace=True)      # E8M0 scales
```

Scoring (`model.py:555-560`), fp32 accumulate:

```
score[b,s,h,t] = Σ_d q[b,s,h,d] · index_k[b,t,d]
score = relu(score)
weights = weights_proj(x) * (128^-0.5 * 32^-0.5)     # [B,S,32]; n_heads is the GLOBAL 32
index_score[b,s,t] = Σ_h score[b,s,h,t] * weights[b,s,h]
# TP: all_reduce(index_score)
```

Causality (`model.py:562-568`): a compressed group is visible to a query
once the query has passed its last token. Prefill: query at position `p` sees
`(p+1)//ratio` groups; masked positions set to `-inf`. Decode
(`compress_lens = (p+1)//ratio`): all `compress_len` groups visible,
including the group completed by the current token.

Two-level top-k:

- **Level 1 — candidate blocks** (`model.py:584-611`), run by L20's indexer
  on its own full `index_score`: block score = max over the block's 8
  compressed positions (`-inf` pads the tail); the block containing the
  newest position `(compress_lens-1)//8` is pinned to `+inf` so it survives;
  keep the top `candidate_topk_blocks = 2048` blocks; picks whose score is
  `-inf` are dropped; result is a bool mask expanded to per-position width
  and published as `shared_attn.candidates`. Indexers at 24/28/32/36 mask
  their own scores with `~candidates → -inf` before their own top-k
  (`model.py:574-576`).
- **Level 2 — final top-k** (`model.py:579-581`):

```
topk = min(index_topk=512, end_pos // ratio)
idxs = topk(index_score, topk).indices.sort(dim=-1).values   # re-sorted into position order
idxs = (idxs < compress_lens) ? idxs + offset : -1           # offset = window_kv.size(1) (=128 decode)
```

Selection feeds sparse MLA purely through `topk_idxs`: the attention
gathers those rows of `compress_kv` (concatenated after the window rows,
§3). Consumers between index sources reuse the published `topk_idxs`
verbatim (`model.py:723-738`).

## 5. CSA compressor (kv_source_layers 2, 8, 14, 20)

A compressor exists only on kv-source layers (`model.py:659-660`); every
`compress_ratio > 0` layer reads the cache the source published.

Ratio 2 (L2, L8, L14) (`model.py:437-485`) — weights fp32:

```
wkv  Linear(5120→512) fp32
wgate Linear(5120→512) fp32
norm RMSNorm(512, eps=1e-20)
kv_state   buffer [B, 2, 512] fp32     # carries the open pair across decode steps
score_state buffer [B, 2, 512] fp32    # initialized 0 / -inf
```

Decode, one token per call, all math fp32 (`model.py:476-482`):

```
slot = p % 2
kv_state[B, slot]    = wkv(x)          # x cast to fp32 first
score_state[B, slot] = wgate(x)
if (p+1) % 2 == 0:                     # pair (2j, 2j+1) just completed
    latent = Σ_slot kv_state[slot] * softmax(score_state, dim=slots)   # softmax over exactly 2 slots
    latent = norm(latent.to(bf16))     # RMSNorm output bf16
    # latent stands for position 2j (first token of the pair)
else:
    return None                       # open pair: nothing appended this step
```

"Open pair" rule: the current token's `wkv`/`wgate` outputs park in the
state buffers; the compressed KV cache, the index-key cache, and
`compress_len` all advance only on pair completion (odd `p`). The latent is
produced **pre-RoPE**: the indexer consumes it first, then `Attention`
rotates it and writes the cache (`model.py:750-762`):

```
apply_rotary_emb(latent[..., -64:], freqs_cis[2j])       # group j at position j*ratio
fp4_act_quant(latent, block=16, inplace=True, scale=e4m3) # §7: E4M3 scales, groups of 16
compress_kv_cache[B, j] = latent                          # bf16 storage, fp4-snapped values
```

Ratio 1 (L20) is a plain projection, no gate, no fp32, no state
(`model.py:461-462`): `latent = norm(wkv(x))` with `wkv` **bf16**; one
latent per token at position `j = p`; same post-RoPE fp4(16, e4m3) cache
write. Layers 21-39 re-source KV from L20's per-token cache.

Layers 0-1 (`compress_ratio 0`) never touch compressed KV: they attend over
the sliding window only, with theta-10000 non-YaRN RoPE (§3). Consumers
never run a compressor; they read `shared_attn.compress_kv[:B, :compress_len]`
where `compress_len = (p+1) // ratio` with their own ratio (2 for L3-19, 1
for L21-39) (`model.py:744-764`).

## 6. MoE

Per layer (backbone): gate weight `[384,5120]` bf16, `bias [384]` fp32,
`bias_vl [384]` fp32 (VL bias exists because vision is enabled;
text tokens use `bias`). Routing experts FP4; shared expert FP8.

Gate (`model.py:794-829`) — all fp32:

```
scores = (x.float() @ gate.weight.float().T) / gate_temp        # gate_temp = 1.0
score_func "sqrtsoftplus": scores = sqrt(softplus(scores))
bias = image_token ? bias_vl : bias
indices = topk(scores + bias, 6)                # bias STEERS SELECTION ONLY
weights = scores.gather(1, indices)             # weights come from UNBIASED scores
weights /= (weights.sum(-1, keepdim) + 1e-20)   # norm_topk_prob; 1e-20 literal, NOT norm_eps
weights *= route_scale (1.5)
```

Expert GEMM structure (`model.py:832-853`), per selected expert, fp32
intermediate:

```
gate = w1(x);  up = w3(x)                        # FP8-act × FP4-weight GEMMs, bf16 out
gate, up → .float()
swiglu_limit = 10: up = clamp(up, -10, 10);  gate = clamp(gate, max=10)   # asymmetric
h = silu(gate) * up
h = h * routing_weight                           # [n_sel, 1] broadcast over inter_dim
y += w2(h.to(bf16))                              # cast to bf16 BEFORE w2
```

MoE assembly (`model.py:891-906`): accumulator `y` is **fp32**; each local
expert's bf16 output is added into it; TP adds an fp32 all-reduce; then

```
y += shared_experts(x)      # shared expert = same Expert class, FP8 weights, swiglu_limit 10,
                            # no routing weight, computed on ALL tokens
return y.to(bf16)
```

FP32 rounding expectations called out in code: gate scoring fp32
(`model.py:813`), norm denominator literal `1e-20` (`model.py:827`), silu
product fp32 with bf16 cast only before `w2` (`model.py:853`), fp32 expert
accumulator (`model.py:895`), fp32 all-reduce in `RowParallelLinear`
(`model.py:274`).

Expert weight shapes (per expert): `w1/w3 [2304,5120]`, `w2 [5120,2304]`.
MTP MoE identical structure with 128 routed experts, top-3.

## 7. FP8 / FP4 quantization and GEMM semantics

Constants (`model.py:26-30`): `fp8_block_size = 32`, `fp4_block_size = 32`,
scale dtype `torch.float8_e8m0fnu`, storage `torch.float8_e4m3fn`.

### act_quant (FP8, dynamic, block 32) — `kernel.py:40-124`

Per row, per 32-element group along the last dim, fp32 compute:

```
amax = max|x_i| over group, clamped to minimum 1e-4
round_scale (ue8m0): s = 2^ceil(log2(amax / 448))       # kernel.py:22-37, power-of-2 only
y_i = clamp(x_i / s, -448, 448) cast fp8
```

`inplace=True` (used for every cached K/V and for the `o` before wo_a) does
fused quant→dequant back to bf16 in place:
`y = bf16(fp32(fp8(clamp(x/s))) * s)`. Activations feeding an FP8 or FP4
weight are quantized non-inplace: `linear()` always pairs a quantized weight
with a ue8m0-scaled fp8 activation (`model.py:181-207`).

### fp4_act_quant — `kernel.py:127-204`

Block 32, amax floor differs by scale dtype (`kernel.py:160-166`):

```
E8M0 path (indexer q/k):  amax = max(amax, 6·2^-126); s = 2^ceil(log2(amax/6))
E4M3 path (compressed KV): amax = max(amax, 6·2^-9); s = e4m3(amax/6)     # NOT power-of-2
y_i = clamp(x_i / s, -6, 6) → fp4
```

Compressed KV additionally uses **block 16**, not 32 (`model.py:761`).
`inplace=True` dequantizes back to bf16.

### FP8 GEMM — `kernel.py:207-307`

`C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T`, K iterated in blocks of 32:

```
acc_fp32 += (Σ_k-block fp8·fp8 gemm, fp32 accumulator) * s_a[m, k] * s_b[n//32, k]
```

Per K-block partial products are scale-corrected then accumulated in fp32
(separate accumulator, `kernel.py:242-269`); output bf16.

### FP4 expert GEMM — `kernel.py:477-591`

`C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T`, activation block 32 (ue8m0), weight
block 32 (ue8m0), K iterated in 32-blocks:

```
per k-block: cast B_fp4[n, 32k] → fp32 → fp8;   fp8×fp8 gemm (fp32 acc)
acc_fp32 += gemm * s_a[m, k] * s_b[n, k]
```

Output bf16. FP4 nibble packing (`convert.py:13-34`): stored `[N, K//2]`
int8; **low nibble (bits 0-3) = even K element, high nibble (bits 4-7) = odd
K element**. Decode table e2m1:
`[0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]` (sign =
bit 3). `convert.py` keeps packed int8 bytes and reinterprets as
`float4_e2m1fn_x2` (`convert.py:181`); an fp8-expert fallback path
(`cast_e2m1fn_to_e4m3fn`) exists but is not used for the 1017 fp4 release.

### Weight scales

FP8 weights: `scale [ceil(out/32), ceil(in/32)]` ue8m0 (block 32×32).
FP4 expert weights: `scale [out, in/32]` ue8m0 (block 1×32 along K).

## 8. Engram (layers 1 and 14)

Runs on the hc stream *before* the block (`model.py:1264-1266`). Two
independent tables (one per layer), each `[num_embeddings, 256]` fp8 with
`[num_embeddings, 8]` ue8m0 scales (rows dequantized on lookup,
`model.py:296-325`). Table rows: L1 = 384006168, L14 = 384016682.

Hash (`engram.py:129-184`), per position, on **compressed** token ids
(`token_map`, `engram.py:17-61`: NFKC→NFD→StripAccents→Lowercase→whitespace
collapse→Strip, partial-UTF8 byte tokens keyed by raw form; compressed vocab
size must equal 99092):

```
cache[p] = token_map[input_id]                     # int64 ring over the whole sequence
tokens[shift] = cache[p - shift]  for shift = 0..3  # clamped at sequence start;
                                                   # DEAD (-1) at image spans stops lookback
blocked positions → pad_id = token_map[2]
per layer l, per n-gram size i = 1..3, per head h = 0..7:
    rolling = tokens[0] * mult[l,0] XOR tokens[1]*mult[l,1] XOR ... XOR tokens[i]*mult[l,i]
    hash[l][i][h] = rolling % prime[l][i-1][h] + offset[l][i-1][h]
# result: hash_ids [B, S, 2, 24]; col order: (i=1: 8 heads), (i=2: 8), (i=3: 8)
```

Multipliers (`engram.py:64-83`): per layer, `np.random.default_rng(10007 *
layer_id).integers(0, bound, 4)` with `bound = max(1, (int64max //
compressed_vocab) // 2)`, forced odd (`v*2+1`). Primes (`engram.py:86-126`):
per (layer, n-gram size, head), the next unused prime starting the search at
`engram_vocab_size - 1` (16e6). All deterministic given the tokenizer.

Merge into the stream (`model.py:328-365`):

```
e   = embed_lookup(hash_ids)             # [B,S,24,256], fp8 rows dequantized to bf16
kv  = wkv(e.flatten(-2))                 # Linear(24*256=6144 → 5120*(4+1)=25600) fp8
key, value = split(kv, [20480, 5120])    # key → [B,S,4,5120]
w   = q_weight * k_weight                # [4,5120], only ever used as a product
dot = Σ_d (h * w * key) * rsqrt(mean(h²)+eps) * rsqrt(mean(key²)+eps) * 5120^-0.5
      # normalized per (token, copy) over dim; eps = norm_eps = 1e-20
gate = sigmoid(copysign(sqrt(clamp(|dot|, 1e-6)), dot))    # signed sqrt
gate = 0 where token_mask is False (image spans)
h_out[j] = h[j] + gate[j] * value        # each hc copy gets its own gate times ONE shared value
```

## 9. Drafter (mtp.0-2, DSpark)

Three extra blocks (layer ids 40-42, `compress_ratio 0`), each a full Block:
hc params, window-128 sliding attention (`DSparkAttention`), MoE with 128
experts top-3. Extra modules: `mtp.0`: `main_proj Linear(15360→5120)` fp8 +
`main_norm RMSNorm(5120)`; `mtp.2`: `norm RMSNorm(5120)`, `markov_head`,
`confidence_head` (`model.py:1103-1121`). Embeddings/head tied to backbone
(§1).

Chaining, one decode step at position `p` (`model.py:1277-1285`):

```
main_x = main_norm(main_proj(main_hidden))        # main_hidden = cat of hc-means at L37/38/39 inputs
draft_ids = [last_token, noise, noise, noise, noise]   # block_size 5, noise id 128799
h = embed(draft_ids) → [B,5,5120] → repeat → [B,5,4,5120]
pre_mix = identity
for k in 0..2: h, pre_mix = mtp[k](h, p, pre_mix, main_x)
return mtp[2].forward_head(h, pre_mix, last_token)
```

`DSparkAttention.forward` (`model.py:1034-1077`), per MTP layer, decode:

```
main_kv = kv_norm(wkv(main_x))  → RoPE at p → fp8 snap → window ring[p % 128]   # main-model tokens only
draft q = q path of draft_ids rows, RoPE at positions p+1 .. p+5
draft kv = kv_norm(wkv(draft_x)) → RoPE at p+1..p+5 → fp8 snap (transient, NOT cached)
topk_idxs = [0 .. min(128, p+1)-1] + [128 + 0..4]    # all filled window slots + all 5 draft slots,
                                                     # every draft row sees ALL draft rows (model.py:1022-1031)
o = sparse_attn(q, cat([window, draft_kv]), sink, idxs, 512^-0.5)
# then inverse-RoPE, fp8-snap, wo_a einsum, wo_b — identical to §3
```

Prefill (`start_pos == 0`): each MTP layer only seeds its window ring from
`main_x` and passes through; no drafts (`model.py:1125-1128`).

Head (`model.py:1140-1159`):

```
x = hc_pre(h, pre_mix)                            # [B,5,5120]
logits = head(norm(x), full_logits=True)          # [B,5,129280] fp32
output_ids[:, 0] = last_token
for i in 0..4:
    markov_logits, markov_embed = markov_head(output_ids[:, i])
        # markov_head: embed(vocab,256) → ParallelHead(vocab,256): logits_bias [V], embed [256]
    logits[:, i] += markov_logits
    output_ids[:, i+1] = sample(logits[:, i], temperature)   # Gumbel-max, same sample()
    markov_embeds[i] = markov_embed
confidence = confidence_head(cat([x, markov_embeds], -1))     # Linear(5376→1) fp32 → [B,5]
return (output_ids [B,6], logits [B,5,V], confidence [B,5])
```

`output_ids = [anchor, d1..d5]` proposes tokens for positions `p+1..p+5`.
**Rejection/acceptance is NOT implemented in the reference** — `forward_spec`
is never called by a verify loop (`model.py:129-131`); a server must define
it (see Open questions).

## 10. Sampling / generation (`generate.py`)

- Tokenizer: `AutoTokenizer.from_pretrained(ckpt_path)` — HF fast tokenizer
  from `tokenizer.json` / `tokenizer_config.json` inside the *converted*
  checkpoint dir (copied there by `convert.py:184-189`)
  (`generate.py:128`). `eos_id = tokenizer.eos_token_id` (= 1).
- Sampling is **temperature only** — no top-p, top-k, or repetition penalty
  anywhere. `sample()` (`model.py:1288-1295`) is Gumbel-max:
  `temperature==0 → argmax`; else
  `argmax(softmax(logits/max(T,1e-5), fp32) / Exp(1))`. RNG seeded
  `torch.manual_seed(33377335)` (`generate.py:120`); bit-exact replay needs
  the same generator stream.
- Batch decode (`generate.py:28-87`): prompts right-padded with `-1`
  (padding never enters the model); first pass prefills
  `[:min_prompt_len]`; positions still inside a prompt are teacher-forced
  (`prompt_mask`); a sequence finishes when a generated token == eos outside
  the prompt; loop ends when all finish; completions truncated at the first
  eos.
- Prompt templates: `encode_messages(messages, thinking_mode)` /
  `parse_message_from_completion_text` from an `encoding` package
  (`generate.py:15-24`) — Harmony-style chat, `thinking_mode` ∈
  {chat, thinking}. That package is **not present** in the reference
  directory (see Open questions).
- Special tokens: eos 1, bos 0, pad 2, image 129264, dspark noise 128799.
- Interactive mode sets `max_seq_len = 64k`, `max_batch_size = 1`
  (`generate.py:124-126`); `temperature` comes from the CLI (README suggests
  0.6) and is injected into `ModelArgs` — the *drafter* samples with the
  same temperature (`model.py:1156`).
- A server must replicate: teacher-forcing boundaries, per-step
  `tokens[:, prev_pos:cur_pos]` slicing with `start_pos = prev_pos`, and
  that the model itself returns sampled ids (logits are also returned).

## 11. Load plan

| tensor (per backbone layer unless noted) | on disk | on GPU | host transform |
|---|---|---|---|
| `embed.weight` | bf16 `[129280,5120]` | bf16 | vocab shard for mp>1 (32320 rows/rank @TP4) |
| `head.weight` | bf16 `[129280,5120]` | **fp32** (`model.py:1008`) | vocab shard; bf16→fp32 promote |
| `norm.weight` (final), `attn_norm`, `ffn_norm` | bf16 `[5120]` | bf16 | none |
| `attn.wq_a/wkv` weight | fp8 e4m3 | fp8 e4m3 | none |
| `attn.wq_b` weight | fp8 e4m3 | fp8 e4m3 | column shard (heads) for mp>1 |
| `attn.wo_b` weight | fp8 e4m3 | fp8 e4m3 | row shard (K dim) for mp>1 |
| all FP8 `*.scale` | ue8m0 `[ceil(out/32),ceil(in/32)]` | ue8m0 | shard matching weight |
| `attn.wo_a.weight` | fp8+scale in HF export | **bf16 dense `[8192,4096]`** | **dequantize on host** with 32×32 (or 128×128) scale (`convert.py:157-173`); view `[8,1024,4096]` |
| `attn.q_norm/kv_norm.weight` | bf16 `[1280]`/`[512]` | bf16 | none |
| `attn.attn_sink` | fp32 `[64]` | fp32 | head-shard for mp>1 |
| `ffn.gate.weight` | bf16 `[384,5120]` | bf16 | none (replicated) |
| `ffn.gate.bias`, `bias_vl` | fp32 `[384]` | fp32 | none |
| `experts.E.w1/w3` | fp4 packed int8 `[2304,2560]` | fp4x2 (same bytes) | nibble order: low=even K, high=odd K; expert index shard for mp>1 (96/rank @TP4) |
| `experts.E.w2` | fp4 packed `[5120,1152]` | fp4x2 | same |
| `experts.E.*.scale` | ue8m0 `[out, in/32]` | ue8m0 | same |
| `shared_experts.w1/w3` | fp8 `[2304,5120]` + scale | fp8 | none (replicated) |
| `shared_experts.w2` | fp8 `[5120,2304]` + scale | fp8 | none |
| `hc_attn_fn/hc_ffn_fn` | fp32 `[24,20480]` | fp32 | none |
| `hc_attn_base/hc_ffn_base` | fp32 `[24]` | fp32 | none |
| `hc_attn_scale/hc_ffn_scale` | fp32 `[3]` | fp32 | none |
| compressor `wkv`,`wgate` (L2,8,14) | fp32 `[512,5120]` | fp32 | none |
| compressor `wkv` (L20) | bf16 `[512,5120]` | bf16 | none |
| compressor `norm.weight` | bf16 `[512]` | bf16 | none |
| indexer `wq_b` | fp8 `[4096,1280]` + scale | fp8 | head shard for mp>1 |
| indexer `weights_proj` | bf16 `[32,5120]` | bf16 | head shard for mp>1 |
| indexer `wk` (kv-source layers) | bf16 `[128,512]` | bf16 | none |
| indexer `k_norm.weight` | bf16 `[128]` | bf16 | none |
| engram `embed.weight` (L1, L14) | fp8 `[384006168,256]` / `[384016682,256]` | fp8 | row shard ceil(rows/mp); tail pad weight=0, scale=1 |
| engram `embed.scale` | ue8m0 `[rows,8]` | ue8m0 | same |
| engram `wkv` | fp8 `[25600,6144]` + scale `[800,192]` | fp8 | none (replicated) |
| engram `q_weight`/`k_weight` | bf16 `[4,5120]` | bf16 | none |
| mtp.0 `main_proj` | fp8 `[5120,15360]` + scale | fp8 | none |
| mtp.0 `main_norm`, mtp.2 `norm` | bf16 `[5120]` | bf16 | none |
| mtp.2 `markov_head.embed` | bf16 `[129280,256]` | bf16 | vocab shard |
| mtp.2 `markov_head.head` | bf16 `[129280,256]` | fp32 | vocab shard + promote |
| mtp.2 `confidence_head.proj` | fp32 `[1,5376]` | fp32 | none |
| mtp.K MoE gate/experts/shared | as backbone but 128 experts | same | expert shard 32/rank @TP4 |
| engram `token_map`, primes, offsets, multipliers | — (derived) | int64 buffers | **host-computed** from tokenizer: compressed map (assert 99092), primes from 16e6, multipliers seed `10007*layer` (`engram.py:64-83`) |
| RoPE `freqs_cis` | — | complex64 `[max_seq_len,32]` ×2 variants | precompute per layer class (§3) |

Runtime (non-persistent) buffers: `window_kv_cache [B,128,512]` per
attention layer (43 of them), `compress_kv_cache [B, seq//ratio, 512]` ×4,
indexer `k_cache [B, seq//ratio, 128]` ×4, compressor `kv_state/score_state
[B,2,512]` ×3, engram hash `cache [B, max_seq_len]` int64.

## 12. Open questions / ambiguities

1. **Speculative verification is unspecified.** The reference implements
   the drafter forward only; no code calls `forward_spec` in a verify loop
   (`model.py:129-131`). Exact acceptance rule (greedy match? temperature
   resample? how `confidence [B,5]` gates the draft budget), block rejection
   on first mismatch, and cache rewind semantics are all server-side and
   undocumented here.
2. **Prompt template package missing.** `generate.py` imports
   `encode_messages`/`parse_message_from_completion_text` from `../encoding`
   (Harmony), which is absent from the reference tree. Special-token layout
   of chat/thinking prompts cannot be recovered from this code.
3. **RoPE theta asymmetry is deliberate-looking but unexplained**: layers
   2-39 use `compress_rope_theta` 160000 + YaRN for *all* their rope —
   including the sliding-window K and Q — while layers 0-1 and the MTP
   layers use plain theta 10000 without YaRN (`model.py:681-688`). Nothing
   comments on whether window attention was trained under 160000.
4. **Chunked prefill undefined.** `generate.py` prefills in one chunk;
   `get_window_topk_idxs` and the compressor's `start_pos == 0` branch
   assume a fresh ring/state. Behavior for multi-chunk prefill (ring
   seeding, compressor remainder, indexer `offset` ≠ 128) must be derived,
   not read.
5. **Engram sharding padding rows** (weight 0, scale 1) can be hashed into
   only if a hash+offset lands in the padded tail; the code relies on
   `offsets + prime ≤ unpadded rows` but never asserts it.
6. **Top-k tie-breaking**: `topk` at `-inf` (early decode, short contexts)
   and gate-expert ties rely on torch's topk order; a reimplementation must
   pick a deterministic rule and it may not match torch bit-for-bit.
7. **`sparse_attn` summation order** (64-position blocks, online rescale)
   differs from a naive softmax; exact fp32 equality with the reference
   requires the same block order (block=64, `num_stages=2`).
8. **`hc_mixes` normalization** uses one RMS over the flattened 4×5120
   stream rather than per-copy; confirmed by code (`model.py:953-956`) but
   worth a numerics test against the checkpoint.
9. **`wo_a` block size**: convert accepts 32×32 or 128×128 scales
   (`convert.py:164`); the 1017 export's actual block size should be read
   from the real checkpoint (the census tool already validates this).
10. **Gumbel-max sampling** consumes torch RNG; "temperature 0 output
    identity" gates are fine, but sampled-temperature replay cannot be
    bit-exact without replicating torch's Philox stream.
11. **DSpark draft rows are mutually non-causal** (every draft row attends
    all 5 draft KVs, `model.py:1022-1031`) — intended per the self-attention
    over the draft block, but worth confirming against the training-side
    drafter if one exists.
12. **`fp4_act_quant` E4M3 scale path floors amax at `6·2^-9`** "to keep an
    all-zero group's scale nonzero" (`kernel.py:161-163`) — the exact floor
    value matters for bit-exact compressed KV.
