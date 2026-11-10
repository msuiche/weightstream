# weightstream kernel plan

Every kernel lands with the same gate: output compared against the pure-Rust
CPU reference (`ferroseek-ref`) on identical inputs, tolerance policy per
kernel below. Gates run on a DGX Spark via
`cargo test -p ferroseek-cuda --features cuda`.

Order is dependency order: later kernels reuse earlier ones' device buffers
and conventions.

| # | kernel | spec | gate |
|---|---|---|---|
| 1 | `fp4_dequant` (done) | spec §7 | bit-exact vs `quant::dequant_fp4_block` |
| 2 | `rmsnorm` | eps 1e-20, one statistic over the full row; the hc variant needs one statistic over the *flattened 4×5120* stream (spec §2, open q8) | fp32 exact |
| 3 | `rope` / `rope_conj` | per-layer theta class (10000 plain vs 160000+YaRN, spec §3); conjugate variant for the attention output tail | fp32 exact, incl. YaRN ramp edges |
| 4 | `act_quant_ue8m0` + inplace snap | spec §7: amax floor 1e-4, pow2 ceil scale, clamp ±448 | bit-exact |
| 5 | `fp8_gemm` | block-32 K-loop, per-block scale correction into a separate fp32 accumulator (spec §7) | fp32 max-ulp bound vs CPU gemm |
| 6 | `fp4_act_quant` | E8M0 path (indexer, amax floor 6·2^-126) and E4M3 path (compressed KV, block **16**, floor 6·2^-9) | bit-exact |
| 7 | `fp4_expert_gemm` | FP8 act × FP4 weight, fp4→fp32→fp8 per K-block (spec §7) | fp32 max-ulp bound |
| 8 | `hc_mixes` + `sinkhorn20` | 24 mixes, sigmoid variants, exact 20-iteration Sinkhorn with eps placement (spec §2) | fp32 exact |
| 9 | `sparse_attn` | online fp32 softmax, 64-row blocks, attn_sink-as-zero-key, −1 index masking (spec §3); block order matters for bit-equality (open q7) | fp32 1e-6 rel |
| 10 | `indexer_score` | fp4-snapped q/k dot, relu, weights_proj combine (spec §4) | fp32 exact |
| 11 | `topk_two_level` | block-max candidates with pinned newest block, then sorted top-512 with −1 drop (spec §4); deterministic tie rule, documented divergence (open q6) | set-equality vs CPU |
| 12 | `compressor` | fp32 kv/score state, softmax over exactly 2 slots, pair-completion write (spec §5) | fp32 exact |
| 13 | `engram_lookup` | fp8 row gather + dequant, gate per spec §8 | fp32 exact |
| 14 | `moe_route` | sqrtsoftplus, bias-steers-only top-6, 1e-20 norm, ×1.5 (spec §6) | fp32 exact |
| 15 | `swiglu_limit10` | asymmetric clamps, bf16 cast before w2 (spec §6) | bit-exact |
| 16 | `gumbel_max` | greedy argmax path first; torch Philox replay explicitly out of scope (open q10) | exact ids |

Non-goals for the first pass: CUDA graphs, L2 prefetch, persistent kernels,
fused layer pipelines. Decode latency is memory-bound; correctness first,
fusion behind a profile.

## Device-buffer conventions

- Weights stay quantized on GPU exactly as on disk (spec §11 load plan);
  `wo_a` is the one exception (host-dequantized to bf16).
- One `DeviceArena` per lifetime class: weights (forever), KV caches
  (per-sequence), step scratch (reused every decode step).
- fp32 control/state (compressor state, hc params, sinks, gate bias) lives
  in pinned host-visible memory where kernels read single values; bulk math
  in device memory.
- Ring buffers (window KV, index caches) are pre-allocated at
  `max_seq_len` and never reallocated mid-sequence.

## M-batch decode (decode_m, 2026-10-27)

`GpuModel::decode_m(&[u32])` runs one teacher-forced pass over m known
tokens (prefill chunks, speculative verification) instead of m
sequential `decode` calls. Batched: embedding, every GEMM (fp8_gemm's
grid is (N, M) — same per-block reduction per output, so results are
bit-identical to the m=1 GEMV path), hc (all kernels already grid over
n_tokens), rmsnorm_m, moe_gate_m, rope_uniform_pos_buf, sparse_attn_m
(grid (n_heads, m) over per-position causal idx lists, -1-padded).
Sequential per position: ring writes, the CSA compressor state machine,
indexer key append + selection, engram hashing — the position-dependent
state, evolved in order so the post-pass model state equals m
sequential decodes. Two batching pitfalls found by the gate: the
cand_src layer's candidate blocks must be kept per position (the
source's sequential loop would otherwise leak the last position's set
into every consumer), and the h-init's ones upload must precede the
first hc_post.

Gate (`decode_m_matches_sequential_decode`): 8-position batch vs
sequential — bit-exact (0.0) on every position AND on two further
sequential continuation steps; wall 8.0 ms vs 13.9 ms (1.7x) on the
`decode_prefill` uses the sequential loop by default because the real
64-token story stream diverged from M-batch prefill even though the
synthetic gate above is bit-exact. `FERROSEEK_MBATCH_PREFILL=1` opts into
decode_m chunks for A/B; drafter-loaded models always use sequential
prefill. TP4 is supported: local head/expert shapes, per-position
index/attention/MoE all-reduces and the vocab all-gather. Sparse attention
runs in causal order so later window writes cannot clobber earlier rows, and the final
top-k list is published for sequential continuation. Real-ring A/B:
132-token prefill 9.625 s vs 12.964 s sequential (25.8% faster), with
identical output on all four ranks. New kernels gated bit-exact vs
sequential launches: sparse_attn_m, rope_uniform_pos_buf, moe_gate_m,
rmsnorm_m (same kernel, grid.x = rows).

## Fused sparse-MLA decode (2026-11-07)

For the real DS4.1F shape (16 local heads, head dimension 512), the default
attention path uses `tilelang/sparse_mla_decode.cu`: a split kernel over FP8
SWA records and FP4 compressed records, followed by the merge kernel with
conjugate output RoPE and FP8 activation snapping. Set
`FERROSEEK_SPARSE_FP32=1` to use the older FP32 gather/attention path for
A/B debugging.

The kernel's `SwaLengths` is a **valid-prefix length**, not a physical ring
watermark. GPU index handoff therefore compacts the reference's oldest-first
window list and pads the tail with `-1`; compressed indices follow the window
prefix. Cosine and sine tables are separate `[position][32]` arrays, matching
`rope.cu` (the merge kernel must not use a 64-float interleaved stride).

Verification on the real 1017 checkpoint: four-rank TP4 greedy decode emits
` Paris.` identically on every rank. A CUDA memcheck run during bring-up was
clean. The `sparse_mla_packed_window_prefix` regression test covers the
one-valid-key prefix case that the physical-ring layout used to mask out.
