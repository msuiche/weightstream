# Performance gap: weightstream vs the spark-ds41f reference stack

Handoff for the agent picking up performance work. Everything here is
measured on the same hardware; sources are cited per claim. Repo state:
`main` @ `3e9aa04` (2026-11-07).

## Hardware and model

- 4× DGX Spark GB10 (122 GiB usable RAM/node), driver 580.178.04, CUDA
  13.0. Clocks are capped by design (thermal policy) on all nodes — all
  numbers below, both stacks, were measured under the cap.
- Model: DeepSeek-V4.1-Flash-1017, 510 GB / 48 shards, HF snapshot
  `1bae0e1088993b9b82de08f59de5be5a34e42170` on `spark-4687` at
  `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1…/`
  (48/48 shards, 510.297 GB resolved — verified 2026-10-30; measure
  with `du -L`, the blobs are a shared pool).
- Ring: `spark-4687` (head), `spark-5bc3`, `spark-7425`, `spark-7275`;
  all four GPUs healthy and TP4 validated on 2026-10-31. SSH = 192.168.1.x,
  RoCE = 192.168.100.x.

## Measured numbers (same hardware, same model)
## The GLP-27 production baseline (2026-10-31, measured)

The dsv41f lane with GLP-27 fold-site steering armed (α=3.5), TP4 ring,
DSpark speculative decoding, thermal cap on. Measured by
`bin/spark bench` on the four-node ring, zero foreign requests:

| metric | GLP-27 lane | weightstream upload TP4 (2026-11-07, no-spec) |
|---|---|---|
| decode t/s, single stream (C1) | **58.64** (tpot p50 16.57 ms) | **19.24** (52.0 ms/token, 2026-11-10) |
| decode t/s, aggregate (C16) | **215.94** (tpot p50 73.86 ms) | — (single-request lockstep) |
| cold prefill | **5.3k t/s** (32K / 131K) | not measured at those sizes |

The later LocalArena engine showdown measured the same spark-ds41f
TileLang + DSpark stack at **85 tok/s single-stream**, **262 tok/s at eight
users**, and **6,636 tok/s cold prefill**. Against that newer 85 tok/s
reference, weightstream's current 19.24 tok/s is about **4.4× slower**.

This is the apples-to-apples bar for weightstream's upload-TP4 work: same
hardware, same checkpoint, same DSpark spec-decode, steering armed. The
2026-10-17 reference below is the UNSTEERED stock 1022 lane (a different
graph); the GLP-27 numbers include the fold materialization + companion
writer overhead.

Upload validation used `72323e6` plus the worker-barrier fix `b5e6f61`,
the real 1017 checkpoint, upload mode, and a drained ring. All four ranks
completed `The capital of France is` → ` Paris.` with 620 collectives.
The older C1 measurement used one model load, a 64-token warm-up request,
then 64 greedy tokens for `The capital of Italy is`; mean 97.635 ms/token,
p50 97.196 ms, range 84.524–112.863 ms. GLP was 5.73× faster on that
measurement and is about 3.9× faster at the current 65–67 ms. The current
story decode used `Write a long story about a dragon` to avoid an immediate
EOS and produced the same 20-token stream with and without DSpark. This is
a steady-state C1 data point, not a prompt-matched GLP rerun.
weightstream serving serializes requests. No 32K/131K prefill comparison
was run. Raw logs: `spark-*:/tmp/tp4-upload-rank{0..3}.log` (validation)
and `spark-*:/tmp/tp4-bench-rank{0..3}.log` (benchmark).

DSpark correctness landing (`abf63f2`, `a91917a`, `67402d5`): full
five-draft-plus-anchor acceptance and exact `--no-spec` output identity passed
on all four ranks. The same C1 method above then measured **8.97 tok/s** for
the trained drafter (64 Italy tokens, 7.137 s, mean 111.509 ms/token), versus
**10.24 tok/s** without speculation: **0.876×**, a **12.4% regression**.
Target verification is sequential; no speedup is claimed. Raw logs on the
driver: `/tmp/tp4-c1-dspark-spark-*.log`.

### Speculative-decode note (2026-11-13)

The current engine implements DSpark but the speedup path is disabled by
acceptance gating: `speculate` runs `decode(anchor)`, drafts five tokens,
then **only uses `decode_m` batch verification when the previous step had
`spec_accept >= 5`** (all five drafts accepted). Otherwise it verifies
drafts one-by-one with sequential `decode` calls. `WEIGHTSTREAM_SEQ_SPEC=1`
forces the sequential fallback. This is why DSpark is slower than no-spec
(8.97 tok/s vs 10.24 measured 2026-10-31): ordinary steps never reach the
batched fast path, and a rejection costs the whole anchor step plus every
verified draft. The baseten `deepseek-ultrafast` project shows the intended
shape: 7-token draft blocks + anchor verified as an 8-row batch inside one
CUDA graph, yielding ~5.6 accepted tokens/step. Before any graph work, our
goal is batched verification with a correct rollback boundary, not launch
overhead.

**Gate fix measured and reverted (2026-11-13):** the fast path already
existed in `GpuModel::speculate`, but was gated on `spec_accept >= 5`, which
requires the previous step to accept all five drafts. I changed the gate to
use `decode_m(batch)` whenever at least two drafts fit, preserving the
snapshot/rollback boundary. The four-rank real-checkpoint result was
**slower, not faster**: 12 tokens in ~2.95 s (~4.07 tok/s), versus 1.27 s
(~9.42 tok/s) sequential. `decode_m` costs ~240–370 ms per speculative step
and ring collectives rose from 2,061 to 6,551. Acceptance did not improve.
The gate is reverted; `WEIGHTSTREAM_SEQ_SPEC=1` remains the sequential
fallback, and batch verification is **not viable** until the M-batch
attention/indexer path gets much cheaper or acceptance is substantially
higher. CUDA suite passes 32/32 after the revert.

**Next-step decision (2026-11-13):** do **not** retry CUDA graphs now and do
**not** use the full five-draft `decode_m` verifier. The architecture
comparison with `basetenlabs/deepseek-ultrafast` shows why it wins: its
normal speculative step verifies a draft block through the same fast forward
path and only then wraps it in a graph. Our `decode_m` path is a separate
expensive M-batch path, so batching costs more than it saves. The next
highest-confidence experiment is a **2-draft verification path** that uses
the existing `decode_m` machinery only for pairs, because the measured
acceptance pattern sometimes reaches 2–3 tokens/step but full five-draft
batching is unambiguously slower. If 2-draft verification still loses to
sequential DSpark and eager, leave speculation disabled for serving and
pursue attention/step-cost work instead.

**2-draft verification measured and rejected (2026-11-13):**
`WEIGHTSTREAM_SPEC_PAIR=1` used `decode_m` only for pairs and preserved the
same snapshot/rollback boundary. Four-rank output identity passed
(` who hasisas (... I'm not sure what you mean`), but C1 remained slower than
sequential DSpark: **12 tokens in ~2.0 s (~5.98 tok/s)** versus **1.27 s
(~9.42 tok/s)**, with 3,719 collectives versus 2,061. It is better than the
full five-draft batch (~4.07 tok/s) but still worse than sequential and far
worse than eager (~16.85 tok/s). The pair knob was removed after measurement;
the default `spec_accept >= 5` gate remains. DSpark batching is now rejected
at both 2 and 5 drafts on this hardware/config.

**Attention-side investigation (2026-11-13):** with the ring drained, a
12-token eager story profile on all four ranks showed attention remains the
largest wall cost: **30.2–50.3 ms/step** by rank, versus MoE **20.7–21.5 ms**.
The output tail is strictly dependent (`sparse_mla_decode → wo_a →
emit8_packed → wo_b`), and the `wo_b` all-reduce is consumed immediately by
`hc_post`, so overlapping it would change model state. No safe small
attention overlap was implemented. The next attention win needs kernel
fusion or a collective-schedule redesign, not another launch-order variant.

**Fused attention tail design rejected before build (2026-11-13):** fusing
`fp8_wo_a_gemv` with `emit8_packed` is not a safe incremental patch. The
GEMV uses 256-thread block reductions per output; emit8 uses one 32-thread
block per 32-value group. A naive merged kernel either changes dot-product
accumulation order or cannot see the whole 32-value group. A correct fused
kernel needs a warp/sub-warp-aware redesign and a new numerics proof, not a
quick merge. The sketch was removed before build; no code change remains.

**Exact-preserving fusion proof (2026-11-13):** a fused `wo_a + emit8`
kernel cannot preserve the existing dot-product order and also quantize a
32-value group. The current GEMV uses one 256-thread block per output; each
thread adds columns strided by 256 and `block_reduce_sum` defines the exact
accumulation tree. A fused 32-output block would need each warp/sub-block
to compute multiple outputs, changing every output's reduction order.
Keeping the existing 256-thread-per-output geometry leaves no block-local
32-value group to quantize. Therefore exact-preserving fusion is impossible
without accepting a new reduction order and a new numerics gate. This is a
larger kernel project, not a safe optimization step.

**wo_b collective inspection (2026-11-13):** the attention `wo_b`
all-reduce cannot reuse the MoE overlap trick. MoE overlaps the host
all-reduce with the independent shared expert; attention has no independent
compute after `wo_b` — the reduced value is consumed immediately by
`hc_post`, and the FFN HC block depends on that post value. An async
`wo_b` overlap would therefore change state scheduling, not just timing.
Only removing the final HtoD is possible, but that saves a small copy and
does not address collective latency, while complicating GLP/dump semantics.
No code change was made.

**Aggregate batching inspection (2026-11-13):** serving is one request at a
time by design because TP4 is lockstep — every rank decodes the same
request at the same step. `decode_m` batches **multiple tokens from one
sequence**, not multiple independent requests: it has one KV state, one
prompt prefix, and one position vector per engine instance. Independent
C16 requests need per-request caches/positions and a scheduler that can
advance only requests whose states are compatible. There is no safe
incremental change here; a real batching design must be explicit. The
2026-11-13 review also found that independent-request batching cannot reuse
the current TP4 engine state: each request owns KV caches, logits, and
positions on device, while upload-mode TP4 already dedicates those buffers
to one sequence. Cloning them per request is infeasible within current
memory and changes the core engine design. Serving therefore remains
sequential by design; this is an architectural boundary, not an
implementation oversight.

## Decision boundary (2026-11-13)

All small optimization directions measured this session are now closed:

- CUDA graphs: attempted twice, neutral or wrong-output; fully reverted.
- DSpark batched verification: rejected at two and five drafts.
- Attention/collective overlap: structurally unsafe.
- Independent request batching: requires per-request engine state and a new
  scheduler.

The project therefore needs an explicit architectural choice before more
work:

1. **Serving throughput:** build a new multi-request engine/serving
   architecture, or use the reference vLLM stack for aggregate traffic.
2. **Single-stream latency:** undertake a real attention/kernel redesign,
   not another overlap/gate variant.
3. **Speculation:** revisit only after a cheaper verification architecture
   and substantially higher acceptance exist.

Until one of those is chosen, the correct serving configuration remains
eager/no-spec; serving keeps speculation disabled.

**Measured on a drained ring (2026-11-13):** DSpark on the dragon prompt
emitted 12 tokens in **1,274 ms (~9.42 tok/s)**. Acceptance pattern:
`1,1,1,1,1,1,3,2,1`; only two of nine steps accepted multiple drafts, so the
`spec_accept >= 5` batch gate never triggers. Rank 0 profiler over 21
speculative steps: attention 31.9 ms, MoE 23.8 ms, gate 0.5 ms, Engram 3.3 ms,
head 1.7 ms. The sequential verify path makes DSpark a clear C1 regression
versus the same eager decoder's ~59 ms/token. Output remained identical on all
four ranks: ` who hasisas (... I'm not sure what you mean`.

## Experiment log (2026-11-01) — do not duplicate blindly

- **Production mHC (`eb15ef9`) is rejected (2026-11-09).** Its
  `production_mhc` branch skipped `moe_step` on 38/40 TP4 layers. Restoring
  unconditional MoE did not fix the real story stream (`10.5. E undo
  Plate...` versus the stable stream) and did not improve latency. The
  earlier ~35 tok/s claim is therefore invalid; do not restore this path
  without a real four-rank known-answer gate.
- **Clean-build discipline is required on the ring.** Mixed lockfiles and
  stale CUDA artifacts produced `misaligned address` in single-node prefill
  and divergent TP4 output. A full source/lockfile sync plus
  `cargo clean --release -p weightstream-cuda -p weightstream-gpu -p weightstream`
  restored the known-answer single-node stream (` Paris. The Eiffel`) and
  identical TP4 output. Use `cargo build --release --features cuda --locked`
  on every rank and never compare binaries from mixed workspace states.
- **Validated stable baseline (`c5284e1` + GLP, 2026-11-09).** Full GPU
  suite 32/32; local workspace 68/68. The 16-token dragon story is identical
  on all four ranks, with steady tokens at 57–58 ms (~17.4 tok/s). Rank 0
  profile: attention 33.4 ms, MoE 21.1 ms, Engram 4.4 ms, head 1.8 ms,
  gate 0.5 ms per step. The two-token France probe produces ` Paris.` on
  every rank but is not a throughput benchmark.
- **Indexer replication: rejected for short-context C1.** The dragon prompt
  never exceeds `index_topk`, so the score all-reduce is not executed and
  the collective count is unchanged (81/step: two output reductions per
  layer plus head). Replicating all 32 heads only adds work; the story stays
  identical but latency does not improve. Revisit only with long contexts.
- **GLP kernel completion barrier.** `glp_project` now ends with
  `__syncthreads()`; without it, readback could observe only `y[0]` written.
  The regression test also synchronizes the CUDA stream before host
  readback. Full suite remains 32/32.
- **Validated kernel work (2026-11-09).** `#pragma unroll 8` on the HC dot
  loop preserves the story and cuts `hc_mixes` from 48.5 to 45.0 µs (~0.28
  ms/token). Fusing shared gate+up weights into one grid-72 TileLang launch
  preserves the story and saves ~0.44 ms/token. `act_quant_fp8_emit` now
  writes raw and aligned packed scales in one launch, eliminating all 360
  `pack_scale_words` launches/token and fixing the aligned-capacity bug
  (40 raw scale bytes require 88 bytes, not 80). Parallel Sinkhorn was
  rejected: 34.2 → 35.5 µs due to barrier overhead.
- **FP8 WO-A and graph verdict.** `fp8_wo_a_gemv` dequantizes each FP8
  weight to BF16 in-kernel, exactly matching the old BF16 GEMV bit-for-bit
  while preserving the story. Its 44.6 µs average replaces the old 65 µs
  WO-A average (roughly one-third less time). Shared and grouped-MoE CUDA
  graphs passed all tests but were performance-neutral (~54.4 ms with or
  without them) and were removed.
- **Shared/routed overlap.** After routed D2H, shared-expert kernels launch
  before the host all-reduce; reduced values return via stream-ordered async
  HtoD and shared output accumulates afterward. This preserves the exact
  story and reduces the no-spec median to about 54 ms/token. DSpark remains
  slower: acceptance stays below five, so verification is sequential
  (3-token rounds ~185 ms, 2-token rounds ~129 ms).
- **Dual-stream grouped MoE.** W1 runs on the main stream and W3 on the
  auxiliary stream after an explicit main→aux dispatch event; a second event
  joins them before SwiGLU. This preserves the story and lowers grouped-MoE
  time from ~21 to ~20 ms/token. Without the dispatch event, W3 races the
  routing metadata and the stream changes.
- **Q-A/WKV overlap.** Q-A stays on the main stream; WKV launches on the
  auxiliary stream after activation quantization and joins before KV
  normalization. The validated four-rank story remains identical and steady
  tokens reach **51.96 ms (~19.24 tok/s)**. Full GPU suite: 32/32.
- **Early shared-expert overlap: rejected.** Launching the complete shared
  expert on the auxiliary stream before grouped dispatch contends with W3,
  raises MoE from ~21 to ~26 ms, and does not improve steady token latency
  (~52.4 ms). The shared expert remains overlapped only with host all-reduce.
- **FFN CUDA graphs (2026-11-13, implemented; first TP4 attempt exposed a
  boundary mistake).** `GraphExec` in `weightstream-cuda/src/gpu.rs` plus
  `WEIGHTSTREAM_CUDA_GRAPHS=1` capture one FFN graph per
  `(layer, position slot)`: FFN HC block → `hc_pre` → `rmsnorm`, ending
  before the grouped MoE. Host collectives stay outside the graph;
  `reset()` clears captures; dump, streaming, drafter capture, and position
  0 never capture. The previous neutral experiment was structurally
  different: three tiny graphs per layer (attention, fold, MoE), so graph
  launches ate the launch savings and the TP4/`w1_tl` gate missed `w1_all`.
  Two corrections came from the first real TP4 run:
  1. **Graph boundary cannot include `moe_grouped_tl_step`** because its
     dual-stream event (`main → aux → main`) is not capturable with the
     legacy default stream; the first attempt panicked in
     `act_quant_fp8_emit` with `UnknownError` while capture was active.
     The FFN preamble graph is the safe region.
  2. **Grouped MoE slabs exist only at world_size == 4**, so a TP1 synthetic
     gate cannot exercise the graph path. A synthetic `gpu_cuda_graph_replay_matches_eager`
     test that expected graphs on TP1 was wrong and has been removed; TP4 is
     the only configuration that can validate this feature. The full suite
     still passes 32/32.
  **Verdict: reverted.** The first graph replay on the real checkpoint
  completed but produced ` not the` instead of ` Paris.` on every rank,
  so the captured FFN preamble changed model state. The graph code was
  removed completely; full suite passes 32/32 and the four-rank known-answer
  run after the revert produced ` Paris.` identically on all four ranks
  (steady token ~59.35 ms). Do not retry without a graph-aware
  position/state invariant proof.
- **Rejected performance variants.** Shared-expert TP4 sharding reached
  ~50 ms/token but changed the story by changing reduction order and was
  reverted. L2 next-layer prefetch was neutral/slightly worse and removed.
  The fused 24-warp HC block was also rejected; split `hc_mixes` +
  `hc_finalize` remains, with only the unrolled dot loop retained.

- **Grouped TileLang MoE landed (`051c152`).** The correctness fixes were
  specific: grouped-down reads compacted activation rows (not `route / 6`),
  M-batch dispatch selects every row's top-k, M=1 SwiGLU consumes
  `layout_weights`, and dispatch/gate-up take a runtime top-k. The last fix
  is required for the drafter's `dspark_activated != 6`; without it, stale
  target routes cause illegal drafter accesses. Real-checkpoint pipeline
  tests and all four ranks produce ` Paris.` identically.
- **Active-tile bound + sparse register caching landed (`6858d11`).** Grouped
  MoE now runs `ceil(routes/32) + min(routes, local_experts)` tiles instead
  of all 97. This removed the padded SwiGLU/requant work. Sparse attention
  also caches Q and each selected KV row in registers instead of reading
  both twice. Short decode is 74.8 ms/token; 64-token C1 is ~77 ms/token.
  Profile: MXFP8 13.8 ms, sparse attention 10.9 ms, grouped MoE 5.6 ms,
  BF16 GEMV 3.9 ms per token. Full GPU suite: 28/28.
- **Split-KV sparse attention landed (`b979944`).** Long selections use
  128-entry segments and an ordered merge, matching dsv41f's decode
  schedule without changing the FP32 cache layout. C1 is ~76.6 ms/token,
  output-identical to single-pass. Segment reassociation is bounded at
  6.6e-7 in the unit gate (threshold 1e-5); the full suite is 29/29.
- **M-batch prefill is now opt-in.** A 64-token story diverged between
  M-batch-prefill no-spec and sequential-prefill DSpark. Use sequential
  prefill by default so speculation and no-spec share the same target
  state; `WEIGHTSTREAM_MBATCH_PREFILL=1` enables the faster experimental path
  for A/B. This does not affect steady-state decode throughput.
- **Offline M-batch selection fix (GPU validation pending).**
  `attention_step_m` now passes the valid selection length instead of always
  processing `window + index_topk` entries. The tail was already -1-padded,
  so this skips only invalid work and directly targets the slow M=5
  verification path. `cargo check -p weightstream-gpu` passes; no timing or GPU
  correctness claim is made while the ring is offline.
- **TMA tensor-map caching (GPU validation pending).** Grouped MoE layers now
  encode each weight's tensor map once at upload instead of 120 times per
  token. Kernel inputs and launch geometry are unchanged; `cargo check`
  passes, with timing and GPU validation deferred while the ring is offline.
- **Offline fixes validated (2026-11-07).** CUDA-only issues in the
  tensor-map extraction and M-batch selection branch were fixed in
  `3e9aa04`; the full GPU suite passes 29/29 on `spark-4687`. Host-only
  `cargo check` had not covered those CUDA-only paths.
- **Current DSpark identity and speed (2026-11-07).** Sequential prefill
  removed the stream mismatch: DSpark and no-spec both produce
  `who is afraid of mice. Include a mouse in the story.` on all four ranks.
  DSpark averages ~72 ms/token versus ~66 ms without speculation, so it is
  still a regression. Target verification remains sequential; no speedup is
  claimed. Full GPU suite: 29/29.
- **Production restored (2026-11-07).** The promoted GLP-27 TP4 containers
  are running on all four nodes. `/v1/models` and a completion succeed,
  steering is active at α=3.5, and live doctor matches the configuration.
- **Warp-shuffle block reduction: REJECTED.** A shared-tree/warp hybrid
  failed HC and long-decode numerical gates; irregular block sizes made the
  first version diverge, and the power-of-two path was not transparent
  either. Reverted completely. Do not retry without a bit-exact proof on HC
  and long decode, not just a sparse-attention unit test.
- **Thread-0-only softmax scalarization: REJECTED as performance-neutral.**
  Computing the two exponentials once per block added a broadcast barrier
  that offset the saved work. Reverted to per-thread scalar state.
- **Batched DSpark verification: CORRECT BUT NOT VIABLE YET.** One `decode_m`
  over five drafts plus snapshot/rollback and accepted-prefix replay passed
  ` Rome.` on all ranks, but M=5 target verification cost ~0.5–0.8 s and was
  slower than five M=1 decodes. Reverted to sequential verification. Do not
  restore batched speculation until `decode_m` beats sequential M=1 on the
  same short batch. Separate target/drafter M-scratch and runtime top-k fixes
  remain landed.
- **Quantized sparse-MLA + adaptive DSpark (2026-11-07).** Fused M=1
  attention and M-row TileLang verification preserve the four-rank story
  stream. Full-cache speculative snapshots cost 177–359 ms; targeted
  checkpoints of only draft-touched rows cost 6–7 ms. M=5 target compute was
  293 ms, still too expensive when acceptance is low, so batching is gated on
  a previous four-draft acceptance and ordinary calls stay sequential.
  Warmup + 64-token story C1: adaptive DSpark **13.29 tok/s** (75.2 ms/token)
  versus no-spec **17.33 tok/s** (57.7 ms/token). No-spec is the production
  candidate; `cmd_serve` already disables speculation. Raw logs:
  `spark-*:/tmp/tp4-c1-adaptive-*` and `spark-*:/tmp/tp4-c1-nospec-*`.
- **Projection, collective, and launch fixes (2026-11-08).** Short-context
  top-k skips only provably dead score work (compressed indices remain
  `win..win+compress_len`), WO-B uses mapped pinned staging before the RoCE
  reduction, head/compressor GEMVs keep BF16 weights with an exact F32
  fallback for synthetic weights, padded MoE SwiGLU rows are skipped, and
  TileLang function attributes are initialized once. The final four-rank
  story stream is identical and measured **17.93 tok/s** (55.8 ms/token).
  Rejected experiments: FP8 WO-A conversion, direct GPU Engram gathers,
  Engram row cache, HC vectorization, and parallel Sinkhorn normalization.
- **Asynchronous Engram host gathers (2026-11-08).** Both host-mmapped
  tables now start their random FP8 row reads on scoped-lifetime worker
  threads at step start; a join-on-drop guard covers early returns, and each
  layer joins only its own rows. The four-rank story stream is unchanged and
  the full GPU suite passes 30/30. Profiled Engram time fell from 6.6 to
  3.6 ms/token. Two final C1 runs measured 57.35 and 60.47 ms/token
  (17.44 and 16.54 tok/s), versus 61.56 ms in the pre-prefetch profile run;
  variance remains high, so the earlier 55.8 ms / 17.93 tok/s result stays
  the best absolute record. Raw logs: `/tmp/tp4-c1-final-*` and
  `/tmp/tp4-c1-final-repeat-*`.
- **Rejected HC and GEMM experiments (2026-11-08).** The reference-style
  40×7 tiled HC reduction changed the greedy stream. An exact-arithmetic
  variant that overlapped the norm used 256 threads per dot CTA and regressed
  to about 60 ms/token. Fused Q-A/WKV and combined shared gate/up projections
  preserved the stream but were performance-neutral. All three were reverted.
- **Build note:** `cargo clean -p weightstream-cuda` without `--release` removed
  nothing useful; use `cargo clean -p weightstream-cuda --release` when forcing
  a kernel rebuild. Normal builds were otherwise reliable.

TP4 M-batch prefill measured separately on the same checkpoint: a
132-token window-crossing prompt took 9.625 s versus 12.964 s with
`WEIGHTSTREAM_SEQ_PREFILL=1`, a 25.8% reduction. All four ranks produced
`The capital of Japan`. Logs: `spark-*:/tmp/tp4-shortwin-rank{0..3}.log`
(M-batch) and `spark-*:/tmp/tp4-shortseq-rank{0..3}.log` (sequential).

## Measured numbers (same hardware, same model)
| | spark-ds41f TP4 ring (2026-10-17) | weightstream (best measured) |
|---|---|---|
| decode step | 36.7–39.1 ms | — |
| decode t/s, per stream | C1 **98.2**, C8 49.1, C16 35.6 | ~0.5 single-node stream |
| decode t/s, aggregate | C16 **465** | — |
| cold prefill | 4.5–6.2k t/s (3.5k–109k tokens) | ~0.6 t/s (8.3 s / 5 tokens) |
| weights | resident: 119 GB/rank as cudaMalloc @ 226 GB/s | streamed/staged @ ≤163 GB/s |

weightstream data points:

- 2026-10-30, `spark-4687`, single-node `--engine gpu --stream`,
  `WEIGHTSTREAM_STREAM_BUDGET_GB=24`, clocks locked 2418 MHz (diagnostic):
  load 18.3 s, prefill 5 tok in 8.3 s, decode 1.5–2.3 s/tok, output
  ` Paris. The Eiffel` — correct. Real runs at the thermal cap are
  slower.
- 2026-10-26, TP4 `--stream`, 4 ranks: 6 tokens in <45 s (~0.15 t/s);
  ring time 4.1–4.9 s of it (620 collectives, 6.7–7.9 ms avg —
  lockstep-wait dominated, so per-rank compute is the bottleneck).
- Reference numbers: `spark-4687:~/projects/spark-ds41f/experiments/2026-10-17-site-tp4-4k-1017/results.json`.

## Gap decomposition (single-stream, per token)

Multiplicative, ordered by impact:

1. **Weight residency / bandwidth regime.** Original: slices resident
   as ordinary cudaMalloc, 226 GB/s. weightstream streaming: registered
   page-cache mmaps, 163 GB/s ceiling (0.72×), and below the
   registration budget most expert bytes are per-use staged HtoD copies
   — far under that. Traffic ~6 GB/token (4.3 GB fp4 experts + 1.6 GB
   fp8 attention). weightstream's own doc floor for streaming is
   ~40 ms/token warm; measured is 1.5–2.3 s.
2. **MoE dispatch.** Original: grouped-expert GEMM, routing consumed
   on-device. weightstream streaming: host-dispatch loop — gate readback
   + pipeline sync per layer, then top-6 experts × 3 GEMMs serially
   (documented ~150 ms/token at 40 layers before staging; ~240 kernel
   launches+syncs per token). The fused path (`moe_grouped.cu`,
   `w1_all`) exists but only when experts are device-resident (upload
   mode).
3. **No CUDA graphs.** A ~37 ms step at ~240 ops is ~150 µs/op —
   unreachable with per-op launch+sync. Needs step capture/replay.
4. **No batching.** m=1 everywhere in the measured paths; the original
   scales C1→C16 (98→465 t/s aggregate). Serving exists (OpenAI
   endpoint, SSE, sampling) but is one-request lockstep by design. The
   2026-11-13 review closed this as an architectural boundary: independent
   request batching requires per-request engine state and a new scheduler;
   it is not an incremental optimization (see Aggregate batching inspection).
5. **Not a factor: clocks.** Both stacks measured under the same
   thermal cap.
6. **Separate bug class: 96 GB registration budget → RM pathology.**
   Default budget wedges single-node streaming (RM `irq 429` storms,
   minutes-long expert stalls; teardown Xid 31 is an artifact, not the
   cause). Workaround: `WEIGHTSTREAM_STREAM_BUDGET_GB=24`. Reproduced at
   e834f0c and HEAD — not a decode_m regression. Full writeup:
   `docs/streaming.md` (2026-10-30 note). Ceiling unbisected.
7. **Reference-stack configuration drift (measured 2026-11-12).** The live
   vLLM endpoint was NOT the promoted GLP-27 setup: it ran
   `vllm-ds41f-trace:latest` with `--enforce-eager`, no
   `--speculative-config` (DSpark off), a different DSpark cost dir, and
   `VLLM_SINK_REJECTION=0`. Direct streaming measurement gave **13.14 tok/s
   C1** and **182.41 tok/s C16 aggregate**, versus 58.64/215.94 for the
   promoted setup — a 4.46× C1 regression explained by eager execution and
   missing speculation. Restoring the promoted profile is itself blocked:
   the current `config/cluster.json` points at unpublished commit
   `fa8a2ad3` and a missing GLP-39 steering file. Lesson: before any
   cross-stack comparison, run the live-identity doctor and refuse
   mismatched deployments, exactly as `bin/spark bench` already does.

## Current blockers (priority order)

1. **P0 — grouped-MoE for streamed experts.** Removes the
   host-dispatch loop; gate already lives on-device. Also unlocks
   `decode_m` (M-batch prefill) in streaming/TP — it currently declines
   and falls back to sequential per-token decode.
2. **P1 — CUDA graphs for the decode step.** FFN-region graphs are
   implemented behind `WEIGHTSTREAM_CUDA_GRAPHS=1`; validation and timing
   need a drained ring (see the 2026-11-13 experiment-log entry and gap #7).
3. **P1 — batching** (multi-request per lockstep step) and
   prefix-cache reuse on the serving path.
4. **P2 — bisect the registration-budget ceiling** (24 GB clean,
   96 GB wedges) to retune the `WEIGHTSTREAM_STREAM_BUDGET_GB` default.

## Already verified — do not redo

- Synthetic GPU suite 29/29 on `spark-4687` (2026-11-01):
  `cargo test --release -p weightstream-gpu --features cuda -- --test-threads=1`.
- Real-checkpoint streaming decode (above) — known-answer green.
- Ring transport: 10.5 µs p50 4 KB all-reduce, zero loss in 100k soak
  (`docs/transport.md`).
- TP4 upload decode on all 4 ranks, 2026-10-31 (`b5e6f61`): real
  checkpoint ` Paris.`, warm C1 benchmark recorded above.
- GPU kernels each gated vs the CPU reference; the CPU reference is
  gated against the production vLLM TP4 stack.

## Repro commands

(spark-4687, `~/weightstream`; binary `target/release/weightstream` rebuilt
2026-10-31 with cuda; toolchain `~/.cargo/bin/cargo`,
`/usr/local/cuda/bin/nvcc`)

```bash
# build
cargo build --release --features weightstream/cuda

# real-checkpoint streaming validation (the green config)
S=~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
WEIGHTSTREAM_STREAM_BUDGET_GB=24 ./target/release/weightstream decode "$S" \
  "The capital of France is" 6 --engine gpu --stream
# expect: " Paris. The Eiffel"

# TP4 upload validation: per rank
./target/release/weightstream decode "$S" "The capital of France is" 6 \
  --engine gpu --tp4 --rank N --ring nodes-ring4.local.json
```

Env knobs: `WEIGHTSTREAM_STREAM_BUDGET_GB` (registration budget, default
96, 0 = off), `WEIGHTSTREAM_PROF=1` (per-section decode profiler, prints
at exit), `WEIGHTSTREAM_DUMP_GPU=<dir>` (per-layer/op tensor dumps),
`WEIGHTSTREAM_MBATCH_PREFILL=1` (opt into M-batch prefill; sequential is the
correctness default).

## Key code locations

- `crates/weightstream-gpu/src/engine.rs` — `moe_step` (host-dispatch
  loop + `dump_u8` expert staging probes), `moe_step_m` (M-batch),
  grouped path behind `layer.w1_all`, `W8`/`Q4` staging (~line 359),
  `decode_m` (~line 3581), `decode_prefill` fallback (~line 1460).
- `crates/weightstream-gpu/src/stream.rs` — `HostArena` registration /
  prefault / `device_ptr`, `PinnedBuf` staging.
- `docs/streaming.md` — bandwidth table (226/163/37 GB/s), budget
  pathology note. `docs/transport.md` — milestones, upload-vs-stream
  rationale, upload validation and worker-barrier timeout fix.
