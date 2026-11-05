# Performance gap: ferroseek vs the spark-ds41f reference stack

Handoff for the agent picking up performance work. Everything here is
measured on the same hardware; sources are cited per claim. Repo state:
`main` @ `b979944` (2026-11-05).

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

| metric | GLP-27 lane | ferroseek upload TP4 (2026-11-01, no-spec) |
|---|---|---|
| decode t/s, single stream (C1) | **58.64** (tpot p50 16.57 ms) | **~13.0** (~77 ms/token) |
| decode t/s, aggregate (C16) | **215.94** (tpot p50 73.86 ms) | — (single-request lockstep) |
| cold prefill | **5.3k t/s** (32K / 131K) | not measured at those sizes |

This is the apples-to-apples bar for ferroseek's upload-TP4 work: same
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
measurement and is about 4.5× faster at the current ~77 ms. The current
64-token run used `Write a long story about a dragon` to avoid early EOS,
so it is a steady-state C1 data point, not a prompt-matched GLP rerun.
ferroseek serving serializes requests. No 32K/131K prefill comparison
was run. Raw logs: `spark-*:/tmp/tp4-upload-rank{0..3}.log` (validation)
and `spark-*:/tmp/tp4-bench-rank{0..3}.log` (benchmark).

DSpark correctness landing (`abf63f2`, `a91917a`, `67402d5`): full
five-draft-plus-anchor acceptance and exact `--no-spec` output identity passed
on all four ranks. The same C1 method above then measured **8.97 tok/s** for
the trained drafter (64 Italy tokens, 7.137 s, mean 111.509 ms/token), versus
**10.24 tok/s** without speculation: **0.876×**, a **12.4% regression**.
Target verification is sequential; no speedup is claimed. Raw logs on the
driver: `/tmp/tp4-c1-dspark-spark-*.log`.

## Experiment log (2026-11-01) — do not duplicate blindly

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
  state; `FERROSEEK_MBATCH_PREFILL=1` enables the faster experimental path
  for A/B. This does not affect steady-state decode throughput.
- **Offline M-batch selection fix (GPU validation pending).**
  `attention_step_m` now passes the valid selection length instead of always
  processing `window + index_topk` entries. The tail was already -1-padded,
  so this skips only invalid work and directly targets the slow M=5
  verification path. `cargo check -p ferroseek-gpu` passes; no timing or GPU
  correctness claim is made while the ring is offline.
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
- **Build note:** `cargo clean -p ferroseek-cuda` without `--release` removed
  nothing useful; use `cargo clean -p ferroseek-cuda --release` when forcing
  a kernel rebuild. Normal builds were otherwise reliable.

TP4 M-batch prefill measured separately on the same checkpoint: a
132-token window-crossing prompt took 9.625 s versus 12.964 s with
`FERROSEEK_SEQ_PREFILL=1`, a 25.8% reduction. All four ranks produced
`The capital of Japan`. Logs: `spark-*:/tmp/tp4-shortwin-rank{0..3}.log`
(M-batch) and `spark-*:/tmp/tp4-shortseq-rank{0..3}.log` (sequential).

## Measured numbers (same hardware, same model)
| | spark-ds41f TP4 ring (2026-10-17) | ferroseek (best measured) |
|---|---|---|
| decode step | 36.7–39.1 ms | — |
| decode t/s, per stream | C1 **98.2**, C8 49.1, C16 35.6 | ~0.5 single-node stream |
| decode t/s, aggregate | C16 **465** | — |
| cold prefill | 4.5–6.2k t/s (3.5k–109k tokens) | ~0.6 t/s (8.3 s / 5 tokens) |
| weights | resident: 119 GB/rank as cudaMalloc @ 226 GB/s | streamed/staged @ ≤163 GB/s |

ferroseek data points:

- 2026-10-30, `spark-4687`, single-node `--engine gpu --stream`,
  `FERROSEEK_STREAM_BUDGET_GB=24`, clocks locked 2418 MHz (diagnostic):
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
   as ordinary cudaMalloc, 226 GB/s. ferroseek streaming: registered
   page-cache mmaps, 163 GB/s ceiling (0.72×), and below the
   registration budget most expert bytes are per-use staged HtoD copies
   — far under that. Traffic ~6 GB/token (4.3 GB fp4 experts + 1.6 GB
   fp8 attention). ferroseek's own doc floor for streaming is
   ~40 ms/token warm; measured is 1.5–2.3 s.
2. **MoE dispatch.** Original: grouped-expert GEMM, routing consumed
   on-device. ferroseek streaming: host-dispatch loop — gate readback
   + pipeline sync per layer, then top-6 experts × 3 GEMMs serially
   (documented ~150 ms/token at 40 layers before staging; ~240 kernel
   launches+syncs per token). The fused path (`moe_grouped.cu`,
   `w1_all`) exists but only when experts are device-resident (upload
   mode).
3. **No CUDA graphs.** A ~37 ms step at ~240 ops is ~150 µs/op —
   unreachable with per-op launch+sync. Needs step capture/replay.
4. **No batching.** m=1 everywhere in the measured paths; the original
   scales C1→C16 (98→465 t/s aggregate). Serving exists (OpenAI
   endpoint, SSE, sampling) but is one-request lockstep by design.
5. **Not a factor: clocks.** Both stacks measured under the same
   thermal cap.
6. **Separate bug class: 96 GB registration budget → RM pathology.**
   Default budget wedges single-node streaming (RM `irq 429` storms,
   minutes-long expert stalls; teardown Xid 31 is an artifact, not the
   cause). Workaround: `FERROSEEK_STREAM_BUDGET_GB=24`. Reproduced at
   e834f0c and HEAD — not a decode_m regression. Full writeup:
   `docs/streaming.md` (2026-10-30 note). Ceiling unbisected.

## Current blockers (priority order)

1. **P0 — grouped-MoE for streamed experts.** Removes the
   host-dispatch loop; gate already lives on-device. Also unlocks
   `decode_m` (M-batch prefill) in streaming/TP — it currently declines
   and falls back to sequential per-token decode.
2. **P1 — CUDA graphs for the decode step.**
3. **P1 — batching** (multi-request per lockstep step) and
   prefix-cache reuse on the serving path.
4. **P2 — bisect the registration-budget ceiling** (24 GB clean,
   96 GB wedges) to retune the `FERROSEEK_STREAM_BUDGET_GB` default.

## Already verified — do not redo

- Synthetic GPU suite 29/29 on `spark-4687` (2026-11-01):
  `cargo test --release -p ferroseek-gpu --features cuda -- --test-threads=1`.
- Real-checkpoint streaming decode (above) — known-answer green.
- Ring transport: 10.5 µs p50 4 KB all-reduce, zero loss in 100k soak
  (`docs/transport.md`).
- TP4 upload decode on all 4 ranks, 2026-10-31 (`b5e6f61`): real
  checkpoint ` Paris.`, warm C1 benchmark recorded above.
- GPU kernels each gated vs the CPU reference; the CPU reference is
  gated against the production vLLM TP4 stack.

## Repro commands

(spark-4687, `~/ferroseek`; binary `target/release/ferroseek` rebuilt
2026-10-31 with cuda; toolchain `~/.cargo/bin/cargo`,
`/usr/local/cuda/bin/nvcc`)

```bash
# build
cargo build --release --features ferroseek/cuda

# real-checkpoint streaming validation (the green config)
S=~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
FERROSEEK_STREAM_BUDGET_GB=24 ./target/release/ferroseek decode "$S" \
  "The capital of France is" 6 --engine gpu --stream
# expect: " Paris. The Eiffel"

# TP4 upload validation: per rank
./target/release/ferroseek decode "$S" "The capital of France is" 6 \
  --engine gpu --tp4 --rank N --ring nodes-ring4.local.json
```

Env knobs: `FERROSEEK_STREAM_BUDGET_GB` (registration budget, default
96, 0 = off), `FERROSEEK_PROF=1` (per-section decode profiler, prints
at exit), `FERROSEEK_DUMP_GPU=<dir>` (per-layer/op tensor dumps),
`FERROSEEK_MBATCH_PREFILL=1` (opt into M-batch prefill; sequential is the
correctness default).

## Key code locations

- `crates/ferroseek-gpu/src/engine.rs` — `moe_step` (host-dispatch
  loop + `dump_u8` expert staging probes), `moe_step_m` (M-batch),
  grouped path behind `layer.w1_all`, `W8`/`Q4` staging (~line 359),
  `decode_m` (~line 3581), `decode_prefill` fallback (~line 1460).
- `crates/ferroseek-gpu/src/stream.rs` — `HostArena` registration /
  prefault / `device_ptr`, `PinnedBuf` staging.
- `docs/streaming.md` — bandwidth table (226/163/37 GB/s), budget
  pathology note. `docs/transport.md` — milestones, upload-vs-stream
  rationale, upload validation and worker-barrier timeout fix.
