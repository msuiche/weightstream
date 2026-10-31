# Performance gap: ferroseek vs the spark-ds41f reference stack

Handoff for the agent picking up performance work. Everything here is
measured on the same hardware; sources are cited per claim. Repo state:
`main` @ `b5e6f61` (2026-10-31).

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

| metric | GLP-27 lane | ferroseek upload TP4 (2026-10-31) |
|---|---|---|
| decode t/s, single stream (C1) | **58.64** (tpot p50 16.57 ms) | **10.24** (tpot p50 97.20 ms) |
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
The C1 measurement used one model load, a 64-token warm-up request, then
64 greedy tokens for `The capital of Italy is`; mean 97.635 ms/token,
p50 97.196 ms, range 84.524–112.863 ms. GLP-27 remains **5.73×** the
ferroseek C1 throughput. C16 is not an apples-to-apples measurement:
ferroseek serving serializes requests. No 32K/131K prefill comparison
was run. Raw logs: `spark-*:/tmp/tp4-upload-rank{0..3}.log` (validation)
and `spark-*:/tmp/tp4-bench-rank{0..3}.log` (benchmark).

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

- Synthetic GPU suite 25/25 on `spark-4687` (2026-10-30):
  `cargo test --release -p ferroseek-gpu --features cuda`.
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
`FERROSEEK_SEQ_PREFILL=1` (force sequential prefill).

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
