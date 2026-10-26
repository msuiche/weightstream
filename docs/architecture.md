# ferroseek architecture

Target: DeepSeek-V4.1-Flash-1017, NVFP4/FP8/BF16 native weights, on a
switchless DGX Spark fabric — TP4 ring (four nodes, 1M context, 16
sequences) first, TP3 triangle second. Everything else is out of scope
until phase 5.

## What the reference stack proves

spark-ds41f is the bar. Its production shape, distilled:

- **Kernel family**: TileLang for SM121 — sparse MLA, the MXFP4 indexer,
  block-32 FP8 and BF16 projections, FP8×FP4 routed experts, vocabulary
  heads, RoPE, the CSA compressor, KV writers, Engram gate/hash/row gather,
  the WO projection, the drafter's context-KV projection. B12X is the
  fallback backend in the same image.
- **Collectives**: sparknet one-shot RoCE all-reduce / all-gather, explicit
  and fail-stop — never the PyTorch NCCL process group in the serving path.
  Sequence-parallel prefill past ~205 tokens; Engram's sequence-parallel
  product exchanged all-to-all.
- **Memory**: 64 KiB base pages, FP8 MLA KV (`fp8_ds_mla`), routed-expert
  weight scales as a row base plus 4-bit codes (~2 GiB/rank reclaimed), the
  embedding and output head parked in the display carve-out, L2 weight
  prefetch during decode, fail-closed memory guards (5 GiB startup, 3 GiB
  steady).
- **Decode shape**: DSpark speculative decoding, 5 draft tokens, block
  rejection, a draft budget that is a rule of the inputs — nothing timed at
  boot, so the same build repeats within ~0.4%.
- **Discipline**: frozen recipes and baselines, one-session benchmark
  comparisons with 95% intervals, 256-prompt temperature-0 output identity
  as the correctness gate.

An engine that cannot match the correctness gate does not get benchmarked;
an engine that cannot match the repeatability does not get promoted. Those
two rules are ferroseek's too.

## Where Rust wins, where it doesn't

The step scheduler, admission control, KV block management, the RoCE
transport, checkpoint loading and the serving layer are host code. There,
Python costs real latency per step and Rust costs none. The matmuls do not
care what language launches them — rewriting proven TileLang/B12X kernels
for its own sake buys nothing and risks the numerics. So:

- **Phase-gate every kernel.** Reuse first (FFI into compiled artifacts),
  rewrite only behind a profiler, keep bit-exactness gates per kernel.
- **CUDA Rust is the authoring track for new kernels**, not a migration
  track: `cuda_std`/NVPTX for genuinely new kernels (KV writers, gather,
  dequant+GEMM fusions), with `cust` on the host side for driver-API
  control without a C++ shim. nvcc-built `.cu` stays allowed where a
  reference kernel must be ported verbatim — PTX is PTX.

## The arch inventory (what must exist before "hello token")

DS4.1F is not Llama-shaped. The load-bearing pieces:

1. **MLA** — 576-wide compressed latent per token per layer
   (kv_lora_rank 512 + qk_rope 64); FP8 KV with per-block scales.
2. **DSA indexer** — MXFP4 index heads scoring cached tokens, top-k page
   selection into sparse MLA; decode and prefill take different tile
   shapes.
3. **MoE** — 256 routed experts, FP8×FP4 GEMMs over the dispatch layout's
   live tiles, scales as row-base + 4-bit codes; one shared expert in FP32.
4. **mHC** — multi-hyper-connection stream (4× widening, Sinkhorn-rescaled)
   with its FP32 projection.
5. **Engram** — n-gram hash tables; rows read by a native worker in layer
   order, gathered per-rank, exchanged all-to-all under sequence parallel.
6. **DSpark drafter** — trained draft head, block rejection, draft budget
   from input rules.
7. **Transport** — one-shot RoCE collectives on the ring, rank-ordered FP32
   partial summation so outputs are batch-independent.

Items 1–3 decode a correct (slow) token. Items 4–6 make it a DS4.1F token.
Item 7 makes it a cluster.

## Phases

- **P0 — scaffolding (here).** Workspace, config parsing, KV sizing,
  correctness-gate tooling design.
- **P1 — host runtime, wrong-but-shippable.** Tokenizer (HF `tokenizers`
  crate), weight loader for the checkpoint's safetensors layout, KV block
  pool, step loop, single-node decode of a *supported small model*
  (Qwen3.8-27B NVFP4) through cust + linked kernels. Proves the Rust→CUDA
  pipeline on GB10 end to end before DS4.1F numerics enter.
- **P2 — DS4.1F single-node decode.** MLA + dense-MLA fallback (indexer
  off), MoE on linked B12X kernels, correctness gate: temperature-0
  identity vs the spark-ds41f serving stack on the 256-prompt set,
  teacher-forced KL where outputs diverge.
- **P3 — the DS4.1F specials.** DSA indexer, mHC, Engram, DSpark draft.
  Each lands behind its own gate: indexer on/off must change speed, never
  outputs.
- **P4 — TP4 ring.** sparknet-compatible one-shot collectives (FFI first;
  a Rust RoCE verbs transport is the cleanest big rewrite candidate in the
  project — no tensor numerics, pure host code), rank-ordered reductions,
  sequence-parallel prefill.
- **P5 — serving.** OpenAI-compatible endpoint, continuous batching with
  late admission, the bench harness (same protocol as the reference: gate
  kernel changes on step time, use acceptance to explain tok/s).

## Verification contract

- Correctness: 256 greedy prompts, byte-identical to the reference stack,
  before any performance claim. Numerics changes ship with teacher-forced
  KL, not vibes.
- Performance: one-session A/B/A comparisons, 95% intervals, step time as
  the kernel metric. Claims against the published spark-ds41f baselines,
  same node count, same transport.
- Stability: fail-closed memory guards from P2 on. A rank that cannot
  prove its peers are alive stops rather than computes on unwritten
  buffers.

## Open questions

- Checkpoint loader: the reference stack parks embedding + output head in
  the display carve-out — reproduce via the same carve-out ioctl path, or
  accept ~2 GiB of pool pressure?
- The output head's 12-bit packed BF16 form: exact format is a spark-ds41f
  implementation detail; port or substitute the BF16 head at a small
  memory cost (measure first).
- cuda_std maturity on sm_121a: PTX target works, but block-scaled MMA
  intrinsics (MXFP8/FP4) may need inline-asm fallback until the Rust CUDA
  stdlib exposes them.
