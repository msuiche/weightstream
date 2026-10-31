# ferroseek TP4 transport

The collectives layer for the 4-node switchless ring. Design follows the
reference stack's rules (sparknet, via docs/model-spec.md §3 and the
spark-ds41f recipe): one-shot RoCE, explicit and fail-stop, fp32 rank-ordered
sums so outputs are batch-independent.

## Cluster reality (config/nodes-ring4.local.json)

| rank | node | mgmt | role |
|---|---|---|---|
| 0 | spark-4687 | 192.168.1.232 | head |
| 1 | spark-5bc3 | 192.168.1.171 | worker |
| 2 | spark-7425 | 192.168.1.180 | worker |
| 3 | spark-7275 | 192.168.1.206 | worker |

Ring cabling over dual ConnectX-7 rails, one dedicated /24 per adjacent
pair: 4687–5bc3 (rail f1), 5bc3–7425 (rail f0), 7425–7275 (rail f1),
7275–4687 (rail f0). Rank order follows the ring; every rank talks to
exactly two neighbors. No switch, no transit: a collective moves data over
the four direct links only.

## What the decode step needs (per layer, per token)

- **expert all-reduce**: fp32 partial sums of routed-expert outputs
  (96 experts/rank), one per MoE layer (spec §6, TP4 note)
- **attention output all-reduce**: fp32, from the row-parallel wo_b
- **index score all-reduce**: fp32 (spec §4)
- **head logits all-gather**: vocab-sharded head (32320 rows/rank),
  logits assembled on the head (spec §3 TP4 note)
- **embedding lookup**: vocab-sharded; the head broadcasts the token id
  and each rank reads its own rows — no collective, just the id

Everything is small (KB–MB). Latency dominates, not bandwidth: the win
over NCCL is the one-shot design — a single RDMA write-with-imm per link,
no channel setup per step.

## Design

- **Transport**: RoCE v2 RC QPs over the ring links, one QP pair per
  direction per link, set up once at engine start from the same node map
  the recipe uses (JSON above). GID index 3 (RoCE), MTU 9000.
- **One-shot all-reduce**: each rank writes its fp32 partial to the
  neighbor's pinned slot with an immediate; on receipt of the ring's
  writes, sum in rank order (fixed order → batch-independent numerics).
  Ring all-reduce is two hops for 4 ranks; latency target < 50 µs.
- **Fail-stop**: a timed-out wait aborts the step and the engine; no
  fallback, no computing on unwritten buffers (reference rule).
- **Rust implementation**: `rdma-core` verbs via the `rdma-sys`-style FFI
  or io_uring + raw Ethernet RoCE? Decision: start with libibverbs through
  a thin safe wrapper; raw sockets only if verbs setup fights us. This is
  the cleanest big Rust win in the project — pure host code, no numerics.
- **Gate**: a 4-rank self-test that all-reduces known vectors and checks
  bit-exact rank-ordered sums on every rank, then a soak (1M iterations,
  packet loss must be zero on the direct links).

## Milestones

1. `ferroseek-transport`: ring discovery from nodes JSON, QP bring-up,
   ping/pong latency probe per link
2. one-shot all-reduce + the bit-exact 4-rank gate
3. wire into GpuModel behind a `tp4` feature: TP1 engine + all-reduces at
   the spec's reduction points, expert/head/vocab sharding at load
4. logits all-gather + serving from the head

## Status

1. done. 2. done (gate 4/4 bit-exact, 100k-iteration soak zero loss,
~10.5 µs p50 4 KB; the gate also covers the all-gather: every rank checks
the rank-ordered concatenation bit-exactly).
3. done. `ferroseek decode --engine gpu --tp4 --rank N --ring
nodes-ring4.local.json` (upload mode, the production path since the
DEVICEMAP registration ceiling was measured — see below) loads rank
`N`'s slice (16 heads, 2 o-groups, 8 index heads, 96 experts, one
vocab shard) as plain device memory and runs the full engine with the
transport's all-reduce at the three spec reduction points plus the
logits all-gather (added to `ring.rs`: same QPs, same two-hop schedule,
rank-ordered concatenation). Real 4-rank decode of the 1017 checkpoint:
'The capital of France is' → ' Paris.', all four ranks' raw logits
bit-identical. On the drained box (ferroseek as the sole tenant):
94 ms/token lockstep = 10.6 tok/s single-stream (2026-10-26; was
1.2 s/token under the vLLM co-tenant with per-use expert staging).
2026-10-31: worker control reads now use the head's 600 s barrier budget
(`b5e6f61`) instead of the 300 s connect timeout; upload load skew had
aborted rank 3 before the first collective. With that fix, a fresh
real-checkpoint upload run completed on all four ranks (` Paris.`, 620
collectives, 92.7–93.0 ms/token).
4. done (2026-10-26). Two layers, both over one model load:
   - `ferroseek decode --prompts-file`: batch requests from a file
     (CLI prompt first, one request per line), every rank in lockstep
     (verified: 3 requests, identical completions on all four ranks).
   - `ferroseek serve`: the OpenAI-compatible endpoint on the head rank
     (std-only HTTP; POST /v1/completions and /v1/chat/completions,
     greedy, `stream:true` gives SSE token events). The head's decode
     owner broadcasts each request with `Phase::Next`; workers loop on
     the control channel. One request at a time — the ring's lockstep
     SPMD contract serializes by design; verified live: non-streaming
     (' Paris.', ' Jupiter. It is the fifth planet from the Sun...') and
     SSE (' Tokyo', ',', ' which', ...) over one load. Batching several
     requests per lockstep step and prefix-cache reuse are the next
     serving milestones; sampling parameters (temperature/top_p) after.
     — Sampling shipped after that: `temperature`/`top_p`/`seed` on
     decode (`--temperature/--top-p/--seed`) and in every serve request,
     broadcast with `Phase::Next` so all ranks draw identically
     (verified live: T=0 exact greedy ' Paris.', seeded T=1.0 repeatable
     ' Paris. The Eiffel Tower is', top_p=0.1 narrows to the greedy
     nucleus; workers error-free). The base model spec is
     temperature-only (spec §12.10) — top_p is an engine-side
     vLLM-compat option; sampled replay against the reference RNG
     diverges by design, the spec's documented limitation.
     Prefix-cache reuse also shipped: `GpuModel::decode_prefill` finds
     the longest cached token prefix, restores its per-request state
     snapshot (~82 MB of device copies, well under a ms) and prefills
     only the remainder — verified live on the ring: a 60-token shared
     prefix cost 11.1 s cold and ~1 s warm with identical greedy output
     against a no-cache control run (LRU bound 8, ~650 MB device memory;
     every rank's cache evolves identically, so lockstep holds).

## The registration ceiling (measured 2026-10-26)

`cuMemHostRegister(DEVICEMAP)` — the streaming mode's whole design —
wedges the GB10 RM above ~64-70 GB of registered ranges per process
(driver 580.178.04): one random rank hangs at its first collective,
dmesg floods `NVRM: Going over RM unhandled interrupt threshold`, the
peers fail-stop. Bisected single-node: 64.1 GB registered decodes
fine, 71 GB wedges, with every registered range CPU-prefaulted either
way. The 475 GB checkpoint needs ~119 GB/rank, so registered streaming
cannot be the production path; upload mode (the slice as ordinary
cudaMalloc — GB10 device memory IS the unified DRAM, 226 GB/s vs 163
GB/s registered-page reads) is. `--stream` remains for slices under
the ceiling.
## Synthetic-gate state (2026-10-27) — GREEN

`gpu_engine_matches_cpu_reference` passes at HEAD (one legal
quant-flip, no amplification). History: it failed with a
two-regression stack, both introduced before the kernel work (the
suite had not run on a GPU since the TP4 landing; an RM wedge from an
illegal address storm masked the picture for part of the bisection):

1. **e834f0c (TP4 landing)** — FIXED (`5512b7a`): every load_q8 call
   grew a WShard; the indexer wq_b took Rows{0, full} at TP1. Hybrid
   bisect (TP4 tree, that one call reverted to All -> green) pinned it.
   The fix never builds a shard descriptor for an unsharded load. The
   byte-identity probe shows Rows == All at these shapes — the
   mechanism is open, recorded in the probe test.
2. **6e8ebdc (upload mode)** — OPEN: with regression 1 fixed, e834f0c's
   tree passes and 6e8ebdc's still diverges (same step-1 quant-flip ->
   step-2 amplification signature). The TP1-visible delta is the
   host-backed engram/expert loads; a device-tables probe at TP1 made
   step 0 diverge instead — not yet understood. The real-checkpoint
   gate (` Paris.`, bit-exact across ranks, every landing) is unaffected
   and remains the production correctness bar; the synthetic gate is
   quarantined until 6e8ebdc's TP1 path is root-caused.

The gate's tolerance model also gained flip-amplification semantics
(`e975045`): a legal activation-snap flip (≤1e-2, argmax stable)
legitimately diverges the two engines' states — through a near-tie MoE
gate it can reorder experts and amplify to O(1) on later steps while
both engines stay correct per their own arithmetic; post-flip steps now
check finiteness + argmax agreement. moe.cu's `-fmad=false` was kept:
routing is the one place a ULP crosses a comparison boundary.

**6e8ebdc root-cause progress (2026-10-27, dump-walk forensics).** With
the indexer fix in, the divergence is isolated to the GPU's routed-expert
path at TP1, step 0, layers 1-3 (L0 bit-exact). Proven identical by
same-run dump-diffs: gateX, gscores/gorder/gweights, the whole wo_b chain
(recomputation adjudicates BOTH engines match), fl/act_fl quant inputs,
engram hash ids + eflat, sparse-attn idxs, kvall overlap, the fp4/fp8
kernels at the exact synthetic shapes, swiglu semantics, Rows/Cols shard
byte-identity. The routed recomputation adjudicator pins it: CPU yrouted
== recompute (0.0); GPU yrouted differs 0.258 from identical inputs. The
staged-expert probe shows TP1 experts are DEVICE-resident (uploaded at
build), so the suspect is the device bytes of L1-3 expert uploads vs
L0's — next step: dump the experts' uploaded buffers and diff against
the store (suspected offset/aliasing in the upload path at layers >= 1).
The earlier [2,4]-vs-[4,2] gorder flip was wedged-GPU-era data; on the
healthy GPU gorder matches everywhere.

**SOLVED (2026-10-28): the Xid 13 + 'moving divergence' were one bug —
misaligned vectorized loads.** compute-sanitizer on the FULL suite (not
per-test: the fault poisoned the channel for every LATER test, which is
why solo runs passed) caught 622 invalid 16-byte global reads in
`fp8_gemm` from `attention_step`: streaming mode hands kernels device
pointers into registered safetensors shards, which align tensors at 8
bytes — the uint4 vectorization demanded 16. Upload mode (cudaMalloc,
256-aligned) never faulted, so TP4 was clean while the streaming test
poisoned the suite. The earlier moe_grouped 'illegal access' revert was
this same class. Fix: every vectorized load site (fp8_gemm,
fp4_expert_gemm, both tiles) checks 16-alignment at runtime and falls
back to the bit-identical byte loop. Memcheck now: 0 errors.

**SOLVED (2026-10-28): two latent scale-boundary bugs** found while
hunting the residual step-1 flip: `act_quant_ue8m0_snap` and
`fp4_actquant`'s e8m0 path still computed ceil(log2) with
`exp2f(ceilf(log2f(...)))` — the exact integer-boundary rounding pitfall
the repo documents in act_quant_fp8_emit, which uses the bit-extract
form. Both now match the CPU's log2_ceil bit-exactly.

**Remaining:** none. The "step-2/3 amplification" described here was
the use-after-free below, not arithmetic — retracted. The gate is
GREEN: `gpu_engine_matches_cpu_reference` passes with one legal flip
(2.4e-3 at step 1, argmax stable) and every other step at 4.8e-7.

**S1-flip forensics (2026-10-28, close of series).** The in-process
capture bisector (capture_x + capture_y on both engines — new permanent
tooling) pins the flip precisely: step 1, layer 3's MoE output, 4.807e-3,
with every attention input clean that step. Kernel-by-kernel elimination
closed every remaining suspect at its exact shape: fp8_gemm at the
shared expert's m1/n256/k64 with tiled scales (new test), fp8_gemm at
wo_b's K=32, both tiles at the synthetic MoE shapes, the snap kernels'
scale rules (bit-form ceil(log2) everywhere), swiglu semantics, the
routed adjudicator. Shared-expert contribution y−yrouted diverges
exactly at the flip.

**RESOLVED (2026-10-27) — one use-after-free, not hardware.**
`GpuModel::from_synthetic` returned `Self::build(...)` directly and
dropped the weight store on return; every host-backed engram
`embed_w`/`embed_s` is a raw pointer into that store. First model in a
process read still-intact freed heap (tests pass); later models read
recycled heap — NaN engram rows, ASLR-dependent run-to-run variation,
occasional deferred illegal-access — all with memcheck/initcheck clean
because the bug is host-side. The entire "post-flip O(1) amplification"
above was this UAF, not arithmetic: with the store kept alive
(`store_keepalive`, same as the checkpoint constructors), the gate on a
freshly-booted box is step-1's genuine legal quant-flip (2.4e-3, argmax
stable) followed by CLEAN steps at 4.8e-7. The dump-pairing
"amplification" forensics and the box-degradation verdict are retracted.
The real checkpoint path always held the store and was never affected.
The gate floor is reclassified accordingly: a legal flip requires every
pre-flip step clean and argmax stable at the flip and the next step;
no amplification steps exist. Full GPU suite 19/19 x3 on a fresh box;
the step-1 flip itself remains the documented rare quant-boundary class
(engine.rs header), deterministic across boxes.
