# ferroseek

A DGX Spark-native inference engine for DeepSeek V4.1 Flash, written in
Rust. One model, one hardware target, no generality tax: the scheduler, KV
cache, cluster transport and serving layer are Rust; the tensor-core kernels
are CUDA, reused from proven stacks first and rewritten only where profiling
says so.

Status: **TP4 upload mode, ~15 tok/s without speculation.** Real 4-rank
decode of the 1017 checkpoint on the drained ring (2026-11-07): a greedy
story decode runs at 65–67 ms/token in lockstep. Grouped TileLang MoE,
active routing tiles, fused quantized sparse-MLA attention, and cached TMA
maps are landed; the GPU gates and four-rank output-identity checks pass.
What works today:

- **CPU reference engine** (`ferroseek-ref`) — full DS4.1F decode on the real
  475 GB checkpoint, **gate-passed against the production vLLM TP4 stack**
  (teacher-forced top-1 identity, transport-bounded logprob deltas)
- **16 GPU kernels** (`ferroseek-cuda`) — every DS4.1F primitive, each gated
  on GB10 (sparse attention, DSA indexer, CSA compressor, hc Sinkhorn, MoE
  routing, FP8/FP4 block-scaled GEMMs, engram, quant snaps)
- **GPU decode engine** (`ferroseek-gpu`) — real-checkpoint decode via
  unified-memory weight streaming from page-cache mmaps; tokens identical to
  the CPU engine (` Paris.`)
- **Fused sparse-MLA attention** — TileLang split/merge kernels consume FP8
  sliding-window records and FP4 compressed records directly; valid window
  entries are packed into the kernel's prefix contract. The FP32 attention
  path remains available as `FERROSEEK_SPARSE_FP32=1` for A/B checks.
- **TP4 transport** (`ferroseek-transport`) — one-shot two-hop fp32
  all-reduce over the 4-node switchless RoCE ring, bit-exact on all ranks,
  ~10 µs/4 KB, zero packet loss in a 100k soak
- **TP4 sharded decode** — `ferroseek decode --engine gpu --stream --tp4
  --rank N --ring nodes-ring4.local.json`: per-rank slices (16 heads,
  96 experts, vocab shard) with all-reduces at the spec's three reduction
  points plus the logits all-gather
- **TP4 M-batch prefill (opt-in)** — `FERROSEEK_MBATCH_PREFILL=1` enables
  TP-local `decode_m` batches; sequential prefill is the default until its
  real-checkpoint greedy-stream divergence is root-caused. The measured
  132-token A/B was 9.625 s vs 12.964 s sequential (25.8% faster).
- **DSpark speculative decode** — three MTP blocks, target-hidden capture,
  non-causal draft attention, Markov-biased proposals and greedy target
  verification; full-draft acceptance and `--no-spec` output identity verified
  on TP4. C1 is 8.97 tok/s versus 10.24 without speculation (0.876×);
  verification is currently sequential, so no speedup is claimed.
- **Serving** — `ferroseek serve --tp4 --ring ...`: OpenAI-compatible
  endpoint on the head rank, SSE streaming, `temperature`/`top_p`/`seed`
  broadcast with `Phase::Next` so all ranks draw identically

Remaining: multi-request batching on the serving path. See
[`docs/architecture.md`](docs/architecture.md),
[`docs/model-spec.md`](docs/model-spec.md), [`docs/streaming.md`](docs/streaming.md),
[`docs/transport.md`](docs/transport.md).

## Why

vLLM carries every model and every GPU. A single-arch, single-platform
engine can bake in what the tuned DS4.1F stacks discovered the hard way:
64 KiB pages, FP8 MLA KV, one-shot RoCE collectives over a switchless ring,
unified-memory weight streaming, L2 weight prefetch during decode, and a
scheduler with no Python in the step loop. Rust earns its place in the host
code — the step scheduler, the cluster transport, the serving path — not in
rewriting mature matmul kernels.

## Layout

| path | what it is |
|---|---|
| `crates/ferroseek` | CLI front door (`info`, `weights`, `tokens`, `decode`, `serve`) |
| `crates/ferroseek-core` | config, safetensors/mmap loader, quant primitives, KV sizing |
| `crates/ferroseek-ref` | CPU reference decode engine (the numerics gate) |
| `crates/ferroseek-cuda` | CUDA runtime + the 16 gated kernels (sm_121) |
| `crates/ferroseek-gpu` | GPU decode engine + unified-memory streaming |
| `crates/ferroseek-transport` | TP4 ring collectives (one-shot RoCE all-reduce) |
| `crates/ferroseek-tokenizer` | HF tokenizer + engram compressed token map |
| `docs/` | model-spec, architecture, kernels, streaming, transport |
| `scripts/gate.py` | teacher-forced logit gate vs the serving stack |

## Build

```sh
cargo build
cargo test                                  # host-only suite
# on a DGX Spark:
cargo test -p ferroseek-cuda --features cuda
cargo test -p ferroseek-gpu --features cuda
./target/release/ferroseek decode /path/to/DeepSeek-V4.1-Flash-1017 \
    "The capital of France is" 6                 # CPU reference engine
./target/release/ferroseek decode /path/to/DeepSeek-V4.1-Flash-1017 \
    "The capital of France is" 6 --engine gpu --stream
```

## Credits

Inspired by, and measured against,
[christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f)
(Apache-2.0) — the reproducible TP4-ring/TP3-triangle DS4.1F deployment whose
release discipline (frozen recipes, one-session benchmark comparisons,
temperature-0 output identity) this project adopts as its own contract.
CUDA Rust kernel authoring follows NVIDIA's
[CUDA Rust](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/)
tracks. License: Apache-2.0.
