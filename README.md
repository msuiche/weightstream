# ferroseek

A DGX Spark-native inference engine for DeepSeek V4.1 Flash, written in
Rust. One model, one hardware target, no generality tax: the scheduler, KV
cache, cluster transport and serving layer are Rust; the tensor-core kernels
are CUDA, reused from proven stacks first and rewritten only where profiling
says so.

Status: **phase 0 — scaffolding.** `ferroseek info <model-dir>` parses a
checkpoint's `config.json` and prints shape and KV sizing. Nothing touches a
GPU yet. See [`docs/architecture.md`](docs/architecture.md) for the kernel
strategy and the phase plan.

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
| `crates/ferroseek` | CLI front door (`info`, `serve`, `bench`) |
| `crates/ferroseek-core` | config parsing, KV/memory sizing — GPU-free, testable anywhere |
| `docs/architecture.md` | engine design, kernel inventory, phases, verification discipline |

## Build

```sh
cargo build
cargo test
cargo run -- info /path/to/DeepSeek-V4.1-Flash-1017
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
