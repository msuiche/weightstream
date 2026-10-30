# Unified-memory weight streaming (measured)

GB10 has one physical memory: the GPU reads host pages directly when they
are registered with `cudaHostRegister(cudaHostRegisterMapped)`. Measured on
spark-5bc3 (driver 580.178.04, CUDA 13, a 4-8 GB streaming reduction
kernel, `/tmp/zcopy2.cu`):

| source | bandwidth |
|---|---|
| device memory (cudaMalloc) | 226 GB/s |
| file mmap, page-cache warm | 163 GB/s |
| anonymous malloc, registered | 37 GB/s |

File-backed page-cache pages stream at 72% of device bandwidth; anonymous
host memory does not (uncached C2C reads). The shape of the win is
specific: keep the checkpoint's safetensors shards mmapped from page
cache, register the mappings, and give kernels device pointers into them.
Anonymous-memory staging buffers must be real device allocations.

## What it buys

The 1017 checkpoint is 475 GB quantized; ~272 GB of that is packed-FP4
routed experts. No single Spark holds it as an upload — but decode touches
only top-6 of 384 experts per layer: ~4.3 GB of expert traffic per token
plus ~1.6 GB of fp8 attention weights. At 163 GB/s the streaming floor is
~40 ms/token (~25 tok/s) with page cache warm, before kernel efficiency.

Cold-miss path: shard pages evicted under memory pressure reread from
NVMe (~10 GB/s). The steady-state working set per token is ~6 GB; the OS
cache holds the hot experts once the engine's access pattern settles.

## Design consequences

- `ferroseek-gpu` gains a streaming weight mode: fp4 expert tables and fp8
  projections stay in registered shard mmaps; small hot tensors (norms,
  hc params, compressor state, gate, KV caches, scratch) are device
  allocations.
- TP4 (docs/transport.md) becomes the production-speed path, not a memory
  requirement: single-node decode works first, sharding adds throughput.
- The expert-read pattern (top-6 decided by the gate per layer per token)
  is the prefetch signal: next layer's selected experts are known one
  gate earlier than needed — a one-layer-deep prefetch thread can fault
  those pages in ahead of the GEMM.

## Implementation notes (ferroseek-gpu, measured on the first real run)

- `cuMemHostRegister` needs a current context (register after
  `Cuda::global()`), is serialized per process, and on this driver
  (580.178.04) stops accepting ranges after ~19 x 10 GB —
  shard 19 of 48 fails with "range already mapped". The engine therefore
  registers each shard independently, tracks the registered ranges, and
  any tensor outside them silently falls back to a device upload
  (`ferroseek-gpu/src/stream.rs`). Registration itself faults the range's
  pages (~30 s per 10 GB shard, cold): a full 475 GB registration pass is
  a one-time per-process cost worth eliminating — register only the
  projections + engram tables (~340 GB of the 475 GB), or lazily
  per-shard on first touch.
- Prefetch hook: the MoE gate readback in
  `ferroseek-gpu/src/engine.rs::moe_step` is where each layer's top-6
  expert page addresses become known; a background thread can
  `madvise(MADV_WILLNEED)`/read those 17.7 MB-per-expert ranges one
  layer ahead of the GEMMs (TODO(gpu-prefetch)).
- Registration budget pathology (measured 2026-10-30, spark-4687,
  driver 580.178.04, world=1, the real 475 GB checkpoint): with the
  96 GB default budget the RM falls into unhandled-interrupt storms
  (`irq 429` in dmesg) within the first decode layers — expert GEMMs
  stall for minutes at 96% util with no progress, and killing the
  stalled process surfaces Xid 31 MMU faults as a teardown artifact
  (use-after-unmap, not the primary failure). Same behavior at e834f0c
  and 2c381c7 — not the decode_m changes; the 2026-10-26 TP4 run used
  the same default cleanly, so the threshold is driver/RM-state
  dependent, not a fixed ceiling. `FERROSEEK_STREAM_BUDGET_GB=24` runs
  the full single-node decode clean: load 18 s, prefill 8.3 s,
  1.5-2.3 s/token, ` Paris. The Eiffel`. Treat 96 GB as suspect until
  the ceiling is bisected. GPU clocks were locked to 2418 MHz
  (`nvidia-smi -lgc`) for these runs as a diagnostic only — the boxes
  run capped clocks by design (thermal), so real decode at the default
  clocks is slower than the 1.5-2.3 s/token measured here, and the
  305 MHz idle clock multiplies any stalled-step diagnosis by another
  order of magnitude.
