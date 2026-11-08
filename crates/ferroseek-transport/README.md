# ferroseek-transport

One-shot RoCE collectives for the 4-node DGX Spark ring — ferroseek's TP4
transport layer (docs/transport.md). Pure host code: pinned host buffers
registered with `ibv_reg_mr`, no GPU memory, no NCCL.

## What it does

- Parses the recipe's `nodes-ring4.local.json` (schema v1) and derives the
  ring: each rank's successor and predecessor, with both direct ConnectX-7
  rails per adjacent pair resolved to (my ibverbs device, peer ibverbs
  device, shared /24).
- Rendezvous and phase barriers over the management network (plain TCP,
  newline-delimited JSON — data never flows there).
- Brings up one RC QP per (neighbor, rail) — 4 QPs per rank — at engine
  start, exchanged once via the control channel. No per-step channel setup.
- **One-shot two-hop fp32 all-reduce** (the decode-path design):
  hop 1 writes my partial to both neighbors (rail 0 toward the successor,
  rail 1 toward the predecessor) with an immediate tagging the source rank;
  hop 2 forwards the predecessor's partial to my successor. After two hops
  every rank holds all 4 partials and sums them locally in rank order
  (0+1+2+3, left to right) — fixed order, bit-identical on every rank,
  batch-independent numerics. 3 sends / 3 receives / 2 hops per rank.
- **Fail-stop**: every RDMA wait has a deadline (default 100 ms). A timeout
  or any CQE error aborts the operation and the run — the engine never
  computes on partially written buffers. Run-ahead safety: immediates carry
  (step seq:16 | src rank:8) and slots are double-banked per step parity,
  so a neighbor running one step ahead cannot corrupt the current step.
- RDMA ping/pong probe per link per rail (8-byte write-with-imm round trips).

## FFI choice

libibverbs is bound through hand-written `extern "C"` declarations
transcribed field-for-field from the rdma-core 50 headers on the Spark
nodes (`/usr/include/infiniband/verbs.h`, aarch64/glibc). No bindgen, no
`rdma-sys`: the verbs surface needed here is ~20 functions and ~15 structs,
the nodes' headers are the ground truth, and a hand transcription keeps the
build dependency-free (the crate is rsynced alone to the nodes and built
there with the system toolchain).

Consequences worth knowing:

- rdma-core 50 makes `ibv_poll_cq` / `ibv_post_send` / `ibv_post_recv`
  inline wrappers over the `ibv_context.ops` function-pointer table; the
  safe layer calls the ops slots directly (`src/verbs/mod.rs`).
- The exported `ibv_query_port` takes the (smaller) `_compat_ibv_port_attr`;
  `ibv_port_attr` here includes the trailing `active_speed_ex`.
- `IBV_QPT_RC` is **2** (SMI=0, GSI=1 — IB ordering), and this mlx5/kernel
  rejects QP INIT transitions without `IBV_QP_PKEY_INDEX` in the mask.
- The node map's interface entries ("rocep1s0f0"...) are ibverbs *device*
  names; the netdevs are named differently (e.g. `enp1s0f0np0`). Port
  discovery therefore matches a RoCE v2 GID whose embedded IPv4 falls in
  the rail's /24 — no netdev names involved. The QP path MTU follows the
  port's active MTU.

## The test runner

```
ferroseek-transport bench --ring nodes-ring4.local.json [--port 29871] \
    [--ping-iters 2000] [--soak-iters 100000] [--gate-rounds 8] \
    [--verify-every 1000] [--wait-ms 100] [--elems 1024]
```

Any node can launch it; rank 0 (head) listens on the management net and
drives the phases: linkup → RTR → RTS → probe (ping/pong per link per
rail) → gate (bit-exact all-reduce of vectors carrying f32::MAX/-MAX,
smallest subnormals, denormals) → soak (100k × 4 KB all-reduce with
per-iteration latency, bit checks every N, and sysfs loss accounting:
netdev statistics + RoCE port hw_counters, netdevs resolved via
`/sys/class/infiniband/<dev>/device/net`). Head prints the report and
exits non-zero on any failure.

Off-cluster: `cargo test` skips everything ring-bound. On a single node,
`FERROSEEK_RING=self cargo test --test ring` runs a QP loopback smoke test
(two RC QPs connected to each other, one RDMA write-with-imm) that
validates the FFI end to end.

## Wiring into GpuModel (tp4)

The all-reduce call sites from the spec §3 TP4 note, i.e. where
`Ring::allreduce` replaces the TP1 in-place sums:

- `ferroseek-gpu/src/engine.rs::moe_step` — routed-expert output reduction
  (96 experts/rank shard): the fp32 accumulator the experts add into
  (model.py:895 equivalent) is per-rank; all-reduce after the expert GEMMs,
  before the residual add.
- row-parallel `wo_b` (attention output, fp32, no bias) — after the fp8
  GEMM, all-reduce the 5120-wide partial.
- `index_score` all-reduce (model.py:559-560) after `weights_proj`.

Everything else is sharding at load (heads 16/rank, experts 96/rank, vocab
32320 rows/rank) and a logits all-gather on the head — the all-gather can
reuse the same QPs (write-with-imm into rank-indexed slots; head sums
nothing, just concatenates ordered slices).
