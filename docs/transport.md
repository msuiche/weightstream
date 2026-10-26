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
