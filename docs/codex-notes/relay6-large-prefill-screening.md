# Relay6 large-prefill staging screen

Date: 2026-10-16 UTC  
Scope: read-only source and existing-artifact review. No environment, source,
GPU, model, or service changes were made.

## Decision

Raising `GGML_CUDA_P2P_AR_RELAY_MAX_BYTES` from 4 MiB to 16 MiB is structurally
supported by the current compact-staging allocator for a 10 MiB payload, but
only if the environment is set before communicator initialization and the
communicator is reserved to the final maximum before graph capture. The relay
path uses the full-slot `stage[]` arena sized by `small_cap`; it does **not** use
the compact segmented `stage_compact[]` arena. There is no current 10 MiB
end-to-end dispatcher result, so this is a safety/layout finding, not a
performance result.

A 10 MiB ceiling costs an estimated additional 72 MiB per rank per target
communicator versus the existing 4 MiB small arena. With two independent target
contexts, that is about 144 MiB extra on each V100, leaving roughly 531 MiB if
the stated 675 MiB pre-test free-memory figure applies and the actual reserved
capacity is 10 MiB. If the reserved capacity reaches 16 MiB, the increment is
144 MiB per communicator, or 288 MiB per GPU across two contexts, leaving about
387 MiB. Therefore, check each communicator's actual `cap`/`small_cap` and the
live memory after reservation; do not assume the 16 MiB limit means only a
10 MiB allocation.

The current relay6 kernel is **not** leader-buffer-only. It touches payload
staging on every rank. Making relay scratch exist only on ranks 0 and 3 would
require a new kernel/communication path, not just changing allocator sizes.

## Current production dispatch and capacity

`p2p_ar_relay6_max_bytes()` reads
`GGML_CUDA_P2P_AR_RELAY6` and `GGML_CUDA_P2P_AR_RELAY_MAX_BYTES` once in a
function-local static. The current `run.sh` sets the relay ceiling to 4 MiB.
The dispatcher chooses relay6 only for six ranks with
`204800 <= bytes <= relay_max_bytes`; otherwise, a large payload falls through
to the ring/segmented route. With `GGML_CUDA_P2P_AR_COMPACT_STAGE=1`, that route
uses `stage_compact[]` and `segment_cap`. See
`src/ggml/src/ggml-cuda/allreduce-p2p.cu:373-381,393-428` and
`fixtures/p2p-relay6-20261015/run.sh:20-25`.

For compact staging, reserve computes:

```text
small_limit_bytes = max(ring_min_bytes, relay6_max_bytes)
small_cap         = min(cap, ceil(small_limit_bytes / sizeof(float)))
segment_cap       = ceil(cap / 6)
```

The launch builder initially sets `p.cap = small_cap` and `p.stage[] =
ar->stage[]`. When the relay6 branch wins, it changes the flags to `flags_seg`
but leaves `p.cap` and `p.stage[]` unchanged. Thus for payload size `B <= 16
MiB`, `cap >= B`, and `ring_min_bytes=4 MiB`, `small_cap >= B`; relay6's slot
addresses fit. The subsequent compact-segmented branch is the one that switches
to `segment_cap` and `stage_compact[]`. See
`allreduce-p2p.cu:393-428,566-598`.

## Relay6 payload accesses

For rank `r`, `stage[r]` is the full double-buffered staging allocation resident
on that physical rank. In each block and epoch half, the relay kernel accesses:

| Rank | Payload staging access |
|---|---|
| 0 | Reads `stage[0]` slots 1–5, sums in rank order, writes the result into `stage[1]`, `stage[2]`, and `stage[3]`. |
| 1 | Writes its partial into `stage[0]` slot 1; reads the result from `stage[1]`. |
| 2 | Writes its partial into `stage[0]` slot 2; reads the result from `stage[2]`. |
| 3 | Writes its partial into `stage[0]` slot 3; reads the result from `stage[3]`, then writes it to `stage[4]` and `stage[5]`. |
| 4 | Writes its partial into `stage[0]` slot 4; reads the result from `stage[4]`. |
| 5 | Writes its partial into `stage[0]` slot 5; reads the result from `stage[5]`. |

These are direct accesses in the current kernel at
`allreduce-p2p.cu:148-212`. It uses payload buffers on all six ranks, even
though ranks 0 and 3 are the two island leaders. The flag/epoch synchronization
also involves all ranks. A leader-only buffer design could gather at rank 0,
let ranks 1/2 read that island's result, relay one result to rank 3, then let
ranks 4/5 read rank 3's result. That requires changing the kernel's current
stage writes/reads and flag protocol; allocating buffers only on ranks 0 and 3
under the current kernel would leave live `stage[1,2,4,5]` references invalid.

## Memory arithmetic

For compact mode, `stage[]` allocation per GPU per communicator is
`2 * 6 * small_cap * 4` bytes; the factor 2 is epoch double buffering. The
independent segmented arena is `2 * 6 * segment_cap * 4` bytes. The first two
table rows assume a communicator reserved to exactly 10 MiB
(`cap=2,621,440` floats); the last row is for reserved capacity of at least 16
MiB:

| Configuration | `small_cap` | `stage[]` per GPU/context | Delta vs 4 MiB | Two contexts, per GPU |
|---|---:|---:|---:|---:|
| Current 4 MiB relay ceiling | 1,048,576 floats | 48 MiB | — | 96 MiB baseline |
| Relay ceiling 16 MiB, reserved `cap=10 MiB` | 2,621,440 floats | 120 MiB | +72 MiB | +144 MiB |
| Relay ceiling 16 MiB, reserved `cap >= 16 MiB` | 4,194,304 floats | 192 MiB | +144 MiB | +288 MiB |

With the stated 675 MiB pre-test free space, the two-context 10 MiB-cap case
leaves about 531 MiB per GPU (only 19 MiB above a 512 MiB reserve). If either
communicator reserves at least 16 MiB, the 16 MiB relay ceiling can leave about
387 MiB per GPU, below that reserve. Read each communicator's `max_tensor`,
`small_cap`, and actual free memory before deciding which row applies.

At a 10 MiB `cap`, `segment_cap=436,907` floats and `stage_compact[]` is
20,971,536 bytes (about 20 MiB) per GPU/context; raising only the relay limit
does not change this segment allocation. The two per-rank flag arrays each
contain `6*64*16` 32-bit words (24 KiB); the per-rank epoch array is 256 bytes.
Those allocations do not scale with the payload ceiling. If
`GGML_CUDA_P2P_AR_RING_MIN_BYTES` exceeds 16 MiB, it becomes
`small_limit_bytes` and enlarges `small_cap`/`stage[]` beyond the relay ceiling.

For a hypothetical separate leader relay arena for 10 MiB, an input/gather
buffer at rank 0 needs six slots over two halves (120 MiB/context); rank 3
needs one result slot over two halves (20 MiB/context). Two target contexts
would add 240 MiB to GPU 0 and 40 MiB to GPU 3, with no new relay arena on the
other four cards. This is only an alternative design estimate: current code
cannot use it. With 675 MiB free, GPU 0 would have about 435 MiB remaining;
that concentration is a risk even though aggregate cluster memory is lower.

## Reserve and graph-lifetime constraints

`ggml_cuda_p2p_ar_reserve()` synchronizes all communicator devices before it
frees/reallocates stage storage on a growth. It updates `ar->cap`; the
ALLREDUCE op params include that cap so a grow changes the node and prevents a
captured graph from being reused with stale stage pointers. Set the relay and
ring thresholds before communicator initialization, reserve final capacity
before graph capture, and recapture after any capacity change. Do not change
the limit on an already initialized communicator; both environment parsing and
the derived capacities are cached. See `allreduce-p2p.cu:30-38,373-381,556-609`.

## Existing fixture evidence and its boundary

The historical `p2p-relay6-20261015` fixture directly includes the private CUDA
translation unit and invokes its path 2 (relay6) kernel as an A/B/A sequence;
it checks bit-exact FP32 output and canaries. Its `--graph-window` test includes
15,000,000 floats (60,000,000 bytes, about 57.22 MiB). In
`results/20261015T102848Z.log:82-97`, the direct kernel path is bit-exact at
that size; one-shot reported about 53.28 ms and relay6 about 19.58 ms, including
hot graph replay. However `run.sh` caps `GGML_CUDA_P2P_AR_RELAY_MAX_BYTES` at 4
MiB, and the fixture calls the kernel directly rather than exercising
`p2p_ar_prepare_launch()`'s 10 MiB selection branch. That result validates the
kernel at 60 MiB, not production dispatch, current compact-staging sizing, V32
concurrent contexts, or a V32 throughput gain. No 10 MiB integrated result is
currently established.

## Recommended bounded follow-up

If the root chooses to test the existing kernel without new code:

1. Use a fresh process with six-rank compact staging, relay enabled, a 16 MiB
   relay ceiling, and the current 4 MiB ring threshold. Ensure those variables
   are set before `ggml_cuda_p2p_ar_init()`.
2. Reserve each target communicator to the exact largest V32 partial size
   before capturing any graph. Record the actual `cap`, `small_cap`,
   `segment_cap`, and per-GPU free memory; abort if the required safety margin
   is crossed.
3. Run a bounded exact/canary 10 MiB-shaped dispatcher fixture and require
   `kind=3` on all six ranks, `p2p relay6 selected` logs, rank-ordered FP32
   equality, zero-slice behavior, and cold/hot graph replay. Use the exact
   element count from V32 runtime logs rather than the approximate 10 MiB.
4. If qualifying two target contexts, instantiate both before the test and
   account for their allocations together. Do not grow reserve after graph
   capture.

Do not add a leaders-only allocation under the current kernel. If the per-rank
memory ledger rejects the existing scheme, treat leader-only relay as a
separate kernel project and prove its new peer-buffer ownership and readiness
protocol before measuring performance.
