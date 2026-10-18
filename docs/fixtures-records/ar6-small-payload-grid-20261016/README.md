# Six-rank P2P AllReduce grid screen at 160 KiB and 10 MiB

This host-prepared microgate reuses the V77-tested six-rank CUDA composition
path. Each grid option runs in its own process and creates an immutable P2P AR
communicator. The process tests both payloads without changing that grid:

- Decode-sized payload: `2560 * 16 * sizeof(float) = 163,840 bytes` (160 KiB).
- Prefill-sized payload: `10,485,760 bytes` (10 MiB).

Each per-rank graph contains one F32 identity-scale node. Composition has 49
child graphs and 48 rank-six AllReduce boundaries per rank. The one-node child
keeps extra arithmetic from hiding the communication cost. The fixture checks
the actual routes: with the V77 4 MiB segmented threshold and relay6 disabled,
160 KiB maps to the one-shot path, while 10 MiB maps to segmented, by the
frozen source's size thresholds and the runner's forced 4 MiB threshold with
relay6/compact staging disabled. V77 has no runtime algorithm marker, so the
report labels the route as source/environment-inferred rather than claiming an
observed runtime dispatch.

Correctness gates use rank-specific, non-associative F32 partials so the
left-to-right rank 0..5 result is observable. Each payload runs an ordinary AR
control, then composed A/B/A replays, then a warmed timing window. Every output
element on all six ranks must bitwise match the rank-order oracle; host D2H
buffers also have front/back canaries. A failed gate exits before its timing
summary can be accepted.

The wall timer covers `compose_replay()` plus synchronization across all six
CUDA backends. Per-rank CUDA event spans bracket the same whole composed replay
on each backend stream; the report takes the maximum rank duration. Both values
include the 49 child graphs and 48 AR6 boundaries. Dividing by 48 is an
amortized per-boundary figure, not an isolated AllReduce-kernel latency.

## Host link

The build uses frozen `runtime-perf-v77` libraries and performs no CUDA
initialization:

```sh
./build-host.sh
```

## Bounded GPU run

Run one grid per process, only after root grants a bounded six-GPU window. The
runner pins the six known UUIDs, verifies the V77 runtime manifest, checks that
every GPU is idle and below 58 C, caps runtime at 120 seconds, and records a
separate log directory per process.

```sh
AR6_GRID_GPU_AUTH=V77_AR6_GRID_SCREEN ./run-one-grid.sh 8
AR6_GRID_GPU_AUTH=V77_AR6_GRID_SCREEN ./run-one-grid.sh 16
AR6_GRID_GPU_AUTH=V77_AR6_GRID_SCREEN ./run-one-grid.sh 32
AR6_GRID_GPU_AUTH=V77_AR6_GRID_SCREEN ./run-one-grid.sh 64
```

Do not run multiple grid processes concurrently: each uses all six cards and
its own communicator. This fixture does not load a model.
