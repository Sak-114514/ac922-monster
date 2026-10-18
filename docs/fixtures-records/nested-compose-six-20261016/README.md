# V72 physical six-CUDA graph composition qualification

This fixture is a private, host-prepared copy of the V71-qualified nested
Meta2{Meta3,Meta3} and real PLE weight-view probes. It does not edit engine
sources. The V72 implementation is default-off and is selected only by
`GGML_META_NESTED_COMPOSE=1` together with fragment cache, async P2P6, replay
guard, and parent-manifest prerequisites.

## Gates

The nested probe retains the V71 exact-F32 oracle, output canaries, nested TP3
merge, cache hit/miss, async route, initial A/B/A, A4 two-reduction/no-cold
gate, same-UID/new-buffer invalidation, new-UID cold-drain ordering, and
parent-manifest forced-rebuild checks. It then warms the same UID and input
binding and submits 16 more identical A inputs. Each output must still match
the CPU oracle exactly and preserve canaries.

For fixture-only observability, `GGML_META_NESTED_COMPOSE_TRACE=1` requests the
root's detailed `phase=record` / `phase=replay` logs, bypassing the normal
first-eight/every-sixty-four log throttle. The stable window must show at least
12 additional `phase=replay` records, `selected=1`, exactly two AR6 nodes per
physical rank in the compose-ready log, and no additional ordinary child
AllReduce dispatch during the 16 stable replays. The new logs distinguish a
recorded graph from a replay; a record alone does not satisfy this gate.

The PLE probe keeps the real 81,920-byte GGUF F16 weight, four actual
view/cont/reshape/cast taps, exact packed-weight checks, nested-island
A/B/A/reset outputs, parent-manifest selection, and output canaries. Reset
discards the old scheduler and graph context before rebuilding fresh view
tensors; allocated tensors from before `ggml_backend_sched_reset` are not
reused.

## Host build

After the relevant runtime closure is frozen, sync this directory to AC922 and
host-link the probes against that exact closure:

```sh
./build-fixtures.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v73 v73-nested-compose
```

This builds host executables only. It does not initialize CUDA or load a
model. Record the exact runtime tag in the build log.

## Bounded GPU run

Only after root grants a dedicated window:

```sh
NESTED_COMPOSE_GPU_WINDOW_AUTHORIZED=1 \
  ./run-suite.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v73 v73-nested-compose
```

The composition and PLE runners require all six validated UUIDs idle, below
58 C, and at or below 512 MiB preexisting use. Each process is capped at
1 GiB/card and 180 seconds. The PLE probe reads only its one small tensor. It
does not load the full model.

## Large-payload segmented P2P AR control

The regular build creates `segmented-ar-control-$TAG` from the same F32
nested matmul/ADD partial-AR graph, with `NESTED_COMPOSE_N=1,048,832`. That
produces 4,195,328 bytes, 1,024 bytes above the CUDA P2P AR default 4 MiB
segmented-kernel threshold, without `MUL_MAT_ID` or expert routing. Its
single-graph control mode disables composition and runs one exact/canary
output.

```sh
./build-fixtures.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v73 v73-nested-compose-host3
```

The GPU runner requires a separate root authorization, caps execution at 120
seconds, and explicitly sets composition off, the 4 MiB threshold, and P2P AR
debug logging so the log must show `kind=1` (segmented) for all six physical
ranks. Debug mode lowers the stuck-kernel spin limit; this control verifies
dispatch and exact output, not performance.

## Current status

- V74 host build/link succeeded for the composition, PLE, and pure-F32
  segmented-AR control binaries; `ldd` resolved all four GGML libraries to
  the V74 runtime. The V74 segmented-control attempt timed out before its
  first AllReduce; see `STATUS.md`.
- V75 host build/link succeeded for the same three probes; `ldd` resolved all
  four GGML libraries to the V75 runtime. The first composition run exposed a
  stale A4 instrumentation counter; the corrected runner passed the nested
  composition and PLE gates. See `STATUS.md` for both preserved logs and exact
  evidence.
- No composition or PLE GPU probe was run on V73 or V74.
- The initial large-payload attempt reused an EP fixture and aborted in the
  preceding MMID path before AllReduce. A later V74 pure-F32 control attempt
  timed out during fixture execution before any AllReduce launch; see
  `STATUS.md`. Neither attempt establishes an AllReduce result.
- The prior V71 parent-manifest and corrected PLE results are recorded in the
  sibling `meta-parent-manifest-20261016/STATUS.md`.
- Do not run a GPU control without a fresh root-issued window.
