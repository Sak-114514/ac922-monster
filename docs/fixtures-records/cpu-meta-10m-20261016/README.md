# V77 CPU → nested Meta2 10 MiB adapter fixture

Scope: qualify the existing CPU async-upload adapter with an outer
`Meta2{Meta3A,Meta3B}` destination. This fixture does not modify core source or
change the process-wide pinned-memory budget (still 64 MiB). The 10 MiB copy
limit is selected per run with `GGML_META_CPU_ASYNC_COPY_MAX_BYTES=10485760`.

The test creates two independent destination owners, A and B, over the same six
CUDA leaves. It sends 10 MiB CPU COMPUTE tensors through A, then B, then A again.
Each temporary source is overwritten and freed as soon as
`ggml_backend_tensor_copy_async()` returns. Readback checks every result on all
six physical leaves with exact F32 values and surrounding canaries. An 11 MiB
copy must take the existing policy fallback and preserve its result too.

The adapter records per-owner `slot_waits`; `meta-event-query-trace-v77.so`
observes actual nested Meta event creation, recording, and query statuses. The
log validator also checks that A-after-B reuses an A event handle, that both
owners logged their eligible 10 MiB path, and that the 11 MiB request was
rejected by the configured cap. `slot_waits` is reported from the runtime log;
the test does not add a synthetic GPU delay to force a wait.

The `static_assert` in the fixture checks
`cpu_meta_copy_candidate::kDefaultAggregateBytes == 64 MiB` using V77 headers.
10 MiB grows each slot to 16 MiB (the ring grows geometrically), so two slots
for each of A and B reach the existing 64 MiB aggregate limit without raising
it. The 11 MiB cap rejection occurs before another ring slot is allocated.

## Host-only build/link

`build-host-v77.sh` checks the frozen V77 runtime manifest hash
`5d5570afb0bd16cc765232020387b615d7466c3cca20cd23c7b2d1b1c518a9bd`, compiles
the fixture and event observer, and records SHA-256 plus `ldd`. It does not run
the binary or initialize CUDA.

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v77 \
  bash fixtures/cpu-meta-10m-20261016/build-host-v77.sh
```

## Bounded GPU run (only after root grants the window)

The run wrapper requires `CPU_META_10M_GPU_AUTH=ROOT_CPU_META_10M`, verifies all
six expected GPU UUIDs are idle, enforces at least 512 MiB free and below 58°C,
and has a 60-second hard timeout. It does not reset GPUs. It keeps nested GPU
AllReduce disabled because this test is solely CPU-to-Meta copy and events.

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
CPU_META_10M_GPU_AUTH=ROOT_CPU_META_10M \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v77 \
  bash fixtures/cpu-meta-10m-20261016/run-gpu-once.sh
```

Expected successful log counts: A=4 eligible copies, B=3, one A 11 MiB cap
fallback, 48 exact/canary physical-leaf readbacks, and 8 temporary sources
mutated/freed after API return. This is a copy/lifetime qualification, not a
performance claim or model result.
