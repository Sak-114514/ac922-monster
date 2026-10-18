# V92 ordered Relay6 allreduce fixture

This fixture extends the copied Relay6 actual-dispatch harness to test V92
ordered Relay6 kind 4 beside segmented kind 2 and Relay6 kind 3. It stays
fixture-only: the .cu file includes the frozen V92
src/ggml/src/ggml-cuda/allreduce-p2p.cu, and the build links V92 libraries.
It does not rebuild the engine or load a model.

## Matrix

Each process selects one mode, payload, graph-entry path, and zero rank.
Separate processes are required because the dispatcher reads environment
settings once per process.

The bounded default run has 16 cases: ordered kind 0 at 160 KiB through both
entry paths with zero rank -1; ordered kind 4 at 200 KiB and 10 MiB through
both entry paths with zero ranks -1, 0, and 3; then 10 MiB segmented kind 2
and prior Relay6 kind 3 dispatch controls with zero rank 3. This confirms
fallback kind 0 at 160 KiB and kind 4 at both larger sizes, and compares the
old relay and ordered paths in one run window.

The ordered cases use both dispatch through CUDA stream capture and the
explicit ggml_cuda_p2p_ar_graph_append() API. They cover zero ranks -1, 0,
and 3.
Each process creates two communicators, runs three A/B/A replays per context,
then runs an all-negative-zero frame. The rank-order FP32 oracle uses
non-associative cancellation values and exact output comparison. The
all-negative-zero oracle expects -0 for zero rank -1; skipping rank 0 or 3
introduces +0 and the expected output is +0. Input-change records, canaries,
per-GPU added-memory limits, exact dispatch kind, relay cross-payload marker,
and all six rank markers are required.

## V92 host build

The V92 build ID is 20261016T112235.532285Z, runtime
runtime-perf-v92, and included allreduce-p2p.cu SHA-256
b0790248c39350d862e09d0841575a8c7bad1e3c5de19fe50e55c8c9ad311b19.
The host build uses -fno-fast-math and -ffp-contract=off to keep the
rank-order F32 oracle meaningful. USE_CUDA_GRAPH is provided by V92's
common.cuh when graph support is enabled; the fixture calls the explicit
graph-append API.

On AC922, after V92 is frozen, build the fixture binary only:

    ./fixtures/relay6-ordered-20261016/build-v92.sh \
      /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v92 \
      v92-host3

The build verifies the V92 report, source and DSO hashes, direct inclusion,
symbol availability, and resolved V92 DSO closure. It compiles the fixture
only; it does not launch it.

## Bounded GPU matrix

GPU execution requires an explicit root grant. The runner checks all six
V100s for an idle compute table, temperatures below 58 C, at least 512 MiB
free memory, and no more than 1 GiB added-used memory per GPU. Defaults cap
each case at 15 seconds and the complete matrix at 120 seconds. Root can
select a subset by setting RELAY6_ORDERED_CASE_MATRIX to newline-separated
`mode|payload_bytes|entry|zero_rank` rows; the runner permits at most 16
unique rows and restricts controls to the authorized 10 MiB dispatch cases.

    RELAY6_ORDERED_GPU_WINDOW_AUTHORIZED=1 \
      ./fixtures/relay6-ordered-20261016/run-bounded.sh v92-ordered-$(date -u +%Y%m%dT%H%M%SZ)

The runner preserves per-case stdout, stderr, combined logs, validation
results, and V92 provenance. It stops if any GPU crosses a temperature,
free-memory, added-memory, exit-status, or timeout bound. It does not alter
fan or BMC state.

The completed 16-case run is recorded at
`results/v92-ordered-v92-host3-eventwait-20261016T114500Z/`. Its summary JSON
and per-case validation records report 16/16 passes. Ordered kind 0 was
selected at 160 KiB through both APIs; ordered kind 4 was selected at 200 KiB
and 10 MiB through both APIs on all six ranks. Rank-ordered FP32 outputs,
A/B/A replay, changed inputs, all-negative-zero signs, and canaries passed.
At 10 MiB dispatch with zero rank 3, mean max-GPU-event time was 1.326 ms for
ordered kind 4 and 1.949 ms for Relay6 kind 3 in the same bounded window.
These are fixture timings only, not model throughput or quality evidence.
The earlier host2 event-readiness failure is preserved and described in
`RESULTS.md`.

The copied V81/V86 logs and reports in this directory remain historical
artifacts. They are not V92 evidence.
