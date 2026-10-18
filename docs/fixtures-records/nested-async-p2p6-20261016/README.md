# Nested asynchronous P2P6 hot-replay fixture

This V68-only microfixture uses the real tiny F32 `Meta2{Meta3A,Meta3B}` graph
and preserves the V67 exact oracle/canary gates. It runs three same-parent
frames A/B/A, a fourth same-parent A4 frame that must be hot-ready, then a new
parent UID using a different input buffer. Input buffers are owned by this
fixture context and are marked `COMPUTE` only after allocation. Before changing
usage, the fixture queries/caches both Meta2 and Meta3 split descriptors; it
snapshots physical leaf shapes, K-axis slices, buffers, and data aliases before
and after that usage change; all must remain identical.

The A4 submission is the hot-replay gate: it must log exactly two async P2P6
outer reductions and zero cold-drain count delta over the entire graph call.
The subsequent new-UID graph is queued without host readback; it must cold-drain
the pending six-rank work before parent-generation invalidation. Both queued
outputs are then read once and checked bit-exactly against the CPU oracle.

Build only the fixture host executable against the frozen V68 closure:

```sh
./build-fixture.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v68 v68-async-hot-host-r2
```

After a fresh root-granted GPU window, run the single ON arm:

```sh
META_FRAGMENT_CACHE_GPU_WINDOW_AUTHORIZED=1 ./run-probe.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v68 v68-async-hot-host-r2
```

The runner requires six free GPUs, each below 58 C and 512 MiB preexisting
usage; the fixture is limited to 1 GiB/card and 180 seconds. It enables
`GGML_META_FRAGMENT_CACHE=1`, `GGML_META_NESTED_ASYNC=1`, the view/replay/FIFO
guards, and CUDA graph path logging. It does not repeat the OFF comparison.

## Earlier V68 evidence

`results/v68-async-host-async-20261016T031326Z.log` is the prior bounded run.
It returned `exit_status=0` with exact A/B/A and new-parent outputs, and showed
the real async route plus cold-drain-before-invalidation. It did not include an
A4 hot replay: `async_calls=10` and `cold_drains=10`, so it is not evidence of
per-boundary hot overlap. Its initial shell post-check was overstrict because it
required a new CUDA capture inside the new-UID transition. The captured graph
had already been prepared earlier in that run; the corrected runner checks the
global cold-drain-before-capture ordering and the transition cold-drain before
parent invalidation without requiring a redundant recapture.

This fixture tests route and scheduling behavior only. It does not establish
model-level correctness or throughput improvement.

## Recorded A4 hot-replay result

The earlier A4 attempt (`v68-async-hot-host-r2`) aborted during fixture input
setup. `ggml_backend_tensor_set` used the `assume_sync=false` split-state key;
the fixture had only prequeried the inner Meta3 split. The corrected setup first
sets the allocated input buffer to WEIGHTS to freeze both split keys, queries
outer and inner K-split geometry, then changes the context-owned buffer to
COMPUTE and verifies all physical slice shapes, buffer handles, and data
aliases stayed identical. That correction is host-only; no engine source was
changed.

`results/v68-async-hot-host-r3-async-20261016T032530Z.log` is the authorized
R3 GPU proof. The fixture process returned `exit_status=0` and all numerical
and canary gates passed:

- A/B/A warm sequence plus same-parent A4 and new parent UID/different input
  buffer all matched the oracle with `max_abs=0`;
- A4 recorded `async_before=6`, `async_after=8`,
  `async_outer_ar_delta=2`, `cold_before=5`, `cold_after=5`,
  `cold_drain_delta=0`;
- overall `nested_hits=4`, `nested_misses=12`, `parent_invalidations=2`,
  `outer_ar=8`, `async_outer_ar=8`, `inner_merged_graphs=12`,
  `max_inner_subgraphs=2`;
- the new-parent transition cold-drained before the parent-generation
  invalidation; CUDA graph capture had occurred after a cold-drain earlier in
  the same run.

The first R3 runner invocation returned 1 only because a shell grep expected
two fields to be adjacent on the `HOT_REPLAY_PROOF` line. The saved-log post
checks now validate those fields independently and confirm the full transition
order; they pass without another GPU run. The GPU fixture was run once for R3.
