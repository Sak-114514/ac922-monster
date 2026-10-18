# V71 qualification status

Date: 2026-10-16 UTC

The fixture was built against frozen
`/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v71`.
No model was loaded. The PLE probe read only the GGUF metadata and its single
81,920-byte F16 weight tensor.

## Nested parent-manifest fixture: PASS

Build tag: `v71-parent-manifest-a2`

- Host build/link passed. Binary SHA-256:
  `89249acc6c0891b583e51d908e0419b8ade7eae488b3853810a9dfc1204cd97c`.
- Run log:
  `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/meta-parent-manifest-20261016/results/v71-parent-manifest-a2-parent-manifest-20261016T040713Z.log`
- Exit status: 0. `PASS META_PARENT_MANIFEST` recorded exact oracle and
  canaries, nested A/B/A, A4 hot replay, same UID with a separate graph and
  input buffer, and the original new-UID transition.
- Parent summary: `calls=8 generations=4 forced_rebuilds=3`.
- Same-UID/new-binding transition: two `reason=parent_generation`
  invalidations, exact output, canaries intact.
- A4 hot replay: two async outer reductions and no cold-drain delta.
- New-UID cold-drain ordering: global cold drain at log line 67, capture at 92,
  transition cold drain at 132, and invalidation at 133.
- GPU state after the run: all six cards idle, 1–79 MiB used, 38–40 C, no
  compute PIDs.

## PLE weight-view scheduler-reset fixture: PASS AFTER GRAPH REBUILD

Build tag: `v71-reset-owner-a4`

- Host build/link passed. Binary SHA-256:
  `6c33dca2b92e89a10b727be647a103abab3462c1cf86c94577daf91770a59c28`.
- Run log:
  `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/meta-parent-manifest-20261016/results/v71-reset-owner-a4-ple-parent-manifest-20261016T041226Z.log`
- Exit status: 0. Real PLE standalone/nested A/B/A, scheduler reset with fresh
  graph UID (`6 -> 16`), exact post-reset output on both islands, weight-byte
  checks, parent-manifest selection, and output canaries passed.
- Parent summary: `calls=8 generations=2 forced_rebuilds=1`.
- After reset, the new scheduler/graph context registered new owner IDs 21,
  25, and 26 for the PLE weight-root views; all taps remained attached to
  `blk.1.ple_conv1d.weight` and the post-reset graph completed.
- GPU state after the run: all six cards idle, 1–76 MiB used, 38–39 C, no
  compute PIDs.

## Reset-contract diagnosis

Two earlier PLE attempts intentionally remain in the remote results directory.
They reused the same allocated graph tensors after `ggml_backend_sched_reset`
and aborted at `ggml-backend-meta.cpp:906` with “no live owner for external
view”. The backend contract says reset invalidates those prior allocated
tensors; the fixture now frees the old scheduler and graph context, rebuilds
fresh view tensors, and keeps only the static GGUF weight and input contexts.
The V69 attempt used the same stale-graph fixture and is not a version
comparison. The corrected V71 run passed without core changes.
