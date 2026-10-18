# Parent-walk V77 integration qualification

This directory contains a private integration copy of the V75-qualified
nested Meta2x3 composition and PLE owner-view fixtures. It does not change the
engine source or the sibling generic CPU tests. The V77 parent-walk path is
enabled only in these fixture runners with `GGML_META_PARENT_WALK_FAST=1` and
`GGML_META_PARENT_WALK_TRACE=1`.

The nested fixture retains all A/B/A, A4 AR6-equivalence/cold-drain,
same-UID changed-buffer, new-UID invalidation, exact-oracle, canary, composition
and PLE gates. Its stable 16-call window changes input contents over frames
5/6/7 while keeping the same parent UID and buffer bindings. It additionally
requires at least 12 selected parent-walk log events in that window, with both
`original_hits` and `prepared_hits` increasing by at least 12. This evidence
must come from `meta-parent-walk ... selected=1` records, not just the final
context summary. The shell runner independently checks the proof marker and
requires the same-UID new-buffer invalidation path.

The BMC/host/GPU and 58 C/512 MiB/1 GiB checks are retained from the bounded
V75 runners. `run-parent-walk-bounded.sh` caps the combined nested-plus-optional
PLE probe window at 60 seconds; it refuses to run without an explicit
`PARENT_WALK_GPU_WINDOW_AUTHORIZED=1`. Do not run it until root issues the GPU
grant.

The runner sets `GGML_META_PARENT_WALK_TRACE=1`. V77 honors it by logging each
prepared hit, allowing the fixture callback to prove both counters grow during
the marked 16-call stable window. Keep this opt-in scoped to the bounded fixture
run; production logging remains throttled.

Host-link against the frozen closure without initializing CUDA:

```sh
./build-integration.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v77 v77-parent-walk-host2
```

This links only the two fixture executables and verifies all four GGML shared
libraries resolve to the supplied V77 runtime. It does not execute either
binary.
