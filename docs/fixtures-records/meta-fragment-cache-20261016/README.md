# Nested Meta fragment cache fixture

This fixture exercises the V67 default-off fragment cache through the actual
Meta2{Meta3A,Meta3B} backend graph path. It uses tiny F32 tensors and four
outer partial boundaries across a Meta2x3 graph. Each outer parent fragment
contains an inner Meta3 TP partial -> mirrored ADD -> another partial node,
so the child graph must build a merged TP3 graph containing its reduction
boundary (reported as two subgraphs). The outer Meta2 then performs its native
six-rank P2P reduction at the two outer boundaries.

The same parent graph and UID runs with changed inputs A/B/A. The fixture checks
every frame against an exact F32 CPU oracle and uses output canaries. It runs in
a fresh process with the fragment cache OFF and ON, so both modes must produce
the same expected values. In the ON process it also requires a real cache hit,
initial misses, inner TP3 merge evidence, and invalidation when a new parent UID
is paired with a different input buffer/data binding.

No model is loaded. The runner is gated by
`META_FRAGMENT_CACHE_GPU_WINDOW_AUTHORIZED=1`, checks all six cards for existing
compute PIDs and temperature/memory bounds, and has a bounded timeout. The V66
runtime is not used; build against the frozen V67 closure:

```sh
./build-fixture.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v67 v67
```

After a root-assigned GPU window, run both modes sequentially:

```sh
META_FRAGMENT_CACHE_GPU_WINDOW_AUTHORIZED=1 ./run-probe.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v67 v67 off
META_FRAGMENT_CACHE_GPU_WINDOW_AUTHORIZED=1 ./run-probe.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v67 v67 on
```

Host compilation is permitted during other GPU work. GPU execution is not.

## Recorded V67 result

The corrected ON gate requires a merged child graph with at least two
subgraphs. That is the observed representation of the single inner TP3
reduction boundary; the two outer P2P6 boundaries are counted separately.

- `v67-host-off-20261016T024050Z.log`: PASS; A/B/A and new parent UID/input
  binding exact, output canary intact.
- `v67-host-on-20261016T024104Z.log`: preserved initial FAIL. The only failure
  was the fixture's old `>=3` subgraph expectation; this run did not reach the
  new-parent invalidation case.
- `v67-host-r2-on-20261016T024529Z.log`: PASS after correcting that expectation
  to `>=2`. It reports `nested_hits=4`, `nested_misses=12`,
  `parent_invalidations=2`, `outer_ar=8`, and
  `inner_merged_graphs=12, max_inner_subgraphs=2`; A/B/A and the new
  parent UID with a different input buffer/data binding are exact and the
  canary remains intact. Exit status is 0.

The rerun used `GGML_META_DEBUG=1` to capture the inner merge evidence. This is
a tiny correctness fixture only; it does not establish model-level behavior or
performance gain. No further GPU run is part of this result.
