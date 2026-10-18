# Nested Meta fragment cache fixture

This fixture exercises the V67 default-off fragment cache through the actual
Meta2{Meta3A,Meta3B} backend graph path. It uses tiny F32 tensors and four
outer partial boundaries across a Meta2x3 graph. Each outer parent fragment
contains an inner Meta3 TP partial -> mirrored ADD -> another partial node,
so the child graph must build a merged TP3 graph with at least two internal
partial boundaries. The outer Meta2 then performs its native six-rank P2P
reduction at the two outer boundaries.

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
