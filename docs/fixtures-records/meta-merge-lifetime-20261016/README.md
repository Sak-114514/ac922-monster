# Meta graph merge descriptor-lifetime regression

This fixture exercises the V64 Meta3 in-graph AllReduce merge and the V65 fix.
It creates two independent graph generations, A and B, in one Meta compute
buffer. Each graph has the same six-node shape: two K-axis-partial F32 matrix
multiplications, each followed by a mirrored bias ADD, a VIEW alias on the first
consumer, and a final ADD. Meta reports three subgraphs, so the merge inserts
at least two AllReduce boundaries.

The fixture wraps only the three child CUDA backends' `graph_compute` callbacks.
On A's first (unmerged) warmup it records each physical node's `src[]`,
`view_src`, and `view_offs`. The next same-UID A call triggers merge. It compares
those same still-live physical descriptors immediately after the merge and
checks exact output plus canaries. V64 is expected to mutate the two consumer
`src[0]` fields per rank to temporary AllReduce descriptors. V65 must keep the
original descriptors unchanged. With V65, the fixture then runs the distinct
UID B generation and returns to A with changed inputs; all four outputs must
match the CPU oracle exactly.

Build each binary against the exact runtime:

```sh
./build-fixture.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v64 v64-final
./build-fixture.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v65 v65-final
```

Bounded three-GPU runs use the first V100 island only and enforce no preexisting
compute PID, `<58 C`, `<=512 MiB` before launch, `<=1 GiB` during execution, and
a 120-second timeout:

```sh
./run-probe.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v64 v64-final mutation
./run-probe.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v65 v65-final immutable
```

See [RESULTS.md](RESULTS.md) for the observed V64 mutation, V65 A/B/A exactness,
raw logs, and build/runtime hashes. These are small graph correctness tests;
no model was loaded and no performance claim is made.
