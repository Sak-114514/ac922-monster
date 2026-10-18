# Nested asynchronous P2P6 fixture

This V68-only microfixture reuses the validated tiny F32 `Meta2{Meta3A,Meta3B}`
topology. It requires the V67 cache correctness gates unchanged: same-parent
A/B/A input changes, a new parent UID with a different input buffer, exact CPU
oracle output, and output canaries. It does not repeat the OFF comparison.

The async window queues the final A restore graph, then submits the new-UID B
graph without a host readback or synchronization between those calls. The test
requires the real `p2p6-rank-order-async` route and a six-rank cold-drain during
the parent-generation transition, before cache invalidation. It then drains
once and checks both queued outputs exactly.

Build only the small host fixture against the frozen V68 closure:

```sh
./build-fixture.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v68 v68-async-host
```

After a root-granted and freshly checked GPU window, run the single ON arm:

```sh
META_FRAGMENT_CACHE_GPU_WINDOW_AUTHORIZED=1 ./run-probe.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v68 v68-async-host
```

The script requires six free GPUs, each below 58 C and 512 MiB preexisting
usage; the fixture itself is limited to 1 GiB/card and 180 seconds. It sets
`GGML_META_FRAGMENT_CACHE=1`, `GGML_META_NESTED_ASYNC=1`, and the replay/view
guards required by V68. This checks route selection and cold-drain ordering; it
is not a model-level result or a throughput claim.

## Recorded V68 run

`results/v68-async-host-async-20261016T031326Z.log` is the single authorized
GPU run. The fixture process itself returned `exit_status=0` and passed every
numeric/canary check:

- `async_outer_ar=8`, with actual `route=p2p6-rank-order-async` records;
- the new-parent transition logged `meta-nested-async cold-drain` before
  `reason=parent_generation` invalidation;
- A/B/A and new parent UID plus a different input buffer all matched the exact
  oracle; every output had `max_abs=0` and an intact canary;
- `nested_hits=4`, `nested_misses=12`, `parent_invalidations=2`,
  `inner_merged_graphs=12`, `max_inner_subgraphs=2`.

The first runner post-check returned 1 because it required a *new* CUDA graph
capture inside the new-parent transition. The log instead shows captures after
cold drains earlier in the run, followed by a cold drain and parent-generation
invalidation during the transition; there was no additional `mode=capture`
inside that transition. The saved log proves the cold-drain and capture order
without requiring a redundant recapture. `run-probe.sh` now checks that a
global cold-drain precedes capture, and that the transition cold-drain precedes
parent-generation invalidation. This post-check was validated against the
saved log only; the GPU fixture was not rerun.
