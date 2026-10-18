# Meta parent-manifest qualification

This fixture qualifies the opt-in parent-signature/manifest guard added in the
V70 closure, using the later frozen runtime supplied to `build-fixtures.sh`.
It owns only these fixture sources and scripts; it does not modify engine
sources. The V68 nested Meta2{Meta3,Meta3} fixture is copied here so this
directory remains reproducible and does not depend on another fixture's
working files. The real PLE view fixture is copied from the V63 validation and
extended with a scheduler reset/reallocation case.

## Gates

`meta-parent-manifest-fixture.cpp` retains the exact tiny-F32 oracle, output
canaries, nested TP3 merge, cache hit/miss, async P2P6 path, A/B/A and A4 hot
replay gates. It adds a distinct second graph and second input-buffer binding
that deliberately reuses A's parent graph UID. During that transition the
fixture requires a `reason=parent_generation` child invalidation and exact
oracle output, proving the manifest detected the changed binding despite the
reused UID. It then restores A at the same UID and retains the separate new-UID
transition with cold-drain-before-invalidation ordering. The process log must
also contain parent-manifest `selected=1` evidence and a nonzero forced-rebuild
count in the Meta teardown summary.

`ple-weight-view-fixture.cpp` reads only the real 81,920-byte F16 tensor
`blk.1.ple_conv1d.weight` from the GGUF, builds the production-shaped four-tap
view/cont/reshape/cast chain, and retains exact static weight-byte checks. It
checks A/B/A on standalone TP3 and both nested islands, then resets and
reallocates the nested scheduler graph, which must assign a fresh graph UID.
It revalidates the packed PLE root bytes and checks exact A output on both
islands. Output readback has canaries. It requires the parent-manifest path to
remain selected through the reset and both post-reset graph calls to complete;
a forced rebuild is not required for this new-UID transition.

The suite runner executes the two bounded probes sequentially. Both stop before
58 C, 1 GiB/card, 180 seconds per process, or any preexisting GPU PID / more
than 512 MiB preexisting use. It requires a separate root-issued authorization
environment variable and cannot be used to launch while the model owns the
cards. Neither probe loads the full model; the PLE fixture seeks and reads
only its one small tensor payload.

## Host build

After the runtime closure is frozen, sync this directory to AC922 and build
against that exact closure:

```sh
./build-fixtures.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v71 v71-parent-manifest
```

The build is host-only. It creates two fixture executables and a private
backend plugin directory in this directory; it does not run a CUDA API or
touch a GPU.

## GPU run (only after root grants the window)

```sh
META_PARENT_MANIFEST_GPU_WINDOW_AUTHORIZED=1 \
  ./run-suite.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v71 v71-parent-manifest
```

The runner requires six free V100 UUIDs in the validated 3+3 order and records
each probe's log, binary hash, GPU pre/post state, and exit status. GPU
execution is not authorized by this file; the root grant remains required.

## Current status

- Host build against frozen V71 succeeded.
- The nested parent-manifest fixture passed all gates, including the same-UID
  changed-buffer proof and the original A4-to-new-UID cold-drain ordering.
- The real PLE A/B/A plus scheduler reset/rebuild fixture passed after it began
  creating a fresh graph context and fresh tap views after reset. Earlier
  stale-graph failures are documented in `STATUS.md`; they were invalid fixture
  use of scheduler-reset tensors, not a runtime regression.
