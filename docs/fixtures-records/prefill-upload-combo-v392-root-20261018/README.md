# V392 preparation: pinned upload stream + tail-only graph helper

Status: **preparation only; not launchable**. No engine sources outside this fixture were edited. No compilation, model load, remote command, or GPU activity was performed.

## Workload

The scenario order remains the exact Paris 8192-token native canary, then one clean 64K/8 request, then one diagnostic 64K/8 request. The existing 90-second cooldowns remain between those phases. Results stay separate; clean and diagnostic metrics must not be averaged.

V392 combines the V386 harness with:

- R389 `GGML_SCHED_LAYER_PINNED_UPLOAD_STREAM=1`, required at source and profile level; the controller checks a runtime `requested=1 selected=1` marker.
- The forthcoming R390 tail-only helper and runtime, still represented by root-fillable placeholders. The controller and hostlink reject placeholders.
- Monotonic microsecond timestamps on all target and MTP graph-phase begin/end markers.
- `GGML_META_GPU_SPANS=1` and period 1 before each context is created. Scheduler span ACTIVE remains off for native/clean and turns on only for diagnostic 64K.

The D2/MTP, V373 flags, QSA mask2D, device tail, PLE placement and stripe quotas, UB512, 64K contexts, KV pad 65536, and other math/quality settings remain as V386. `ROOT_PREFILL_BATCH` and target chunk size stay 32768. A logical batch of 65536 is recorded only as a future budget-review item. The source currently caps active target chunks at 32768 even if the context batch capacity is raised; a larger `n_batch` would still enlarge batch capacity, while the graph/workspace delta has not been quantified. With approximately 2.8 GiB GPU5 headroom reported, the profile keeps 32768 and makes no allocation estimate for 65536.

## Root-owned release steps

1. Pin the frozen R390 runtime ID/build report SHA and R390 helper path/SHA in `v392-root-profile.json`.
2. Run the CPU-only hostlink template with `V392_BUILD_AUTHORIZED=1`; update runner and receipt identities in the profile.
3. Register the exact `root-v392-inband` prefix with the existing root guard and update `launch_eligible` only after checking the profile and guard conditions.
4. Launch only through `run-v392-root.py` with `V392_ROOT_AUTHORIZED=1` and sudo input on stdin. The controller saves clean/diagnostic result JSON and raw runtime/trace before stage/activation postchecks; a missing diagnostic-span or pinned-upload selection marker fails the run while preserving those artifacts.

## Static verification

```sh
python3 verify_v392_static.py
```

This checks source/profile/scenario invariants, controller ordering and failure gates, and hostlink shell syntax. It does not build or run the model.
