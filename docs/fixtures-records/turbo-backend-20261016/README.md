# TurboPrefill backend fixture (V36)

This fixture builds a real three-split GGML graph on the AC922:

1. CPU `GET_ROWS` from a CPU `WEIGHTS` tensor using host I32 IDs. IDs first pass through a contiguous 4×4 VIEW/RESHAPE; the embedding result is then reshaped to `[head_dim=4, heads=4, tokens=16]` and flattened back to the unchanged `[embedding=16, tokens=16]` MetaA input.
2. MetaA3 performs four F32 GEMMs and updates a persistent F32 state with `CPY`.
3. MetaB3 consumes a contiguous mirrored activation from MetaA, performs four F32 GEMMs, and updates a second persistent F32 state with `CPY`.

The same allocated graph and state are used for the eight-frame serial reference and TurboPrefill sequence. Small integer-valued inputs and identity weights make the expected F32 path exact. Turbo output reads are registered before sealing; each frame owner is held by a weak-pointer watch, and a MetaB event recorded after the sealed work proves the owner is never released before completion. The host IDs are immediately replaced with a valid canary after `submit` returns. The fixture also requires an unsealed `finish` to fail and drain, then reuses the same sequence/position after resetting state (ABA).

The eight successful frames force reuse of the backend's fixed two-slot ring. No fallback counts as success: the run requires scheduler admission, `TurboPrefill selected=1`, single-copy Meta events, CPU gather staging, GPU event waits, eight exact output comparisons, and exact final state comparisons. It does not load a model. The run wrapper limits temporary allocation to under 1 GiB per card by enforcing the 58 C and 512 MiB-free guards and refusing pre-existing GPU compute processes.

## Build and run

`build.sh` is compile-only and accepts `R=/path/to/runtime-perf-vN`. `run.sh v37-reject` is a bounded diagnostic mode that treats the expected `CPU_prefix_has_non_weight_gather_dependency` refusal as success; it does not run serial warmup or label fallback as Turbo. `run.sh full` requires actual Turbo admission and the complete exact-result gate. Both modes require an explicit GPU window authorization and enforce the 58 C, 512 MiB-free, and <1 GiB/card guards.

```bash
R=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v37 \
  bash fixtures/turbo-backend-20261016/build.sh
TURBO_V37_PREFLIGHT_GPU_WINDOW=1 \
  R=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v37 \
  bash fixtures/turbo-backend-20261016/run.sh v37-reject

# After the root's metadata-view predicate fix is frozen in V38:
R=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v38 \
  bash fixtures/turbo-backend-20261016/build.sh
TURBO_GPU_WINDOW=1 \
  R=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v38 \
  bash fixtures/turbo-backend-20261016/run.sh full
```

Runtime, libraries, headers, binary, logs, and safety samples are hash/ldd pinned under `results/`; V37 expected-refusal logs and V38 full-pass logs are kept separately.
