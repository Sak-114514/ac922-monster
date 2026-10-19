# V405 six-CUDA layer-plan regression fixture

## Goal

Prepare a no-model regression for the private four-copy scheduler plan on the actual plain-CUDA topology: CPU `GET_ROWS` over 512 token rows followed by one `ADD` on each of six CUDA backends. The test uses the same `llama_layer_graph_bindings` restore/seal contract as the CPU fixture and runs 24 same-graph A/B/A frames without an intervening scheduler drain.

## Frame sequence

1. Allocate the initial full split and seal root bindings once.
2. For each of 24 frames, verify owner bindings, call `bindings.restore()` exactly once, then rewrite the CPU token-input root.
3. Call `ggml_backend_sched_try_alloc_layer_plan_rebind()`, require success, verify a fresh root UID, seal the new binding, and check that all six cross-backend input edges point at four distinct physical destination tensors.
4. Submit with `ggml_backend_sched_graph_compute_async()`, enqueue output readback into a per-frame CUDA-pinned host slot, then poison the CPU `GET_ROWS` output immediately. The test does not synchronize between frames.
5. After one final scheduler drain, check all 24 output slots and their leading/trailing canaries against an exact synthetic CPU oracle. Check the latest data in every physical destination-copy slot for all six split edges.
6. Restore bindings, change the output layout, require a pre-mutation `0` decline, restore the descriptor, then exercise the ordinary full-split fallback.

The pinned-input route is configured as `GGML_SCHED_LAYER_PINNED_INPUTS=1`, eight host source slots, and `GGML_SCHED_LAYER_PINNED_MIN_MIB=0`. `GGML_SCHED_LAYER_PEER_REUSE_FENCE=1` enables the R389 CUDA peer reader fence. Runner assertions check the independent upload count, eight-slot source wait summary, and rate-limited peer-fence activation logs. The scheduler's helper-field and op-NONE counts are retained as diagnostics; they are not treated as GPU utilization measurements.

## Topology decline checks

The fixture's local admission predicate accepts only six CUDA backends plus CPU with four copies. It checks that a two-CUDA list and an eight-copy request decline. This is a fixture-side topology check; it does not call the private API on an unsupported topology or claim to execute `llama-context.cpp`'s caller gate.

## Build and run boundary

`build-cuda-fixture.sh` is a quick host link of the fixture against the already-built R402 `libggml-base`, `libggml-cpu`, and `libggml-cuda`. It does not compile or edit engine sources and does not initialize a CUDA device. The script requires the private API header from the reloaded R402 source tree.

Execution is guarded by `RUN_LAYER_PLAN_CUDA_V405=1` and an explicit `CUDA_VISIBLE_DEVICES` value. This fixture has not been built, host-linked, or run. No GPU operation is authorized or claimed in this preparation.
