# V444 CUDA graph lease candidate plan

## Scope and source pin

- Work only in `fixtures/graph-lease-v444-root-20261018/`; do not modify active `src/`, compile the engine, connect to the server, or touch a GPU.
- Base the staged CUDA translation unit on `fixtures/six-cuda-prepare-v436-root-20261018/staged-engine/ggml-cuda.cu`, pinned SHA-256 `48752f8b408d3264e51dcffbf3828ac307970b61e7cc90ee86d2e96a0263e27d`.
- Record the existing `common.cuh` behavior as external provenance: R381 source sweep cadence 5 seconds and eviction age 10 seconds. Keep that sweep unchanged.

## Behavior to implement in the staged candidate

- Add an opt-in `GGML_CUDA_GRAPH_RETAIN_MS`; default `0` leaves the existing behavior unchanged, valid range is 0–300000 ms, invalid input fails closed to disabled.
- Keep a CPU-only per-context lease registry capped at 16 active CUDA graph entries. A lease stores only opaque graph key, `ggml_cuda_graph *` generation pointer, and last **actual successful prefill compute** time.
- At `ggml_cuda_graph_get_key` entry, refresh `last_used_time` only for non-expired leases whose key still exists in that context and whose active `unique_ptr` still has the recorded graph pointer. Drop expired or pointer-mismatched records before any stamp. This readiness-time stamp never changes the lease's actual-use time.
- After `ggml_cuda_graph_evaluate_and_capture` returns successfully, lease-touch only when the TLS prefill capture hint is exactly `1`, the call is not the V436 prepare-only path, and the selected graph is enabled, warm, and passes the existing `ggml_cuda_graph_has_executable()` check. This permits the captured-only V2 helper graph whose `graph->instance` is null. Do not lease readiness probes, eager calls, decode work, or failed/prepare-only paths.
- If a context has 16 live leases, decline a new lease; only prune already-expired or generation-mismatched records. Never force-evict a live lease. Keep all compatibility, binding, structural-signature, and CUDA graph selection checks unchanged.
- Remove per-context lease state in the CUDA context destructor after worker drain and before `cuda_graphs` destruction. Registry locking must cover only short host-side map/vector work; there are no CUDA calls under the lease mutex.
- Emit bounded first-eight-per-device admission, refresh, and decline messages to make opt-in lease behavior observable without per-call log spam.

## Deliverables and checks

- Implement the policy/registry in `graph-lease-policy-private.h`; it is included only by the staged `ggml-cuda.cu`.
- Add fake-clock CPU tests for default-off, accepted/rejected TTL bounds, exact expiry boundary, readiness not extending real expiry, pointer-generation mismatch, captured-only eligibility, prepare-only exclusion, context isolation, 16-entry cap/expired-slot reuse, and destructor drop.
- Add `stage_engine.py` that verifies the exact R436 source hash, applies only anchored lease hooks to a fixture copy, and writes the staged source, patch, and source-pin receipt. It must never edit the live engine tree.
- Run only the CPU policy tests, staged-source/hash checks, and shell/Python syntax checks. Do not build, load a model, run a GPU, or start services.
