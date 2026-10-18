# Relay6 10 MiB Dispatcher Fixture

## Finding

The first two V77 fixture binaries were compiled without `GGML_CUDA_USE_GRAPHS`, while the linked V77 CUDA library was compiled with it. In `ggml/src/ggml-cuda/common.cuh`, this macro changes `ggml_backend_cuda_context` layout: the host-only layout probe measured 4664 bytes with the macro off and 4728 bytes with it on. The graph-enabled layout inserts `cuda_graphs` at offset 3584 and moves `concurrent_stream_context` and `pools` by 64 bytes. The fixture constructs the context inline, so the mismatch made the object ABI-invalid.

V77's `build-reuse/compile_commands.json` confirms `-DGGML_CUDA_USE_GRAPHS` is required for the runtime. The fixture builder now passes it as well.

## Evidence

- `v77-dispatch10m-host2-dispatcher-20261016T055148Z.log`: pre-fix build, 12 dispatcher calls logged, then SIGSEGV before graph readiness.
- `v77-dispatch10m-diag-dispatcher-20261016T055544Z.log`: same failure with a diagnostic preload; the preload itself could not unwind this fault. Both failure logs are retained.
- Corrected binary tag: `v77-dispatch10m-abi-corrected`.
- Corrected binary SHA256: `f57dff818213bee1352e00e22e9e91bfb44063c835a20fa287060f6f5d72b872`.
- Corrected run: `v77-dispatch10m-abi-corrected-dispatcher-20261016T060017Z.log`.

The corrected **segmented** route completed the actual dispatcher path on all six ranks for two independent communicators. Both graph captures completed, and all six A/B/A outputs per mode were exact against the rank-ordered FP32 oracle with canaries intact. The two communicator allocations plus graph state used about 116 MiB per GPU. Mean wall time was 5.260 ms and 5.288 ms per replay for contexts 0 and 1; mean maximum GPU-event time was 5.246 ms and 5.273 ms.

The corrected **relay** route did not reach dispatch. Its initialization exceeded the fixture's 512 MiB per-GPU safety ceiling (GPU 1 reached 646 MiB), so the harness terminated it. No relay numerical, timing, or production-capacity result is claimed. Postflight showed all GPUs idle with 2–8 MiB reported use, 39–40 C, and no compute PIDs.

## Scope

Only the fixture build definition changed for the corrected run; no core source, system setting, driver, fan state, or model was changed. The relay case needs a separately authorized bounded run with a memory ceiling sized from an explicit allocation ledger, or a smaller simultaneous fixture footprint. The current result does not validate the 10 MiB relay route.
