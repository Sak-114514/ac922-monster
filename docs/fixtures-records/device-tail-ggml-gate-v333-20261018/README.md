# V333 six-device GGML device-tail gate

This is a bounded synthetic GGML backend fixture. It reuses the V318 graph-binding fixture's two root graphs, A/B/A/B ordering, four scheduler copies, 20 frames, six CUDA stage assignments, 120 exact CPU-oracle outputs, canaries, graph-owner checks, unique scheduler UIDs, and readback fences. It loads no model or model weights and makes no throughput claim.

The only source behavior added is an RAII scope around each real `ggml_backend_sched_graph_compute_async` submission. It resolves `ggml_backend_cuda_set_prefill_capture_hint` through the CUDA registry, sets the thread-local hint to 1 on the calling submit thread, and restores the previous value on that same thread. The source refuses to run unless CUDA graphs are enabled, `GGML_CUDA_GRAPHS_FORCE=1`, the device-tail opt-in is set, and the helper path is absolute.

The bounded runner pins all six physical V100 UUIDs and requires the runtime logs to show `cuda_graph_force_capture selected>0` and `cuda_graph_device_tail selected>0` for **each of CUDA devices 0–5**. A config line or hint-scope line alone cannot pass. The helper DSO must load from the V328 diagnostic R2 helper at `fixtures/device-tail-bridge-v328-20261018/build-r2/libac922-tail-bridge.so` (SHA256 `49ad8e2a500e5cbf24e50eb6e5d1a12888181c9d70d884cde72e4a02b23801b1`). It supplies a private hardware test-policy file containing `4` (fenced serial plus structural graph mode) and requires every CUDA context to log `disable_env_present=0 structural_active=1 test_policy=4`. The gate also requires 20 hint/restoration pairs on the same submit thread and the original `120/120` exact output/canary gate.

## Host link

The host build is pinned to frozen V329 (`build_run_id=20261017T194116.616255Z`, `notes/V329-build.json` SHA256 `d0935c5278ac62c644947cda159235e273f795934ae4afcdfc49b70d11c2f5fe`). It compiles a small host fixture only; it does not initialize CUDA devices or load a model.

~~~sh
fixtures/device-tail-ggml-gate-v333-20261018/build-host.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/src \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v329 \
  v333-v329
~~~

The build writes the binary, DSO resolution, runtime/source hashes, and a hostlink receipt under `build-v333-v329/`.

## Bounded GPU gate

Only root may authorize and launch this gate after the hostlink succeeds and the six GPUs are available. It refuses to run unless `V333_GPU_WINDOW_AUTHORIZED=1`, `GGML_CUDA_DISABLE_GRAPHS` is unset, no `root-v*` service or compute process is active, all GPUs are below 58 °C and have at least 512 MiB free, and telemetry remains available. Its host deadline is 90 seconds; it never calls `cudaDeviceReset`.

~~~sh
V333_GPU_WINDOW_AUTHORIZED=1 python3 \
  fixtures/device-tail-ggml-gate-v333-20261018/run-bounded.py \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v329 \
  v333-v329
~~~

The runner creates a fresh backend plugin directory pointing at the supplied frozen runtime, sets CUDA graph force-capture and the absolute V328 diagnostic-R2 helper path, and checks the actual per-device optimization summaries after the 20-frame run. GPU execution is pending root authorization; a hostlink or static pass is not evidence of selected device-tail launches.
