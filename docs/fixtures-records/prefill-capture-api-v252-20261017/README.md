# V252 scoped prefill capture API fixture

This is a bounded adaptation of the previously validated V220 four-copy graph-binding fixture. It retains the same 20 small frames, four scheduler copies, fresh graph UIDs with stable metadata descriptors, exact CPU-oracle outputs, source poisoning, and output canaries. The only runtime behavior added here is a private CUDA registry setter scope around the actual `ggml_backend_sched_graph_compute_async` call. The scope sets `ggml_backend_cuda_set_prefill_capture_hint(1)` and restores the setter's prior thread-local value through RAII.

`GGML_CUDA_GRAPHS_FORCE=0` remains set globally. The runner enables `GGML_OPTIMIZATION_PATH_LOG=1`, and the runtime checker requires the CUDA backend to report `path=cuda_graph_force_capture selected=1 reason=compatible_changed_graph_capture` on both devices. A hint-request log alone does not pass. It also requires 20 exact outputs/canaries, 20 hint entries/restores, the four-copy key pool, capture/replay calls, and structural graph builds.

This is a tiny synthetic graph/API fixture only. It makes no full-model correctness, throughput, or performance claim. No model weights are loaded.

## Static gate

```sh
python3 fixtures/prefill-capture-api-v252-20261017/test-v252-policy.py
```

`test-v252-key-recycling.py` carries forward the independent CPU-only allocator test from V220. Its tiny helper build is optional and does not initialize CUDA.

## Host link

Use the frozen V241 GGML ABI for the host link and record its exact report and CUDA source hashes. Build output is isolated under this fixture's `results/` directory:

```sh
fixtures/prefill-capture-api-v252-20261017/build-v252-host.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/src \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v241 \
  /path/to/runtime-report.json REPORT_SHA256 GGML_CUDA_SOURCE_SHA256 \
  fixtures/prefill-capture-api-v252-20261017/results/build-v252
```

## Bounded GPU run

Run only after the parent explicitly releases the GPU window and V251 runtime is frozen. The script requires `V252_GPU_WINDOW_AUTHORIZED=1`, has a 45-second timeout, refuses to overwrite logs, and runs the source contract before launching:

```sh
CUDA_VISIBLE_DEVICES=0,1 V252_GPU_WINDOW_AUTHORIZED=1 \
  fixtures/prefill-capture-api-v252-20261017/run-v252-fixture.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v251 \
  fixtures/prefill-capture-api-v252-20261017/results/build-v252/prefill-capture-api-v252 \
  fixtures/prefill-capture-api-v252-20261017/results/v252-scoped-capture.log
```

Prepared source/static checks do not imply a host link or GPU run.
