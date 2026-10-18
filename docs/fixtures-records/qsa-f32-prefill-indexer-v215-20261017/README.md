# V215 F32 QSA prefill-indexer fixture

This operator-only fixture compares the production standard F32 QSA headloop with `ggml_lightning_indexer()` for four heads and 128 dimensions. Each case uses deterministic F32 pooled keys made from groups of four synthetic cells, F32 queries, all-one F32 head weights, and an all-zero F16 mask. No model or GGUF is loaded.

Cases are `q256-k4096`, `q512-k8192`, and `q1024-k8192`. For each arm the fixture builds and allocates one GGML graph, warms it twice, then reports nine warm `graph_compute_async + backend synchronize` wall samples, their median/min/max, and gallocr workspace bytes. It compares every F32 score and reports exact-bit differences, max/RMS absolute error, max relative error, non-finite counts, and stable selected-cell agreement. Top-k reporting follows the existing V178 four-cell expansion convention (`min(4*K, 2048+4-1)` IDs); it reports both equal-position rate and set overlap.

Numerical deltas and selected-cell agreement are reported without an invented acceptance threshold (`REPORT_ONLY`). Non-finite outputs fail the run because those are invalid score results. The run must also emit an actual CUDA route trace with `path=tile`, four heads, the requested query count, and `type=f32`; the runner rejects a missing or vector-path trace.

This fixture measures only synthetic operator behavior on one CUDA backend. It does not prove model-level QSA semantics or full-model speed. Build and GPU scripts are separately gated. Do not run either while another workload owns the GPU or before a fresh window is granted.

```sh
V215_BUILD_AUTHORIZED=1 \
RUNTIME_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-vXXX \
  ./build-hostlink-v215.sh
```

```sh
QSA_F32_PREFILL_GPU_WINDOW_GRANTED=1 \
RUNTIME_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-vXXX \
DEVICE=3 CASE=q256-k4096 ./run-cuda-once-v215.sh
```
