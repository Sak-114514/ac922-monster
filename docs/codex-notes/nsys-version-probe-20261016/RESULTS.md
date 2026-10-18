# Nsight Systems version probe (AC922)

Date: 2026-10-16. Short, bounded six-GPU probe only; no model loaded by this fixture. The existing `meta-replay-guard-cuda-20261015` fixture was reused against frozen `runtime-perf-v13` libraries. It creates two 3-GPU Meta islands, small shared F32 weights (16 elements), 8/16-row graphs, graph capture/replay, a guarded concurrent A/B graph capture, and a 64-byte cross-island mirrored tensor copy. `GGML_META_ASYNC_COPY=1`, `GGML_CUDA_META_COPY_TOPOLOGY=1`, and `GGML_CUDA_META_COPY_GPU_WAIT=1` were enabled. Six UUID order was preserved.

## Results

| Nsight Systems | Profile command result | SQLite export | Key CUDA rows in exported SQL | Fixture |
|---|---:|---|---|---|
| 2023.2.1.122-32598524v0 | Generated `.nsys-rep` (exit path completed; no profiler error) | 11,149,312 bytes; Python `sqlite3` opened it | runtime API: `cudaGraphLaunch` 42, `cudaGraphInstantiate` 21, `cudaEventRecord` 44, `cudaStreamWaitEvent` 186, `cudaMemcpyPeerAsync` 33; 36 kernel, 502 memcpy, 44 CUDA event rows | PASS |
| 2024.2.1.106-242134037904v0 | `profile_rc=0`, Generated `.nsys-rep` | 10,383,360 bytes; Python `sqlite3` opened it | same API/kernel/memcpy/event counts; additionally `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` 42, `CUDA_GRAPH_EVENTS` 87 (66 graph IDs, 21 graph-exec IDs) | PASS |

Both captures showed `meta copy topology: peer source=0 staging leader=3 parallel local fanout=2`, `meta copy slot reuse: GPU event waits`, and `async MIRRORED island copy active Meta(CUDA0,CUDA1,CUDA2) -> Meta(CUDA3,CUDA4,CUDA5), bytes=64`. The fixture compared all outputs to exact CPU expectations and all canaries passed. Thus 2024.2.1 did not fail at the capture/export boundary on this AC922; 2023.2.1 also exported real queryable SQL, although its schema does not include the newer explicit graph-trace tables.

## Artifacts

- `nsys-2023.2.1-profile.log`
- `nsys-2023.2.1-meta.nsys-rep`
- `nsys-2023.2.1-meta.sqlite`
- `nsys-2024.2.1-profile.log`
- `nsys-2024.2.1-meta.nsys-rep`
- `nsys-2024.2.1-meta.sqlite`

Both fixture runs stayed under 45 seconds. Pre-run GPU temperatures were 38–39 C with 0 MiB model allocation; after the runs the fixture released its allocations. The later nvidia-smi snapshot showed a separate `runtime-perf-v43` server (PID 27144) already occupying all GPUs at about 9.4 GiB/card; no profiler or fixture process remained. No driver, kernel, BMC, or core source was modified.
