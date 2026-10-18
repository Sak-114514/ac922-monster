# Stateful P4 fixture on frozen V13 runtime

Date: 2026-10-15. One GPU fixture run with CUDA/NVTX node-level tracing. No model was loaded.

## Runtime and launch

- Binary: `standard-opt-20261014/fixtures/island-frame-state-20261015/island-frame-p4-state-v13`
- Runtime: `standard-opt-20261014/runtime-perf-v13/lib`
- CUDA order: fixed six UUIDs in 3+3 NVLink order; GPU 0-2 PCI `0004:04-06`, GPU 3-5 PCI `0035:03-05`.
- `LD_LIBRARY_PATH` was set explicitly to runtime-perf-v13, CUDA 12.4 ppc64le, and NCCL paths.
- Fixture printed `/proc/self/maps`; all four GGML libraries resolved from `runtime-perf-v13/lib`.
- Capture: `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node`.

## Correctness and cleanup

Both request epochs passed exact comparison for 4 stage outputs, full per-island KV marker arrays, and per-layer GDN state. The ring reported backpressure at two active frames, rejected a stale ticket after slot reuse, drained both stage events, and reset cache state before epoch 2. No `FAIL`, CUDA error, timeout, or OOM line appeared in the run log. The fixture and Nsight process exited; post-run GPU memory was 0 MiB on devices 0-3, 8 MiB on 4, and 3 MiB on 5 (ambient driver allocations), with temperatures 39-40 C.

## Overlap result

There was **no cross-island kernel overlap** for A1 versus B0 in either epoch. The CUDA kernel activity intervals are already in the same Nsight timeline time base:

| Epoch | B0 devices 3-5 kernel span | A1 devices 0-2 kernel span | Result |
|---|---:|---:|---|
| 1 | 2,598,832,282-2,621,293,450 ns | 2,623,483,001-2,627,364,244 ns | 0 overlapping kernel pairs; 2.19 ms gap |
| 2 | 2,640,078,080-2,643,040,730 ns | 2,643,280,654-2,645,668,104 ns | 0 overlapping kernel pairs; 2.40 ms gap |

The host submission sequence was correct and state dependencies were honored, but the B0 submit path returned only after its GPU kernels had finished. This fixture proves safe shared-state ordering, not useful pipeline overlap or performance gain. The next design needs a genuinely nonblocking stage submission path before model integration.

## Readiness-gated hot replay follow-up

A second V13 build added the private `ggml_backend_sched_meta_replay_ready(sched, exact_root_graph)` query for all four frame/stage schedulers. Warm epochs 1 and 2 reported all four roots not ready. After warm epoch 3, all four exact scheduler roots reported ready; the subsequent hot epoch 4 logged `mode=replay` for all twelve child CUDA contexts (six devices across two frame slots). The query and graph-path markers therefore establish a real replay-ready hot path without comparing the root UID to a split UID.

The hot replay still showed no A1/B0 overlap:

| Hot epoch 4 event | Interval |
|---|---:|
| B0 host NVTX span | 1,976,466,113-1,976,961,429 ns |
| B0 kernels on devices 3-5 | 1,976,535,888-1,976,956,042 ns |
| A1 host NVTX span | 1,976,983,106-1,977,443,574 ns |
| A1 kernels on devices 0-2 | 1,977,043,820-1,977,444,166 ns |

That leaves an 87.8 us gap from the last B0 kernel to the first A1 kernel and zero cross-island kernel intersections. B0's host stage span is 495.3 us; its three `cudaGraphLaunch` API calls take 406.3, 370.0, and 359.7 us. A1's host stage span is 460.5 us. The serial host submission still returns after the prior stage has nearly completed, even when graph readiness is true. No throughput claim is supported.

## Artifacts

- `results/stateful-p4-v13.nsys-rep` — full Nsight capture.
- `results/stateful-p4-v13.sqlite` — exported event database.
- `results/cuda-gpu-trace.csv` — Nsight kernel trace.
- `results/run-v13.log` — runtime library map, stage checks, and completion log.
- `results/pre-run-gpu.csv`, `results/post-run-gpu.csv` — GPU memory/temperature snapshots.
- `results/stateful-p4-v13-ready.nsys-rep` / `.sqlite` — capture with readiness-gated warm passes and hot epoch 4.
- `results/cuda-gpu-trace-ready.csv` and `results/run-v13-ready.log` — replay markers, private readiness queries, state checks, and loaded-library maps.
- `results/pre-hot2-v13-gpu.csv`, `results/post-hot2-v13-gpu.csv`, `results/pre-ready-v13-gpu.csv`, `results/post-ready-v13-gpu.csv` — device snapshots around follow-up captures.

The fixture is synthetic and does not establish full-model KV/QSA/GDN semantics, production safety, or throughput. It should remain a diagnostic gate only.
