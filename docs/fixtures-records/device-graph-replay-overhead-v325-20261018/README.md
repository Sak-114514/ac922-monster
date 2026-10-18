# V325: host graph launches vs device-tail graph replays

Status: source and CPU-only hostlink preparation. This fixture has not been executed on a GPU. Root review and a separate GPU release are required before running `run-v325-bounded.py`.

## Question and bounded workload

On one V100/SM70, compare two ways of executing the same child CUDA graph eight times:

1. Host path: eight `cudaGraphLaunch(host_child_exec, stream)` calls.
2. Device-tail path: one host launch of a one-node device graph whose controller tail-launches the same child workload eight times.

The child graph contains either 128 or 1,024 independent single-thread integer-oracle kernel nodes. Every node increments its own integer counter. Both paths must produce exactly eight hits for every node; the device path must also report exactly eight child launches. No model, cuBLAS, expert kernel, allocation during timing, or input data is used.

Each measurement reports three distinct quantities:

- `host_api_ms`: CPU time spent in the graph launch API call(s) only (8 host calls versus 1 controller call).
- `whole_wall_ms`: wall time from before the measured launches until stream synchronization completes.
- `cuda_event_ms`: CUDA event interval around the measured stream work. CUDA 12.4 defines the tail launch as part of the parent graph environment; downstream stream work follows completion of the graph environment and its tail-launched children.

One unmeasured warmup per mode precedes each timed case. Timed replays remain fixed at eight. The two node-count cases run in alternating mode order to reduce systematic warm-order bias. Graph creation, instantiate, upload, warmup, oracle readback, and verification are outside the measurements.

## Hard bounds

- One visible GPU; runtime gate requires V100 SM70 and unified addressing.
- Node count is compiled as the fixed pair `{128, 1024}` and has a hard maximum of 1,024.
- Replays are compiled as exactly 8; device tail self-relaunch has no unbounded loop.
- The root-reviewed bounded launcher applies an 85-second process deadline, 58°C ceiling, 512 MiB free-memory floor, and refuses to run alongside an active `root-v*` service or an existing compute PID on the target GPU.
- The preparation hostlink compiles only; it does not run the executable, call CUDA APIs, or inspect GPU state.

## Future command (not run during preparation)

After root review/release, use the bounded wrapper, not the binary directly:

```sh
python3 run-v325-bounded.py
```

The wrapper executes `device_graph_replay_overhead_v325 --host-vs-device-tail` under `/usr/bin/timeout` for at most 85 seconds. No device reset is used.

## Interpretation limits

This is an API/launch-path microtest, not an inference benchmark. A win would show only that a tiny same-device graph with 128 or 1,024 trivial nodes is cheaper to replay through a device-tail chain on this V100. It does not establish benefits for llama.cpp scheduler graphs, NCCL, graphs containing unsupported nodes, or model workloads. `host_api_ms`, `whole_wall_ms`, and `cuda_event_ms` are separate views and must not be added together.

## CUDA 12.4 references

- [CUDA C++ Programming Guide: tail launch and graph execution environments](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#device-graph-launch)
- [CUDA Runtime API: executable graph launch ordering](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-runtime-api/group__CUDART__GRAPH.html)

V322's already-observed V100 control gate used the same `cudaGraphInstantiateFlagDeviceLaunch`, `cudaGraphUpload`, `cudaStreamGraphTailLaunch`, and bounded self-tail-relaunch API pattern. Its source and result are pinned in `manifest.json`.
