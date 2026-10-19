# V404 actual GGML CUDA graph launch-cost fixture

**Preparation only.** No hostlink, GPU, server, model, or service was used by this preparation.

## Question

For an actual GGML CUDA graph with an 8192-ADD chain, compare the host cost of cold capture/launch and warm graph replay under three V2 helpers:

- R5 device controller graph
- R6 host-child device graph launch
- R398/R7 host-child single-exec without controller allocation

Run each helper in a separate process with the same binary, graph, input sequence, dimensions, and exact output oracle. The candidate keeps V396's A/B/A sequence and 64/128 element graph shapes, expands from 64 to 8192 ADD nodes, and reports host `ggml_backend_graph_compute_async` time plus total time through backend synchronization. Cold first frame and warm replay medians are separate per shape.

The primary comparison has **no CUPTI loaded** and no CUDA events. Optional `--trace TRACE_PATH` loads the existing V223 CUPTI library only around the GGML A/B/A phase. Optional `--cuda-events` records CUDA events on the real GGML backend stream. Those extra observations are kept in separate output files and are not mixed into the primary host timing.

## Correctness and bounds

Every frame changes the input buffer, executes the same 8192 ADD nodes, verifies input bytes remain unchanged, and checks every output element equals `input + 8192`. The metadata arena scales as a fixed 16 MiB base plus 4 KiB per ADD node. Device tensors remain small because each graph has only 64 or 128 elements.

The binary runs the existing direct V2 readiness and V328 cap-32 lifecycle gates before the actual GGML fixture. `run-matrix-v404.sh` runs the three helper modes sequentially and requires a fresh output directory. It defaults to no trace and no CUDA events; `V404_RUN_AUTHORIZED=1` is required before any future GPU invocation.

## Hostlink

`root-host-link-v404.sh` is prepared for a root-owned CPU-only compile and `ldd -r` check against the frozen R400 runtime. It has not been run. No kernel, engine, or runtime sources are changed here.
