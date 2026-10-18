# V210 scheduler-backed Q5_K output-head gate

Run tag: `v210-sched-head-q5k-20261017-once-01`  
Runtime: frozen `runtime-perf-v196`  
Scope: isolated output-head operator using the actual scheduler; no model load.

## Input and scheduler graph

The fixture reused V203's GGUF metadata reader and `output.weight` payload loader. It read only the real Q5_K head payload: shape `[2560, 248320]`, 437,043,200 bytes, row size 1,760 bytes, GGUF data offset 11,026,368. The four 62,080-row shards contain original Q5_K bytes; no conversion or requantization occurred. The M=1 F32 activation was synthetic and allocated on GPU5.

One `ggml_backend_sched` ran with `parallel=false`, one copy, and backend priority/order GPU5, GPU0, GPU1, GPU3, CPU. Each `MUL_MAT` was explicitly pinned to the corresponding row owner (0, 1, 3, 5). All four producers were expanded into the graph before the GPU5 concat chain and argmax were built and pinned. The run explicitly allocated the scheduler graph, then used scheduler async compute and synchronization for warmup and seven timed iterations.

The runtime confirmed `n_backends=5`, `n_copies=1`, and `n_splits=4`. Actual assignments matched the requested owners: matmuls on CUDA0/1/3/5 and all concat/argmax nodes on CUDA5. The raw log prints node order: four `MUL_MAT` producers, three `CONCAT` nodes, then `ARGMAX`.

## Correctness and timing

All 248,320 gathered F32 logits matched the full-GPU5 V203 reference bit-for-bit (`bitdiff=0`, `max_abs=0`). The argmax matched exactly (`41867`).

| Path | Warm M=1 wall median | Samples |
|---|---:|---:|
| Full head on GPU5 | 769.174 µs | 7 |
| Four shards through one scheduler, gather and argmax on GPU5 | 681.713 µs | 7 |

This isolated measurement is 1.128× faster for the scheduler-backed sharded path. It is not a full-model or MTP throughput result.

## Kernel path and limits

The runner used `GGML_SM70_DECODE_LINEAR_CHUNKS=4` and `GGML_SM70_DECODE_LINEAR_PACKED=1`, matching the V176 decode profile. The runtime log reports Q5_K (`type=13`) with `ne11=1`; the V196 dispatch source requires more than four columns for the packed-chunk branch, so this shape stays on the ordinary single-column MMVQ path. The prior V203 [source audit](../../mtp-vocab-row-shard-v203-20261017/results/v203-dispatch-audit.md) found no tensor-name gate on that path; the optional Monpack intercept is based on registered weight data, and this run logged `reg=0`. A source search of the current V208 tree found its nearby name-based matmul special case at `ggml-cuda.cu:2853`, restricted to BF16 `blk.1.ple_value.weight`; it does not match Q5_K. Thus the fixture's `v210_output_weight_q5_k_rows_*` names do not select a separate Q5_K kernel from production `mtp_vocab_q5k_rows_i` names. The runtime tested here remains the frozen V196 closure.

The fixture proves scheduler allocation/ownership and exact operator output using synthetic hidden input. It does not prove production MTP hidden-state equivalence, graph reuse semantics, model sampling, or full-model speed. No core source or CUDA kernel was changed.

## Runtime and artifact hashes

`ldd` resolves `libllama.so.0`, `libggml.so.0`, and `libggml-base.so.0` from `runtime-perf-v196/lib`. CPU and CUDA backend DSOs resolve from `build-reuse/bin`; their SHA-256 values match the V196 copies. The complete resolution list is in [`v210-host-link-ldd.txt`](v210-host-link-ldd.txt), and the binary/source/log and five DSO hashes are in [`v210-runtime-sha256.txt`](v210-runtime-sha256.txt).

- Host binary SHA-256: `49c7645a4eabf09f22794d09a231054ac1da35d3a8074b1c2a10c41e26bd39fe`
- Fixture source SHA-256: `2805dea475c13f773bf81a93406a40e2bcbaa548424f82a73d2a4d41740ac2dd`
- Raw run log SHA-256: `6f624b4a5da50026aa065d9d7e0d8b37419af10dbadde39de0044abaaff0e4c7`

The raw output is [`v210-sched-head-q5k-20261017-once-01.log`](v210-sched-head-q5k-20261017-once-01.log). The exact runtime DSO values and paths are retained in the hash manifest. Preflight and postflight telemetry are retained in [`v210-gpu-preflight.txt`](v210-gpu-preflight.txt) and [`v210-gpu-postflight.txt`](v210-gpu-postflight.txt); postflight showed no compute PIDs, 0% utilization, 39–40°C, and 16,104–16,121 MiB free across all six GPUs.
