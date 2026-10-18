# V223 one32K target-prefill trace analysis

`analyze-v223.py` is a copy of the strict V214 R4 JSONL analyzer adapted to
require the V223 collector's fixed 4,194,304-event capacity. It validates the
R4 metadata and all zero-drop guards before it reads activity events. The input
is streamed line by line; keep the large trace file on the AC922 and write the
compact JSON summary there.

```sh
python3 analyze-v223.py /path/to/v223-target-prefill-r4.jsonl \
  --output /path/to/v223-target-prefill-summary.json
```

The CPU-only synthetic checks need no CUDA, model, or trace file:

```sh
python3 test_analyze_v223.py
```

## Reading the report

- `six_gpu_concurrency_ns` partitions the whole selected window into 0–6
  concurrently busy GPU bins. It sweeps each device's merged kernel intervals,
  so overlapping kernels on one GPU count once in that GPU's busy union.
- `runtime_api_host_time.any_thread_interval_union_ns` unions host runtime API
  intervals across threads. It is not the sum of per-call or per-thread API
  durations and does not prove a causal GPU wait.
- The API/GPU partition reports overlap, `api_only_ns`, GPU-kernel-only time,
  and `neither_ns`. `neither_ns` is time with no observed runtime API or kernel
  interval; it is not a CPU-idle measurement. `host_idle_ns` is therefore null.
- `graph_launch_host_api_calls` counts host graph-launch API intervals.
  `kernel_activity.graph_associated_kernel_event_count` counts device kernel
  records with a nonzero CUPTI graph ID or node ID. These counts describe
  different events and are not a graph replay count.
- `cpu_ple_attribution.status` is
  `not_inferable_without_explicit_markers`. The R4 schema has no CPU PLE marker
  event, so this trace cannot identify CPU PLE work.

## Comparison context

The intended primary comparison is the **actual V218 target-prefill run** with
`GGML_CUDA_GRAPH_BINDING_KEYS=1`, `GGML_CUDA_GRAPHS_FORCE=1`, F32 target
prefill, and `virtualTopK=1`. The V223 summary contains no V218 measurement;
report the matched V218 run separately when interpreting the trace.

The previous V219 UB256 reference is 1605.9 tokens/s whole-prefill and 1843.4
tokens/s target-prefill. Its native Paris canary passed, while eight long-output
cases were unscored. These are prior-run context, not trace-derived findings or
evidence that the V223 run is semantically complete.
