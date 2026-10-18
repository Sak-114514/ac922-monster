# One-request Nsight analysis harness

Date: 2026-10-15. Scope: prepare a reusable analyzer for the next real 8K request trace. This work did not start a server, send an inference request, load a model, build code, or use a GPU.

## Artifact

- Script: fixtures/prefill-trace-20261015/analyze_prefill_trace.py
- Python standard library only: sqlite3, json, argparse, and collections.
- Input SQLite is opened read-only. Output defaults inside this fixture directory.
- The script accepts a request-window JSON with request_start_monotonic_ns and end_monotonic_ns. It maps the window to Nsight timestamps using TARGET_INFO_SESSION_START_TIME and a realtime/monotonic clock offset. For the strongest mapping, have the capture controller also write session_start_monotonic_ns into the window JSON; the script accepts that field or an explicit CLI override.
- Islands default to logical devices 0,1,2 and 3,4,5; override with --island-a/--island-b if the new runtime maps them differently.

## Capture requirement

Capture from server process start with Nsight Systems; do not begin late through cudaProfilerApi. Use --cuda-graph-trace=node so graph nodes are present for the captured request. For CPU stack attribution, add --sample=process-tree --backtrace=fp and keep --cpuctxsw=process-tree. Retain the existing CUDA/NVTX/OSRT trace settings. The profile is for attribution, so do not use its traced latency as a performance result.

## Run after the new SQLite is delivered

Run on AC922, or pass an AC922 capture-time clock anchor if analyzing off-host:

    python3 /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/prefill-trace-20261015/analyze_prefill_trace.py \
      --sqlite /path/to/new/fullrun.sqlite \
      --window /path/to/new/phase-window.json \
      --island-a 0,1,2 \
      --island-b 3,4,5

The script writes analysis.json and analysis.md beside itself by default. It also accepts --session-start-monotonic-ns or --realtime-minus-monotonic-ns if the capture controller provides the corresponding anchor.

## Metrics and accounting rules

- Per GPU and per island kernel interval unions; direct A/B kernel overlap.
- Per island any-activity union across kernels, memcpy, and memset; A/B activity overlap.
- All-GPU activity union and its complementary idle gaps, including counts above 1 ms and 10 ms and longest gap.
- CUDA Runtime API launch, graph-setup, synchronization/wait, and other calls: call count, summed call time, and interval union. Sums can overlap across host threads, so union is separate.
- CPU sample count, thread names, depth-zero stack hotspots, inclusive stack-frame hotspots, and common stack paths from COMPOSITE_EVENTS plus SAMPLING_CALLCHAINS.

Per-device/per-island intervals, cross-island overlap, host API categories, and CPU inclusive stack frames overlap. The report states each denominator and explicitly warns against adding overlapping percentages. Missing GPU activity, missing graph-node capture, missing CPU samples, mapping gaps, and Nsight diagnostics become warnings rather than idle-time claims.

## Verification performed

- Script syntax check and self-test passed for empty, adjacent, overlapping, clipped, disjoint, and intersection intervals.
- Read-only smoke against the existing V11 SQLite validated monotonic-window mapping, all-six-device kernel coverage, graph-node rows, CUDA launch/sync parsing, and the expected lack of CPU samples (the old profile used --sample=none). The old smoke output was removed; it is not a new trace result.

