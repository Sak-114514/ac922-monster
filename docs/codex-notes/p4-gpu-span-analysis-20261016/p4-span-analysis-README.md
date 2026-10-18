# P4 CUDA event-envelope analysis

This is a CPU-only parser for P4 runtime logs and results.json. It analyzes only
the scenario's shared_generation_window interval stored as
shared_window_start_us/shared_window_end_us. It does not substitute the wider
parallel wave or whole-log averages.

Source format audit (V46): ggml-cuda.cu:3478-3482 prints begin, elapsed event duration, and the initial calibration bracket; ggml-cuda.cu:3487-3503 forms the origin calibration; ggml-backend-meta.cpp:3430-3434 maps trace pointers to context/instance/Meta/device; ggml-backend-meta.cpp:4663-4664 increments one span sequence per Meta context. The meta-gpu-span format records an approximate host-monotonic start,
same-device CUDA-event elapsed duration, sequence, child device, graph node
count, and initial calibration bracket. The envelope spans begin and end CUDA
events on that stream; it includes stream gaps and is not a measure of kernel
busy time.

meta-gpu-span-owner lines map trace pointers to Meta context pointer, instance
token, Meta device name, and child CUDA device. Bank IDs are not present in that
line format. When the runner emits bank-config lines, this tool associates
owner records with the next bank-config by creation order and labels that
mapping as inferred. If the records do not support that inference, the parser
keeps context instances unmapped and warns instead of guessing.

For each bank/island, the report shows original envelope durations for spans
intersecting the shared window and the portions clipped to that window. Cross
island overlap is computed as the intersection of unioned per-island event
envelopes, avoiding pairwise double counting. It includes a conservative
calibration bracket based on each event's initial calibration bracket.

P4 results use shared_window_start_us/end_us from the runner's intersection of all
lanes' post-parallel first-to-last token emission spans. Span approx_begin_us
is anchored from ggml_time_us; runner timestamps use std::chrono::steady_clock.
On Linux libstdc++ these are expected to share CLOCK_MONOTONIC, but the report
labels this as an assumption and warns if timestamps do not align plausibly.

CPU timing is reported only where frame-submit lines provide it. draft_us is
host elapsed from draft_begin to target_start and includes draft work plus
batch preparation. Optional guard hold/wait fields are lock metrics. With
serial_cpu_submit=false, zero decode_guard_hold_us is not a measurement of the
llama_decode CPU-call duration; full_target_us includes queued GPU execution
and synchronization, so it is not substituted for CPU submission time.

## CPU-only tests

Run:
  python3 -m unittest -v test_analyze_p4_gpu_spans.py

No CUDA runtime, model, or GPU is required.

## Analyze a run

Run:
  python3 analyze_p4_gpu_spans.py --log /path/to/runtime.log --results /path/to/results.json --label SCENARIO --json-out report.json

The label is required when results.json contains multiple scenarios. Use the final real4 scenario for any steady-state conclusion; semantic gates are parsed descriptively but are explicitly warned as non-steady-state. Incomplete result records are also warned. The P4 source prints bank-config after creating target and draft contexts (dual-context-v10.cpp:129-135); because span-owner records have no bank ID or target/draft pointer, bank association is an explicit creation-order inference and the report warns about it. Frame CPU timing fields come from dual-context-v10.cpp:348-366. The parser does not treat full_target_us as CPU submit time.

