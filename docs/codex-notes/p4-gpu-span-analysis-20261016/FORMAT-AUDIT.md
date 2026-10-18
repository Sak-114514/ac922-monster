# P4 span-format audit and run status

Date: 2026-10-16
Scope: read-only source/runtime format inspection; parser and CPU synthetic tests only. No GPU task was started.

## V46 Meta owner and CUDA event lines

The runtime emits owner records from ggml-backend-meta.cpp around 3432:

    meta-gpu-span-owner context=0x... instance=123 meta=Meta(CUDA0,CUDA1,CUDA2) device=CUDA0 trace=0x...

This is one owner record per child backend. It provides the Meta context pointer, instance token, island name, child backend name, and trace pointer. It does not include P4 bank ID or the outer llama_context pointer.

The CUDA span implementation in ggml-cuda.cu around 3478-3482 emits:

    meta-gpu-span device=0 trace=0x... seq=12 nodes=45 approx_begin_us=104123456789 duration_us=8100 initial_calibration_bracket_us=8

approx_begin_us is host-anchored when the per-device CUDA origin event is calibrated at trace initialization (around lines 3487-3503). duration_us is cudaEventElapsedTime(begin,end), so it is a same-device CUDA event interval. The implementation starts/ends events on the child CUDA stream and samples sequences according to GGML_META_GPU_SPANS_PERIOD (around lines 3507-3537). The span is an event envelope and includes stream gaps; it is not kernel busy time. A separate teardown line reports dropped=N for the trace lifetime and has no timestamp, so dropped counts cannot be assigned to the shared window.

V46 owner trace IDs are process-local pointers; the instance token distinguishes Meta contexts. The parser associates each span with the most recent preceding owner line for that trace. If a trace pointer is reused in one log, it warns and preserves the log-order association for human review.

## Bank mapping

The runner prints bank-config id=... target=... draft=... after constructing that bank's target and draft contexts (dual-context-v10.cpp around lines 129-135). Span-owner lines have no target/draft pointer and no bank ID. The parser therefore maps an owner record to the next bank-config in creation order and labels this inferred_owner_before_bank_config; it is not a direct pointer join. If a trace owner has no following bank-config, the bank stays unmapped and the report retains per-context results.

Island names are resolved from the Meta member names: Meta(CUDA0,CUDA1,CUDA2) is island0 and Meta(CUDA3,CUDA4,CUDA5) is island1. The numeric span device is cross-checked against that member list.

## Shared-generation window and host timings

The correct window is results.json's shared_window_start_us and shared_window_end_us, written by the runner around dual-context-v10.cpp lines 784-787. This is the intersection of each lane's first-to-last emission interval after the parallel wave starts. The parser requires one explicit scenario label and clips/filters spans to this window. It does not use parallel_window_start_us/end_us, global log averages, or token/s as step timing.

The runner now_us() uses std::chrono::steady_clock (around line 34); ggml_time_us() uses clock_gettime(CLOCK_MONOTONIC) on Linux (ggml.c around lines 569-573). Linux libstdc++ is expected to back steady_clock with CLOCK_MONOTONIC, but that mapping is a source-based assumption. The report checks whether span times intersect the selected shared window and warns if they appear far apart. Each span retains initial_calibration_bracket_us; cross-island overlap includes a conservative lower/upper bracket using those per-trace calibration bounds.

frame-submit logs include draft_us, optional draft/decode guard waits and holds, and target_start_us (dual-context-v10.cpp around lines 348-366). draft_us is host elapsed from draft start to target submission and includes draft work plus batch preparation. The optional decode guard hold measures only the locked llama_decode host section when that guard is enabled. With serial_cpu_submit=false, zero hold is not a CPU-call duration. The parser does not substitute full_target_us, which includes queued GPU work and synchronization.

## Parser test status and actual P4 run

The parser lives in this directory as analyze_p4_gpu_spans.py; its CPU-only synthetic suite is test_analyze_p4_gpu_spans.py. It verifies window filtering, context/bank/island association, trace reuse warnings, overlap unioning and calibration brackets, and CPU timing scope.

The actual run path supplied by root is:

    /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v10-prep-20261016/results/p4-d3-20261015T192743Z/

At the last read, run-manifest confirmed V47 and span period 1, but results.json contained only two serial-sem-* records with no shared window yet and runtime.log had no meta-gpu-span lines yet. No actual analysis was run on that partial result. The analysis should wait for the final full real4 scenario, select it by label, and keep semantic warmup records descriptive only.

