# Prefill next critical path: V24 read-only audit

Date: 2026-10-15. Scope: current private standard-opt source and the completed V24 semantic run only. No build, model load, request, or GPU computation was performed.

## Finding

V24 measured a real 8,274-token prompt at 7,835.19 ms (1,056 tok/s). Reaching 6,000 tok/s on the same token count requires about 1.38 seconds, or a 5.7x reduction. The current evidence does not justify changing another kernel yet. The strongest actionable gap is that the two likely overlap mechanisms are requested but not shown to be active: the 3+3 island timeline is unmeasured, and the CPU GET_ROWS/PLE overlap gate counted zero. A V24 PLE-prefetch initialization warning also says its row-cache path was not established.

## V24 facts

- Semantic-long V24 passed with prompt_n=8274, prompt_ms=7835.19, prompt rate=1056.005 tok/s, 19 generated tokens. This is the direct prompt metric; decode time is separate.
- The saved V24 frame log reaches active on 1,024-token shapes with n_kv=8192, graph reused, Meta replay ready, FIFO enabled. It also logs drains at memory-update and at an output-bearing 1,024-token boundary. These are policy/readiness records, not evidence that kernels from both islands overlapped in wall time.
- FIFO shutdown totals: frames=270, staged_inputs=7830, cross_copies=540, cpu_gathers=0, host_results=1620. The positive host_results/cross_copies counters do not mean the CPU GET_ROWS overlap path ran. No “CPU gather pipeline active” record appears.
- V24 emitted “ple-prefetch: row-cache identity or row geometry mismatch” three times despite LLAMA_PLE_PREFETCH and GGML_CPU_PLE_ROW_CACHE being requested. The warning is a combined guard and does not identify which subcondition failed. No current V24 Nsight timeline/profile is in the result directory.
- V24 has LLAMA_FULL_STEP_TIMING=0 and GGML_META_GPU_SPANS unset. The available spec-timing entries describe decode work only, not prompt evaluation or a full MTP host-tax decomposition.

## Source reading

1. Island overlap is possible but unproven for this run.
   - llama-context.cpp:4814-4844 calls ggml_backend_sched_graph_compute_async and returns without a final device synchronization.
   - ggml-backend.cpp:1900-1940 submits graph splits in order and inserts event waits at cross-backend dependencies. This permits asynchronous stream work but does not itself prove the 3+3 streams overlap.
   - ggml-backend-meta.cpp:3977-4021 switches multi-chunk child graphs to lockstep if replay readiness is missing; the ready path posts child backends on workers (4023-4041). Thus graph readiness determines whether a Meta3 backend uses the lockstep fallback.
   - The V24 frame “active” state is reached only after exact shape, graph reuse and replay-ready checks (llama-context.cpp:3365-3391; llama-prefill-frame-policy.h:78-123). It means the frame policy selected the path, not that the two physical island groups were concurrently busy.
   - memory_update drains pending frame work before mutating memory (llama-context.cpp:1540-1548). This is a request/batch boundary; it should not be confused with a synchronization at every 1K ubatch.

2. CPU PLE GET_ROWS acceleration is not demonstrated.
   - The target graph builds one PLE GET_ROWS over per_layer_token_embd.weight from CPU-generated row IDs (src/models/qwen4exp.cpp:3108-3134, 3256-3280).
   - ggml_sched_independent_cpu_gather only accepts a CPU split whose compute nodes are all host-weight GET_ROWS, whose indices root is a CPU input, and whose preceding split is Meta (ggml-backend.cpp:1860-1881, 1905-1914). V24’s cpu_gathers=0 means that exact bypass predicate never matched; it does not mean GET_ROWS was absent.
   - The PLE cache/prefetch guard checks table backing identity and row geometry in llama-ple-prefetch.cpp:248-261. A null cache makes the CPU fast helper return false and lets ordinary GET_ROWS proceed (ggml-cpu/ple-row-cache.h:36-47, 88-97). The warning does not distinguish null mapping from width/row-count failure, so add a one-time reason code before deciding how to fix the cache.
   - The best current PLE direction is to establish whether hybrid PLE backing can be read by the existing row cache and whether the CPU split is fenced behind Meta work. This is a measured-path diagnosis, not the old V11 prefault analysis.

3. MTP is not supported as the cause of the V24 prompt rate.
   - The normal prefill-frame candidate explicitly requires the default target graph/context, n_outputs=0, and inactive single/multi MTP chains (llama-context.cpp:2986-2994). MTP draft/spec timing is therefore not the prompt graph’s direct compute path.
   - V24’s spec-timing averages are decode-only and LLAMA_FULL_STEP_TIMING=0. There is no current evidence assigning any portion of the 7.835-second prompt to MTP host work. Measure it only if the requested work shifts to decode.

## Minimal next trace window

Do not change optimizer flags or rebuild for this trace. Use the same V24 binary, weights, UUID order, topology, quantization and launch environment. Add only LLAMA_FULL_STEP_TIMING=1 and GGML_META_GPU_SPANS=1. The former emits wall-event spans for graph_compute/synchronize; the latter samples CUDA event envelopes for all child devices in each Meta3 backend. The span implementation explicitly includes stream gaps, so it is a cross-check, not kernel-busy timing.

Nsight Systems 2023.4.4 is installed at /usr/local/cuda/bin/nsys. Launch the existing server in a named Nsight session, wait for its normal health check before starting collection, then capture exactly one uncached real 8K prompt request with n_predict=1 and stop immediately after its response. Use CUDA, NVTX and OS-runtime tracing, CUDA-graph node granularity for this one short diagnostic, process-tree CPU sampling, and SQLite export. Do not interpret traced latency as a performance result; it is for attribution only.

Extract from that one request:
- exact prompt_n/prompt_ms from the API timing response;
- per-device CUDA kernel/copy intervals and wall-union overlap for island A (UUIDs 0–2) and island B (UUIDs 3–5);
- Meta child stream envelopes from meta-gpu-span lines, including calibration error and sequence;
- CPU samples around ggml_compute_forward_get_rows / ggml_cpu_ple_cached_rows / row-cache read_many, plus host CUDA wait/launch time;
- frame state/drain records and FIFO counters, especially cpu_gathers and host_results;
- whether Meta reports replay-ready or lockstep/fallback behavior.

If this confirms cross-island overlap is absent or intermittent, the next structural seam is the precise graph-ready/lockstep or A→B event/copy boundary. If CPU samples and the zero counter point to serialized PLE GET_ROWS, first emit the exact independent-gather rejection reason and distinguish row-cache acquisition failure from split eligibility before modifying either path. Keep MTP out of this PP decision unless the trace shows a prompt-phase MTP call.

