# V16 P4 crash and prefill-logit diagnostics

V16 is an isolated copy of V15. V15 remains frozen. V16 keeps the same 16-case native semantic gate, 20 exact serial/P4 token comparisons, uncached 8K prefill prefix gate, four-lane output gate, and performance arithmetic. It adds only optional failure deferral for CUPTI `stop()` and optional full-F32 target prefill-logit dumps. Enabling either diagnostic marks measured rates as non-clean.

The frozen runtime remains V50. No model/GPU run is part of the CPU tests or host link. The V15 SIGSEGV/SIGBUS reporter is reused unchanged: it defaults off, prewarms the unwinder before CUDA, reinstalls at scenario boundaries, writes signal/address/TID/backtrace markers, and exits without recovery.

## CUPTI stop failure behavior

`ROOT_CUPTI_DEFER_FAILURE` defaults to `0`, preserving strict V15 behavior. When explicitly set to `1`, only a nonzero result from `ac922_trace_stop()` is deferred: the runner records the stop status, collector error, and window markers, then continues through later scenarios so the loaded-model run can collect P4 results or a later fatal trace. The final result comparator still rejects the CUPTI trace and the launcher exits nonzero. Prepare/start failures and other runner errors remain strict; a deferred invalid trace is never a successful profiling result.

The collector path is configurable with `ROOT_CUPTI_COLLECTOR`. With CUPTI enabled, `ROOT_CUPTI_COLLECTOR_SHA256` is required and checked against the actual file before launch. The run manifest records that path, file size, and SHA-256 at start, verifies the selected file has not changed at finish, and records the trace hash if one exists. This permits a later R4 artifact without silently accepting a different collector. If no path is supplied, the bundled R3 file is the default; its digest still must be explicit.

## Optional prefill logits

`ROOT_PREFILL_LOGITS_DIR` defaults off (`off`, `0`, or unset). Set it to `1` to write under the run output directory, or pass an explicit directory. At the already-synchronized point where the runner scans the full final-prompt logits vector for finite values and min/max, V16 writes the vector as little-endian raw F32 with no additional GPU operation or synchronization.

Each `.ac922pflogits` file has a fixed magic/version prefix, a JSON header, and the raw logits. The header stores scenario and flow IDs, prompt source/hash, every prompt token ID and position, the final logit-row position/token, dtype, byte order, vector length, byte length, and CRC-32 of the F32 payload. The run manifest hashes every emitted dump at finish. Enabling dumps adds host file I/O and makes run performance diagnostic-only; the original exact-token quality gates remain active and unchanged.

Compare a serial and P4 dump for the same flow after checking CRC and matching prompt identity:

```bash
python3 compare_prefill_logits_v16.py serial.ac922pflogits p4.ac922pflogits
```

The report includes max/mean/RMS/relative absolute differences, argmax agreement, and both top-one margins. It is diagnostic only: it has no tolerance pass threshold and does not replace the runner's existing token/semantic gates.

## CPU checks and V50 host build

Run the CPU-only checks, including strict/deferred CUPTI-stop policy, V15 fatal signal subprocesses, the C++ logit writer/parser/CRC path, original semantic checkers, and all existing P4 fixture gates:

```bash
./run-cpu-tests-v16.sh
```

Build the host runner against the full frozen V50 shared-library closure without executing it:

```bash
ROOT_DYNAMIC_LIB_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v50/lib \
ROOT_BUILD_REPORT=/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V50-build.json \
  ./compile-runtime-v16.sh
```

## Next diagnostic invocation

For the requested follow-up, use `ROOT_CUPTI_PHASE=prefill1`, `ROOT_CUPTI_DEFER_FAILURE=1`, `ROOT_FATAL_BACKTRACE=1`, `ROOT_PREFILL_LOGITS_DIR=1`, `ROOT_TURBOPREFILL=0`, `ROOT_UBATCH=1024`, `ROOT_MTP_DEPTH=3`, and `ROOT_WARMUP_FRAMES=4`. Provide the current frozen collector path and SHA-256 explicitly. CUPTI or logit-dump instrumentation makes all performance numbers from the process diagnostic-only. The V16 launcher still returns nonzero if CUPTI validation fails, even when it was allowed to continue gathering model results.
