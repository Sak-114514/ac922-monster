# V15 crash-diagnostic P4 runner

V15 is a diagnostic-only copy of the frozen V14 dual-bank runner. It preserves V14's 16-case native semantic gate, 20 exact serial/P4 token comparisons, uncached 8K prefill prefix check, four-lane output gate, and performance arithmetic. It does not change model, KV, MTP, scheduling, CUPTI-window, or quality-gate behavior. V14 remains frozen; use this new directory for the SIGSEGV/SIGBUS capture.

The V14 runner contract and scenario definitions are retained in the copied V14 helper files in this directory. The runner still uses the frozen V50 runtime closure and the same R3 CUPTI collector. CUPTI `prefill1` is intended to trace the single-request uncached 8K/32-token probe before the four-lane scenario; a CUPTI-enabled process remains diagnostic-only for every scenario because activity instrumentation is enabled at process setup.

## Fatal signal capture

`ROOT_FATAL_BACKTRACE` is exact `0` or `1`, default `0`. With value `1`, before model/CUDA initialization the runner prewarms the host unwinder, installs SIGSEGV/SIGBUS handlers, and creates an alternate signal stack for the main thread. The persistent bank owner threads also create their own alternate stack. Signal handlers are reinstalled after CUPTI preparation/before backend initialization and at every scenario boundary, to recover if a library changed the process-wide disposition.

On SIGSEGV or SIGBUS, the best-effort handler writes a begin/end marker, signal number/name, `si_code`, fault address, OS thread ID, and `backtrace_symbols_fd()` stack to stderr, then calls `_exit(128 + signal)`. It never tries to resume. The reporter avoids C++ logging in signal context and uses `write()` for its fixed fields. The unwinder itself is not async-signal-safe by POSIX; prewarming reduces first-use work but cannot guarantee a complete trace if memory or stack state is badly corrupted. Treat missing frames as a diagnostic limit, not evidence that the fault lacked a stack.

`run-cpu-tests-v15.sh` launches intentional SIGSEGV, SIGBUS, and owner-thread SIGSEGV subprocesses. It verifies `128 + signo` exit status (139 for SIGSEGV; 135 on Linux or 138 on macOS for SIGBUS) and the marker, address, TID, and end lines. These tests do not initialize CUDA or load a model.

## Host build and run

Build only against the frozen V50 closure; linking does not launch a model:

```bash
ROOT_DYNAMIC_LIB_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v50/lib \
ROOT_BUILD_REPORT=/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V50-build.json \
  ./compile-runtime-v15.sh
```

The runner defaults to the existing 4-lane P4 workload and ordinary UBatch 1024. For the bounded single-request CUPTI prefill trace, set `ROOT_CUPTI_PHASE=prefill1`, `ROOT_FATAL_BACKTRACE=1`, `ROOT_TURBOPREFILL=0`, `ROOT_UBATCH=1024`, and `ROOT_WARMUP_FRAMES=4` in the external launch environment. The R3 collector must be prepared before CUDA initialization. Any such run is diagnostic-only and must not be used as a clean throughput result.

Example command shape (the caller supplies the model path and output directory):

```bash
ROOT_RUNTIME_REVISION=20261015T202046.715105Z \
ROOT_DYNAMIC_LIB_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v50/lib \
ROOT_BUILD_REPORT=/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V50-build.json \
ROOT_CUPTI_PHASE=prefill1 ROOT_FATAL_BACKTRACE=1 ROOT_TURBOPREFILL=0 \
ROOT_UBATCH=1024 ROOT_MTP_DEPTH=3 ROOT_WARMUP_FRAMES=4 \
  ./run-v15.sh ./build-runtime-v15/dual-context-v15 /path/to/IQ3_XXS-MTP-Q8.gguf /path/to/results/p4-v15
```

## Verification boundary

Run `./run-cpu-tests-v15.sh` for CPU-only gates. Run `./compile-runtime-v15.sh` for the host link and exact V50 `ldd` closure. Neither proves model correctness or performance. The real model run is still root-supervised; do not launch from this fixture independently.
