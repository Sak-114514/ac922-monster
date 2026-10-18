# V192 persistent scheduler-sidecar / galloc isolation fixture

This is a no-model, ordinary-CUDA fixture prepared for a later bounded runtime window. It creates one CUDA backend object and one CPU backend object, then gives the same backend pointers to two separately constructed schedulers A and B. Both schedulers use `parallel=false` (`n_copies=1`), so each owns its own graph allocator.

## Workload and ownership

- A owns one fixed four-op F32 graph (`+1`, `*2`, `+3`, `*2`) and one persistent leaf/canary buffer. Small integer inputs make `4*x+10` an exact CPU oracle. The same graph metadata, scheduler, galloc output pointer, leaf buffer, input and canary remain alive across four A executions.
- B reuses its own scheduler across three graph shapes with 1, 2 and 3 rows. Each B graph is submitted and drained; before the next row count, B resets its scheduler and then releases that graph's metadata. B remains allocated while A runs between shape changes.
- After every B shape and after B scheduler destruction, the fixture checks A's output against the last CPU oracle, checks A's graph/context/buffer/tensor/data pointers, and reads its persistent canary.
- Both schedulers reference the same ordinary CUDA and CPU backend objects. There is no Meta backend, model, custom kernel, NVCC step, four-copy pipeline or shared graph allocator.

For graph-path evidence, the process requests `GGML_CUDA_GRAPH_PATH_LOG=1` before backend initialization, but the checker does not accept the environment request as proof. It requires actual `CUDA graph path active ... mode=capture` and `mode=replay` lines inside A's step markers. The fixed graph is called four times with changed input contents but stable tensor/graph addresses.

## CPU-only preparation checks

The source passed local C++17 `g++ -fsyntax-only`; the log checker has a CPU-only self-test. No remote build, GPU run, or model load was done for V192 preparation.

```bash
python3 fixtures/unroll-sidecar-arenas-v192-20261017/verify-v192.py --self-test
```

When the root assigns a frozen runtime and a short window, the one-TU host link is:

```bash
V192_RUNTIME_LIB=/absolute/path/to/frozen-runtime/lib \
  fixtures/unroll-sidecar-arenas-v192-20261017/build-host.sh
```

Capture merged runner stdout/stderr so the CUDA backend's actual path records are in the checked log:

```bash
/tmp/unroll-sidecar-arenas-v192 "$V192_RUNTIME_LIB" 2>&1 | tee /tmp/v192.log
python3 fixtures/unroll-sidecar-arenas-v192-20261017/verify-v192.py /tmp/v192.log
```

The checker fails if actual capture/replay logs are absent, if any exact output or canary check fails, if B does not exercise all three row counts, or if A's graph/allocator addresses change. Keep all graph metadata alive until its scheduler has been synchronized and reset.
