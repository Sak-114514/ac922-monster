# V220 CUDA graph binding-slot fixture

This derives the already exercised V152 ordinary-CUDA scheduler path: CPU-produced inputs are staged to CUDA0, the graph continues through CUDA1, and the scheduler uses four copies. One `graph_case` owns a reusable GGML metadata context; each frame rebuilds a fresh cgraph after `ggml_reset`, while the `mm0`, `mm1`, and graph descriptor addresses stay stable and the scheduler assigns a new graph UID. Each frame still owns its CPU inputs, which stay alive until the final drain. Tensor names remain constant because names are part of V218's exact structural signature. The small fixed-shape case submits 20 frames (five passes through the four-slot ring), poisons each frame's source tensors immediately after asynchronous submission, and checks each CUDA1 output byte-for-byte against the integer-valued CPU oracle with surrounding canaries.

The runner sets `GGML_CUDA_GRAPH_BINDING_KEYS=1` with cap 32, structural signatures and path logging, `GGML_CUDA_GRAPHS_FORCE=0`, and host timing counters. The source-contract check reads the actual V218 `ggml-cuda.cu` and `ggml-backend.cpp` implementations and verifies the default-off/exact-1 opt-in, cap policy, compute-only allocation after signature lookup misses, active/variant live-key protection, opaque context-owned keys, teardown ordering, and both scheduler and staging-ring rotation. `test-v220-key-recycling.py` compiles and executes the actual extracted pure-C++ allocator under ASan/UBSan on the host; it covers cap values 1 through 64, invalid caps, full-live fallback, active/retained key protection, retired-key reuse without pool growth, and RAII cleanup. These are CPU-only gates, not a CUDA core build. LeakSanitizer is unavailable on this Mac; teardown ownership is checked from the real destructor ordering and `unique_ptr` lifetime. The runtime checker requires stable shared metadata addresses with 20 unique scheduler UIDs, exact/canary outputs, and on both CUDA contexts a four-key pool, at least four capture calls, at least four replay calls, successful signature builds, and explicit `mode=capture`/`mode=replay` path records. Correct outputs without capture/replay do not pass this fixture.

## CPU-only source contract

```sh
python3 fixtures/graph-binding-slots-v220-20261017/test-v220-policy.py
python3 fixtures/graph-binding-slots-v220-20261017/test-v220-key-recycling.py
```

The first command is a source-contract scan. The second extracts the real helper verbatim into a tiny host-only executable and runs the policy cases. Neither builds the CUDA core or accesses a GPU.

## Host link and bounded runtime

The build requires explicit pinned hashes for the frozen runtime report and actual `ggml-cuda.cu`; it also verifies all four GGML DSOs resolve from the specified runtime. Example invocation after root freezes V218:

```sh
fixtures/graph-binding-slots-v220-20261017/build-v220-host.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/src \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v218 \
  /path/to/runtime-report.json REPORT_SHA256 GGML_CUDA_SOURCE_SHA256 \
  fixtures/graph-binding-slots-v220-20261017/results/build-v220
```

After the CPU gate and an explicit root GPU window, run once with a fresh log path. The authorization guard is `V220_GPU_WINDOW_AUTHORIZED=1`; the script has a 45-second hard timeout, does not retry, and never overwrites an old log.

```sh
CUDA_VISIBLE_DEVICES=0,1 V220_GPU_WINDOW_AUTHORIZED=1 \
  fixtures/graph-binding-slots-v220-20261017/run-v220-fixture.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v218 \
  fixtures/graph-binding-slots-v220-20261017/results/build-v220/graph-binding-slots-v220 \
  fixtures/graph-binding-slots-v220-20261017/results/v220-binding-slots.log
```

No V220 build or GPU run is claimed by preparing these files.
