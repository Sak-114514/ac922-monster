# V273 Meta6 to same-device CUDA copy microfixture

This is a bounded correctness fixture for the V272 opt-in `GGML_META_PLAIN_DEVICE_COPY=1` path. It does not load `llama_model`, model device tables, full model weights, or the PLE table. It uses only a tiny synthetic IQ4_NL PLE payload with the V122 global type/row geometry and a six-way Meta row split.

The primary case runs twenty fresh graph frames through Meta6 PLE `GET_ROWS`, the production-shaped zero/repeat/add mirrored join, then a physical CUDA0 consumer. Each output is compared bit-for-bit with a CPU oracle; input/output canaries, unique graph UIDs, round-robin slot labels 0–3, and the actual `meta-plain-device-copy selected=1 ... host_staging=0` log are required. The scheduler does not expose its active slot index publicly, so the labels reflect its established `frame % n_copies` allocation contract rather than direct slot introspection. The run records correctness and path selection; it is not a throughput measurement.

The extra four-frame case represents the actual `qwen4exp` `ple_embd` reshape. In production, `build_inp_ple` reshapes the joined embedding before later PLE use, and `ggml_reshape_2d` preserves its source as `view_src`. The V272 bridge explicitly declines tensors with `src->view_src`; each view frame therefore makes an explicit `ggml_backend_tensor_copy_async` call with the actual view tensor, then compares the generic fallback destination byte-for-byte. The fixture records this as ineligible and does not report the view case as a direct-device-copy hit.

No engine source is modified by this fixture. The V274 hostlink script verifies the exact frozen build report SHA before compiling the fixture.

## Static contract check

```sh
./check-v273-contract.py
```

## Hostlink on AC922

```sh
./build-v273-host.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v274 v274-r1
```

This compiles the one fixture source and records the binary hash/`ldd`; it does not execute the fixture or initialize CUDA.

## Guarded GPU run

Run only after root grants `AC922_META_PLAIN_COPY_GPU_WINDOW_AUTHORIZED=1`, using the validated six physical V100 UUIDs in owner order and a fresh tag:

```sh
AC922_META_PLAIN_COPY_GPU_WINDOW_AUTHORIZED=1 \
CUDA_VISIBLE_DEVICES=<six validated V100 UUIDs in physical order> \
  ./run-v273-guarded.sh ./meta-plain-copy-v273-v274-r1 \
  ./backend-v274-r1 v273-v274-r1
```

The guard requires all six devices to be idle, below 58 C, below 5% utilization, with at least 512 MiB free and no compute PID. It polls a 180-second timeout and stores each run under a unique result tag. The runner sets `GGML_META_PLAIN_DEVICE_COPY=1` and rejects success unless it sees the V272 selected log, twenty exact primary frames, four exact reshape-view frames, and the final PASS marker.
