# V270 dualhalf MoE host fixture

This directory prepares a bounded `GGML_OP_MUL_MAT_ID` comparison for the experimental `root_sm70_dualhalf_moe_launch` API in the parent fixture. It does not load a model. The seven weight formats and raw expert170/expert171 sources are the same exact-byte V236 operator samples; the runner verifies both sample manifests and file SHA-256 values before touching a GPU.

The native baseline runs actual `ggml_mul_mat_id` through the frozen ggml CUDA backend with grouped MMQ selected and its path marker required. The custom kernel receives CPU-prepared per-expert route tiles capped at J16. It is timed separately from CPU route preparation and compared against a CPU FP32 reference made by ggml dequantizing the two real expert row templates. Input activations use 32 deterministic distinct patterns, repeated by token, and the N=640/N=2560 pilot banks repeat the same 64 actual rows. This deliberately keeps the CPU reference and weight bank bounded; it is an operator fixture, not model inference.

Each case runs a single correctness sample for grouped MMQ and both custom accumulation modes (3-term and 4-term dual-half), compares all three against FP32, and requires each custom result to have lower RMS error than grouped MMQ before it enters any timing loop. A failure exits that case and the matrix stops without timing that case or starting later cases. The Python runner also requires the real `mmq_moe_grouped selected=1` dispatch log after the run; if it is absent, the case is marked invalid and no performance claim is made. After the correctness gate passes, it separately reports grouped-MMQ full-graph wall time, custom-kernel CUDA-event time, custom launch/sync wall time, and host route preparation time. These are not end-to-end LLM performance numbers.

## Files

- `dualhalf-fixture.cpp`: GGML grouped-MMQ baseline, CPU FP32 reference, custom API caller, timers, and 2-GiB/512-MiB runtime allocation gates.
- `route-plan.h`, `route-plan-cpu-test.cpp`: shared route packing and CPU-only coverage/duplicate/J16-bound checks.
- `dualhalf-host-fixture --cpu-reference-only`: runs raw-sample dequantization and the bounded FP32 reference without CUDA initialization; use it under host sanitizers when changing the CPU oracle.
- `build-hostlink-v270.sh`: separately gated SM70 CUDA object build plus host executable link; it never runs the executable.
- `run-v270-case.py`: manifest/hash checks, 58 C / 512 MiB / PID preflight and in-band guard, 120-second maximum per-case timeout, exact path/quality gates.
- `run-v270-matrix.py`: `small` is all seven formats at T=256/512 with uniform/allhot routes and N=64; `pilot` is N=640 for the five up formats and N=2560 for Q2_0 at T=256, both routes.

## Root-controlled commands

After the CUDA source and runtime are frozen and a host-link window is authorized:

```sh
V270_BUILD_AUTHORIZED=1 RUNTIME_ROOT=/path/to/frozen/runtime \
  ./fixtures/sm70-dualhalf-splitk-v270-20261017/host/build-hostlink-v270.sh
```

After root grants a GPU window, start with the seven-format small matrix:

```sh
V270_GPU_WINDOW_GRANTED=1 RUNTIME_ROOT=/path/to/frozen/runtime DEVICE=3 \
  python3 ./fixtures/sm70-dualhalf-splitk-v270-20261017/host/run-v270-matrix.py \
  --matrix small --run-tag unique-v270-small
```

The N=640 up and N=2560 Q2_0 pilot is separate and should run only after the small-N results justify it:

```sh
V270_GPU_WINDOW_GRANTED=1 RUNTIME_ROOT=/path/to/frozen/runtime DEVICE=3 \
  python3 ./fixtures/sm70-dualhalf-splitk-v270-20261017/host/run-v270-matrix.py \
  --matrix pilot --run-tag unique-v270-pilot
```

The local preparation agent must not run either host-link or GPU command. Root controls the runtime pin and the GPU release. Existing negative Volta tensor-core routes remain separate evidence and are not rewritten by this fixture.
