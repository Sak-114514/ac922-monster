# V152 pinned scheduler input staging fixture

This fixture exercises the V155 `GGML_SCHED_LAYER_PINNED_INPUTS=1` path with a real plain-CUDA scheduler ordered as CUDA0, CUDA1, CPU. The graph computes CPU `GET_ROWS` from CPU-resident embedding weights, transfers CPU inputs to CUDA0 for `MUL_MAT` and two `ADD`s, transfers the activation to CUDA1 for another `MUL_MAT` and `ADD`, and keeps output on CUDA1. Each output is copied asynchronously to a unique pinned sink and checked only after the final drain.

The micrograph constructs a fresh graph and fresh host input tensors for each submission, retaining all frame metadata contexts until the final drain. It runs two independent schedulers interleaved for eight submissions each (including ABA inputs and immediate source overwrite), then resets one scheduler and grows its graph inputs while retaining its staging ring. A final cap value of `1` verifies the production parser's invalid-cap scheduler fallback; per-backend `pinned-budget` rejection is tested without allocating huge buffers by the CPU planner mock.

## Preparation and launch

`test-v152-plan.py` is CPU-only. `build-v152-host.sh` links one fixture translation unit against the pinned runtime and records the binary, core-source, and four DSO hashes. The build waits for the V155 root build to finish. No GPU or remote build is part of the CPU planner step.

The GPU script refuses to run without `V152_GPU_WINDOW_AUTHORIZED=1`. `V152_STAGE_INPUTS=0` and `=1` select the same-binary fallback and opt-in diagnostics; the process prints a libc `backtrace_symbols_fd` trace on SIGSEGV. Both modes use two visible CUDA devices and the production 256 MiB cap. It never retries and refuses to overwrite a prior log.
