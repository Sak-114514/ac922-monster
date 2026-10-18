# Nested Meta + EP6 staging fixture

**Status:** private fixture only. Nothing in this directory has been applied to the shared/core source tree. The topology-aware V54 fixture now passes both flat and nested local-partial ownership checks, then aborts in the nested final-reduction graph; see the preserved runs below. No model was loaded.

## Contents

- `nested-meta-ep6-r1.patch` — the original private R1 patch proposal. V54 has already integrated the nested Meta CPP changes and moved the offset helper to `ggml-meta-ep-coordinates-private.h`; **do not reapply R1 to V54**.
- `base/` and `candidate/` — exact pre-V54 source copies and patched private copies retained for R1 patch verification only.
- `v54/` — private copy of V54's `ggml-meta-ep-coordinates-private.h` and unchanged `ggml-ep-private.h`; the CPU test exercises the helper V54 actually uses.
- `tests/nested-ep-offset-compose.cpp` — CPU-only EP interval composition regression.
- `tests/meta-leaf-walk.h` and `tests/meta-leaf-walk-test.cpp` — shared topology-aware leaf selector and CPU mock-tree test. Flat Meta6 must descend once; nested Meta2×3 must descend twice.
- `tests/nested-meta-ep6-gpu-fixture.cpp` — small `Meta2{Meta3A,Meta3B}` + `MUL_MAT_ID` scaffold with Q8_0 weights, K=32, and eight rows per expert. It quantizes activations once to Q8_0, dequantizes those exact bytes to F32 because the current CUDA `MUL_MAT_ID` consumer requires F32 activations, then shares those same values between flat Meta6 and nested Meta2x3. It checks all experts, eight boundary canaries, foreign-zero slots, fenced replay and route ABA. The CPU dequantized reference is auxiliary; flat-vs-nested bitwise comparison is the exact gate.
- `run-cpu-tests.sh` — applies the R1 patch only to temporary fixture copies, runs CPU assertions and topology mock checks, syntax-checks the Q8 GPU scaffold and current V54 Meta TU; it does not invoke CUDA.
- `link-host-scaffold.sh` — host-links the scaffold against an explicitly selected existing GGML runtime. It does not run the binary or build core.

## Patch behavior

`GGML_META_NESTED_EXPERIMENTAL` defaults off. Nested Meta backend initialization fails with a descriptive message unless the process sets it to `1` **before any Meta backend is initialized**. Ordinary flat Meta3 keeps its existing collective setup and paths.

When an opted-in outer Meta has Meta children, the candidate skips outer CUDA-only span, NCCL and P2P registry discovery. The outer Meta retains the existing generic two-child graph fallback: child Meta3 graphs execute their own TP3 collectives, then the outer graph copies partial values between the two Meta children and adds them. This is a **generic fallback, not the fast CUDA AllReduce path**; no XBUS/P2P throughput claim is made. Inner Meta3 contexts can still use their normal direct-CUDA collective implementation. Null registry checks in the event fast-path fall back to synchronous event handling. The existing island-copy helper already checks for null CUDA registries before use.

EP parameters are composed from the input descriptor at each level. For parent offset 256 and global expert count 512, an inner local expert starting at 85 gets global offset 341; setting the child from the unchanged parent repeatedly returns 341 rather than accumulating to 426, 511, and so on. The existing flat `ggml_ep_private::set()` behavior is unchanged.

V54 already integrated the nested Meta changes; this fixture tests them without editing that source. It does **not** make a production DPA/KV owner scheduler. The GPU scaffold keeps one mirrored activation batch and exercises expert ownership/reduction only. It does not validate independent per-island request/KV/GDN state ownership, host PLE routing, model-wide behavior, or throughput.

## CPU verification

From the Mac fixture mirror:

```bash
bash monster-engine/standard-opt-20261014/fixtures/nested-meta-ep6-20261016/run-cpu-tests.sh
```

On AC922, point the test at the exact source and generated-header trees:

```bash
AC922_MONSTER_SRC=/home/debian/ac922-monster/experiments/standard-opt-20261014/src \
AC922_MONSTER_BUILD=/home/debian/ac922-monster/experiments/standard-opt-20261014/build-reuse \
bash /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nested-meta-ep6-20261016/run-cpu-tests.sh
```

The CPU regression requires the outer ranges `[0,256)` and `[256,512)`, then each inner `256 -> 85/85/86` split. Six expected starts are `[0,85,170,256,341,426]`; it checks every expert 0–511 has exactly one owner, boundary canaries, invalid ranges, and stable repeated materialization using V54's host-only EP coordinate helper.

The remote AC922 run of `run-cpu-tests.sh` passed: patch dry-run, EP interval composition, flat/nested leaf-depth mock, GPU scaffold syntax, staged Meta TU syntax, and V54 Meta TU syntax. Earlier `cpu-staged-r2.log`/`r3.log` preserve two host-source errors (vexing-parse route vector, then include-order mismatch) before the final fix. The local Mac mirror lacks `ggml-quants.h` in its partial source checkout, so the complete runner was verified on AC922 instead. `results/hostlink-staged-r2.log` records the prior V54 API host-link; a fresh V54 host-link also passed before the second GPU run.

The historical R1 patch is applied only to fixture copies for validation. To verify its applicability without changing shared source, the fixture runner copies `base/` into a temporary tree, applies `nested-meta-ep6-r1.patch` there, and compares the result with `candidate/`. The current V54 source is separately syntax-checked in place, read-only.

## GPU window results

The scaffold requires all six GPUs and explicit environment opt-in:

```bash
GGML_META_NESTED_EXPERIMENTAL=1 LLAMA_EXPERT_PARALLEL=1 ./nested-meta-ep6-gpu-fixture
```

**Run r1:** The original two-level leaf walk incorrectly treated flat Meta6 as nested Meta2×3 and called `ggml_backend_meta_tensor_get_simple()` twice. It asserted at `ggml-backend-meta.cpp:1155` before any ownership checks. This was a fixture inspection bug, not an engine verdict. The unmodified log is `results/nested-q8-flat6-vs-2x3-v54-r1-20261016T0559Z.log`.

**Run r2:** A shared topology-aware walker fixed the fixture: flat Meta6 descends once; nested Meta2×3 descends twice. With the same V54 runtime, six-card UUID order, `GGML_CUDA_NCCL_F32=1`, and default P2P AR fixed grid (`blocks=8`, `threads=512`), both `flat6-local` and `nested2x3-local` passed `foreign_zero=1`, `owner_count=1`, all 4,104 token rows, and eight canaries. The run then aborted during the nested final-reduction graph at `ggml-backend-meta.cpp:3036`, through `ggml_backend_tensor_get` / `ggml_backend_tensor_copy_async`. Flat/nested final-output comparison and the later route-replay passes were not reached. Preserve this failure without retry; it is unresolved whether the failure is in the V54 nested final-graph path or the toy graph contract. Raw log: `results/nested-q8-flat6-vs-2x3-v54-r2-20261016T0622Z.log`.

The r2 fixture host-link used only existing V54 runtime libraries; all four linked GGML libraries resolved from `runtime-perf-v54/lib`. Binary SHA-256: `1c98f09cf8ff9e06eee5459995599868186f65af9c30b8e8621dba0125522981`.

## Source provenance

The `base/` files were copied read-only from `/home/debian/ac922-monster/experiments/standard-opt-20261014/src/ggml/src/` on 2026-10-16:

- `ggml-backend-meta.cpp`: `a7223ac569533d9d59ef6b15ec5fc64b77162ed07d18839e0b43b9e791f90af1`
- `ggml-ep-private.h`: `477d832b545398de8cf765cfda0deabafefef3ebfec393fd983f5df46391384a`

Before r2, `nvidia-smi` showed no GPU processes, 0% utilization, 38–39°C, and 0–1 MiB reported per card. Postflight found no running GPU processes, 0% utilization, 39–40°C, and 1–2 MiB reported per card (driver residual). No model was loaded and no full-model process was touched.
