# V119 private-op dispatch gate

This fixture exercises the actual `GGML_OP_TOP_K` private subtype through GGML graph construction and backend dispatch. Root owns `src/ggml/src/ggml-qsa-topk-private.h` and all core handlers. This directory only owns the isolated harness and its bounded runner.

## Graph and reference

The private node is built with `ggml_qsa_virtual_topk_private::build(ctx, score_f32, cellmap_i32, mask_f16, width)`. Its inputs use `[B,Q,S]`, `[K,S]`, and `[K,Q,1,S]` layouts. The reference graph materializes `GET_ROWS(block_scores, cellmap)`, casts the F16 mask to F32, adds it, then calls stock `ggml_top_k`. Both graphs run on the same backend and their I32 outputs are compared exactly.

Shapes are tiny `B=257,K=1025,Q=4,S=2,W=16` and long `B=16384,K=65536,Q=512,S=1,W=2051`. Deterministic A/B/A inputs change scores, cell maps, and masks. The final physical cell maps to the partial last block in the tiny case; its mask stays `-inf` while its mapped block has a high poison score.

The support probe requires CPU to decline a correctly tagged private node, CUDA to accept it, and Meta3 to report all-child support. It also rejects a wrong magic through `valid()`, rejects a correctly tagged node with an extra source on CUDA/Meta3, and rejects width greater than K in the builder. The wrong-magic node is only checked by the private validator; generic backend fallback is intentionally not used as an invalid-private-tag test.

Meta3 uses mirrored tensors on three CUDA children. GGML's mirrored tensor readback copies rank 0; the fixture labels this explicitly and does not claim a per-rank output comparison. The Meta3 device support probe checks all children through the public Meta device API.

## Build and run later

The source and scripts are prepared only. No compile, CPU probe, CUDA execution, or Meta3 execution has occurred. Wait for root to freeze V120 and authorize one serial window before using either script.

The build uses ordinary `g++` with public GGML backend APIs; it does not use NVCC. It requires a new build tag and a frozen V120 runtime:

```sh
QSA_V119_BUILD_AUTHORIZED=1 \
QSA_V119_BUILD_TAG=approved-unique-tag \
QSA_V119_RUNTIME=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v120/lib \
bash fixtures/qsa-virtual-topk-dispatch-v119-20261017/build-v119-dispatch-fixture.sh
```

`builds/<tag>/` is exclusive: a repeated tag fails instead of overwriting its binary or log.

The serial runner performs the CPU support probe first, then CUDA device 0 (default physical GPU3 UUID), checks it is idle, then Meta3 on the three default cohort UUIDs. Each mode uses `GGML_CUDA_GRAPH_FRESH_EXEC=1`, `GGML_CUDA_GRAPH_STRUCTURAL=1`, path logging, and a timeout. If CUDA0 fails, Meta3 is not started; it does not retry.

```sh
QSA_V119_GPU_WINDOW_GRANTED=1 \
QSA_V119_RUN_TAG=approved-unique-run-tag \
QSA_V119_BINARY=/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/qsa-virtual-topk-dispatch-v119-20261017/builds/approved-unique-tag/qsa_virtual_topk_dispatch_fixture \
QSA_V119_RUNTIME=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v120/lib \
bash fixtures/qsa-virtual-topk-dispatch-v119-20261017/run-v119-gates.sh
```

The runner writes only a new `results/<run-tag>/` directory. It stores CPU, CUDA0, and Meta3 logs, runtime `ldd`, and binary provenance. A pass is operator/dispatch evidence only, not model quality or performance evidence.
