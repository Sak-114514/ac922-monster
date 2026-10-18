# V178 plain-CUDA Lightning Indexer numerical fixture

This fixture isolates `ggml_lightning_indexer()` on a single ordinary backend.
It is intended to answer whether the fused score op can be considered for the
plain-CUDA route after the separate Meta3 failure. It does not load a model,
measure model speed, exercise Meta dispatch, or prove semantic parity.

## Arms and represented inputs

- `stock`: the existing Qwen4Exp standard graph shape: F32 `mul_mat`, per-head
  ReLU, then ordered head sum.
- `headloop`: `llama_qsa_standard_headloop::build()` from the existing fixture.
- `lightning`: the actual public `ggml_lightning_indexer()` op with F32 Q/K,
  F32 all-one head weights, and an F16 zero mask, matching its Qwen4Exp call
  site. Visibility/block bias is added only in CPU selected-cell analysis, as
  production adds it after the fused score op.

K source rows are generated as cell keys, quantized to Q8_0 with GGML's public
quantizer, decoded back to F32, pooled four cells per block, then RMS-normalized.
This represents the Q8_0-decode/F32 input boundary to the fused op; the fixture
does not reproduce model projections, RoPE, a live KV cache, or fragmented cell
ordering. Q has shape F32 `[128, 4, Q, 1]`; pooled K has shape F32
`[128, K, 1]` and is viewed by Lightning Indexer as `[128, 1, K, 1]`.

Each arm is evaluated in a separate GGML graph and scores are compared without
requiring bit equality. F64 gold is computed for the first, middle, and last
query (deduplicated) to bound CPU work on the large case. The report includes
max absolute/relative/RMS deltas, counts over two absolute thresholds, selected
cell ID differences versus `stock`, and the margin between the last selected
block and the best unselected block. The model-shaped selected width is
`min(4*K, 2048+4-1)`. Since a block score is copied to four member cells, a
width not divisible by four has a structural within-block tie at the cell
cutoff; this is reported separately from the inter-block cutoff margin.

`REPORT_ONLY` means score deltas and changed selected IDs are diagnostic facts,
not an automatic quality verdict. Non-finite values fail the fixture. Any exact
ID change needs a model-level semantic comparison before changing the clean
`LLAMA_QSA_INDEXER_IMPL=standard` route.

## Cases

| Case | K pooled blocks | Q queries | Intended use |
|---|---:|---:|---|
| `cpu-smoke` | 128 | 1 | small CPU/API sanity case |
| `decode-q1` | 8192 | 1 | decode-shaped CUDA case |
| `decode-q5` | 8192 | 5 | short verification-vector CUDA case |
| `decode-q7` | 8192 | 7 | short verification-vector CUDA case |
| `prefill-q256` | 8192 | 256 | first prefill CUDA case |
| `prefill-q1024` | 65536 | 1024 | large prefill CUDA case; run alone |

The fixture holds the stock score output while comparing one candidate arm at a
time. It does not build the full expanded cell-score/top-k CUDA graph; selected
IDs are produced by a stable CPU block-score selector after each arm readback,
so the large case avoids a second 4x expanded score matrix. This checks
numerical ranking sensitivity at the operator boundary, not production
`ggml_top_k` backend behavior.

## Prepared commands

The host-link script links only GGML backend libraries from an existing runtime;
it does not link `libllama`, build the engine, or invoke NVCC. It is disabled
unless `V178_BUILD_AUTHORIZED=1` is set. No build or GPU run was performed while
preparing this fixture.

```sh
V178_BUILD_AUTHORIZED=1 RUNTIME_ROOT=/path/to/runtime-perf-v178 ./build-hostlink-v178.sh
RUNTIME_ROOT=/path/to/runtime-perf-v178 ./run-cpu-smoke.sh
PLAIN_CUDA_INDEXER_GPU_WINDOW_GRANTED=1 DEVICE=0 CASE=decode-q1 ./run-cuda-once.sh
```

Run the two prefill cases separately under a fresh bounded GPU window. The
runner checks temperature, free memory, and active compute apps. It does not
authorize itself: the parent must grant the explicit window variable.

The binary itself also requires `PLAIN_CUDA_INDEXER_GPU_WINDOW_GRANTED=1` for
`--cuda`; CPU execution accepts only the bounded `cpu-smoke` case. The Python checker validates report completeness, finite numerical fields,
expected arm/case coverage, and optionally exact IDs. It does not reject score
or selected-ID differences by default:

```sh
./check-report.py results/<log> --expect-case prefill-q256
./check-report.py results/<log> --expect-case prefill-q256 --require-exact-ids
```
