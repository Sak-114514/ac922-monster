# V250 MTP Q8_0 grouped MMQ fixture

Derived from the V231/V236 small-J `MUL_MAT_ID` harness, this fixture compares the regular Q8_0 route with the V249 default-off grouped Q8_0 path plus the existing per-expert small-J scheduler. It is a synthetic E=512 expert bank made by alternating exact raw row prefixes from real MTP experts 170 and 171. No quantization/repacking is performed, and no model is loaded.

## Raw MTP row prefixes

`extract-mtp-q8-rowprefixes-v250.py` uses the repository's `gguf-py` `GGUFReader` for tensor-directory metadata, cross-checks it against `metadata/mtp-expert-offset-ledger.json`, and checks the source expert bytes against the V122 MTP layer byte ledger. It then `pread`s only the first 64 rows for experts 170 and 171 from:

| Tensor | Type / shape | Row bytes | Expert stride | Bytes read per expert |
|---|---|---:|---:|---:|
| `blk.48.ffn_up_exps.weight` | Q8_0, K=2560, N=640, E=512 | 2,720 | 1,740,800 | 174,080 |
| `blk.48.ffn_down_exps.weight` | Q8_0, K=640, N=2560, E=512 | 680 | 1,740,800 | 43,520 |

The four slices total 435,200 bytes; the extractor refuses to exceed its default 1 MiB payload-read cap. It verifies current GGUF names, types, shapes, absolute offsets, row sizes, strides, file size, and V122 layer-48 expert byte total before reading any selected tensor bytes. The extracted slice hashes and byte offsets are saved in `samples/sample-manifest.json`.

## Operator matrix

Each format/shape is evaluated at T=256 and T=512, E=512, top-k=10, N=64; each run covers four full-output route patterns:

- `uniform`: 5 assignments/expert at T=256 and 10 at T=512.
- `skew_hot20`: 20 hot experts (>16 rows) plus 492 cold experts.
- `allhot10`: ten experts each receive T rows.
- `boundary16_17_20`: two experts at 16, one at 17, and one at 20; remaining experts are 49 at 4 plus 459 at 5 for T=256, or 29 at 9 plus 479 at 10 for T=512.

Baseline and candidate are separate fresh processes. Both use `GGML_CUDA_MOE_GROUPED=1`, disable the legacy adaptive-J path, and leave `GGML_CUDA_MOE_GROUPED_J` / `GGML_CUDA_MMQ_MOE_J` unset. Baseline sets `GGML_CUDA_MOE_GROUPED_Q8_0=0` and `GGML_CUDA_MOE_SMALL_J=0`; the runner requires grouped-Q8 to reject as `unsupported_quant_type`, then saves full F32 outputs. Candidate sets `GGML_CUDA_MOE_GROUPED_Q8_0=1`, `GGML_CUDA_MOE_SMALL_J=1`, and `GGML_CUDA_MOE_SMALL_J_MAX_TOKENS=512`. The runner requires both actual selected markers:

- `path=mmq_moe_q8_grouped ... requested=1 selected=1 reason=original_Q8_dot_compact_schedule`
- `path=mmq_moe_small_j ... requested=1 selected=1 reason=gpu_prefix_J16_cold_J32_hot_original_dot_kernels`

All candidate outputs are compared against the entire corresponding baseline F32 output. The fixture reports bit differences, max absolute/relative error, RMS, non-finite values, and five warm full-graph wall samples, including IDs helper, routing/classification, activation quantization, and grouped matmul. It sets no invented numeric tolerance or speed threshold.

## Commands

Raw row-prefix extraction runs on AC922 CPU and does not build, load a model, or use the GPU:

```sh
python3 fixtures/mtp-q8-grouped-v250-20261017/extract-mtp-q8-rowprefixes-v250.py \
  --out fixtures/mtp-q8-grouped-v250-20261017/samples
```

After the Q8-grouped runtime is frozen and a separate host-link window is authorized:

```sh
V250_BUILD_AUTHORIZED=1 RUNTIME_ROOT=/path/to/frozen-runtime \
  ./fixtures/mtp-q8-grouped-v250-20261017/build-hostlink-v250.sh
```

After root grants a GPU window, the guarded one-pass matrix is:

```sh
V250_GPU_WINDOW_GRANTED=1 DEVICE=3 RUN_TAG=unique-v250-tag \
  ./fixtures/mtp-q8-grouped-v250-20261017/run-v250-all-once.sh
```

The matrix contains two formats × two token counts × four route patterns × baseline/candidate. The wrappers stop on the first error and do not retry. It does not establish full-model behavior or performance.

## Current state

The row-prefix extractor has run on AC922 using the source GGUF and produced 435,200 payload bytes; the files and manifest are copied into `samples/`. The operator fixture and scripts are prepared only: no host-link, runtime load, GPU run, or model run has been performed.
