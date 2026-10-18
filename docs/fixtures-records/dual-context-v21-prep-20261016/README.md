# V21: one-load, two-bank async Meta CPU-copy check

V21 is an isolated host harness derived from V19. It loads the integrated IQ3_XXS GGUF once, then constructs a fresh target/draft context family with `GGML_META_CPU_ASYNC_COPY=1` (the candidate path integrated in the V58 runtime). An optional `ROOT_META_CPU_COPY_VARIANTS=1,0` recreates and destroys the full context family before a second mode-0 control. Model weights remain loaded once and shared. The default run is candidate mode 1 only.

The fixed layout is two banks × two lanes, D3 MTP, Q8 KV, non-unified 131072 cells per bank = two independent 65536-cell streams. The two banks provide four 64K streams in aggregate; this is not a 256K per-request context. Target ubatch is 1024, MTP draft ubatch is 512, `GGML_META_ASYNC_INPUTS=0`, `GGML_META_NCCL_MIN_BYTES=0`, and `GGML_CUDA_NCCL_F32=1`. The QSA semantic route remains `LLAMA_QSA_INDEXER_IMPL=standard`.

The runner preserves the full V19 quality gates: native semantic checks before performance scenarios, four full 8K prompt flows, exact serial/P4 token comparisons, cold and warm 8K prefix checks, full F32 prompt-logit snapshots before and after MTP, MTP activation evidence, actual KV-shape checks, and output-completion checks. The mode checker requires per-Meta constructor policy markers, successful CPU→Meta transfer evidence, and teardown summaries for the selected mode. No result is considered a performance win unless all gates pass.

The two copy modes are applied only after `llama_model_load_from_file()` and before creating each complete context family. A second family is allowed only after the first family has synchronized work, joined owner threads, and destroyed all target/draft contexts. The runner never toggles the environment between target and draft context creation. `GGML_META_CPU_ASYNC_COPY` is sampled per Meta context at construction.

V21 uses the frozen V58 shared-library closure. `compile-runtime-v21.sh` links only the harness against that closure; it does not rebuild engine libraries and does not execute the model. The launcher requires a new output directory and preserves every prior run.

## Host-only preparation

```bash
./run-cpu-tests-v21.sh
ROOT_DYNAMIC_LIB_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v58/lib \
ROOT_BUILD_REPORT=/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V58-build.json \
./compile-runtime-v21.sh
```

The fixture's build/link and CPU tests are preparatory only. The model runner is intentionally not started by these commands.
