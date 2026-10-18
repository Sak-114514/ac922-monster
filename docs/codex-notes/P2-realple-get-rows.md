# P2 actual PLE GET_ROWS micro-measurement

Date: 2026-10-15 (AC922 local time)
Scope: CPU-only, one real 8K prompt slice, one warm repeat. No full model load, no GPU, no page-cache eviction/advice.

## Input and operator

- GGUF: `/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf` (78,615,622,080 bytes), opened read-only.
- The PLE table was mapped read-only by byte range only: 320,001,536 IQ4_NL rows, row width 90 bytes, 160 columns, data offset 47,039,862,208 (>2 GiB). The whole model was not read.
- Prompt IDs are the real `prompt_tokens.json` slice `[49152:57344]`, exactly 8,192 tokens. The two predecessor IDs `[49150:49152]` were included to match `llm_graph_input_ple::set_input` history semantics.
- Hash generation follows `qwen4exp.cpp::llm_graph_input_ple::set_input`: 3-gram, 16 heads, EOS reset, model multipliers/head vocabularies/offsets from the existing GGUF metadata. Raw GET_ROWS order is token-major/head-major. 131,072 row IDs; 58,323 unique, 72,749 duplicate IDs; logical packed payload 11,796,480 bytes; F32 output 83,886,080 bytes.
- This used the real GGUF parameter file `hardware-next-20261014/fixtures/ple-hash-params.txt` and did not invent indices. Token slice SHA256 `b82889ab07e75034ce541374fbc6831c4f9fbfc7960cb231d0b51e86c2fd9e61`; row-ID SHA256 `da2e43ea28f297dd228414be9a5ae38b794a9da86f9603d037542b2e1317cdf5`.

## Initial one-worker result (retained, not the 32-worker result)

The fixture executed actual `ggml_get_rows` through the GGML CPU backend scheduler against the mapped model tensor. Cache/page-batch switches were enabled, row-cache capacity configured to 262,144, NUMA bound to node 0. It performed exactly two graph computations: first lookup plus immediate warm repeat.

| Pass | scheduler graph wall | hits | misses | observed resident rows | direct rows | fallback | direct groups | direct bytes | direct 4 KiB pages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| First | 12,306.8 ms | 72,738 | 58,334 | 18 | 58,308 | 0 | 58,308 | 244,002,816 | 59,571 |
| Warm | 173.87 ms | 130,417 | 655 | 0 | 655 | 0 | 655 | 2,736,128 | 668 |

- The API reports counts and direct-I/O bytes, but has no per-`read_many` duration counter. `graph_compute_wall_ms` measures the full GET_ROWS graph compute (including dequantization/scheduler work), not an isolated `read_many` timer.
- Full warm output matched the first pass bitwise. Sixteen sampled rows (2,560 F32 values) matched direct IQ4_NL dequantization bitwise; max absolute error 0; no non-finite values.
- Resident/direct figures are observed during the run, not a manufactured cold-cache state. No `fadvise`, `madvise`, or `drop_caches` was used.

## Initial run thread-count caveat

The initial run set backend `n_threads=32` and environment `PLE_TEST_THREADS=32`, but the actual GET_ROWS task count is separately gated by `GGML_CPU_PLE_GET_ROWS_THREADS` in `ggml-cpu.c`. It defaulted to 1, and the cache log confirmed `workers=1/1`. This first result is therefore explicitly a one-worker operator result; it is not used as the 32-worker result below.

## Prefetch-ahead wiring

`llama-ple-prefetch.cpp` implements an opt-in bounded asynchronous row-cache warmer behind `LLAMA_PLE_PREFETCH=1` and `GGML_CPU_PLE_ROW_CACHE=1`. It has a one-active/one-replaceable-pending job queue and `llama_ple_prefetch_submit/tokens/cancel` entry points. A source-wide search of `.cpp`/`.h` under the current standard-opt source found only the declarations and definitions; no caller initializes the prefetcher or submits jobs from llama context/server request scheduling. Thus this code exists but is **not wired to the serving path** in this tree.

## Reproduction

From the fixture directory:

```sh
numactl --cpunodebind=0 --membind=0 env \
  PLE_TEST_THREADS=32 \
  GGML_CPU_PLE_GET_ROWS_THREADS=32 \
  GGML_CPU_PLE_ROW_CACHE=1 \
  GGML_CPU_PLE_PAGE_BATCH=1 \
  GGML_CPU_PLE_PAGE_BATCH_LOG_PERIOD=1 \
  GGML_CPU_PLE_ROW_CACHE_ENTRIES=262144 \
  ./P2-get-rows-real \
  /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf \
  47039862208 320001536 90 160 ./rows.bin
```

The captured run omitted `GGML_CPU_PLE_GET_ROWS_THREADS=32`, which explains the one-worker result above. The command is recorded for the corrected follow-up; no follow-up was run in this task.

## Artifacts and hashes

Own fixture directory: `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/P2-realple-get-rows/`

- `P2-get-rows-real.cpp`: `a6d383d40d986d0fd0bdacea4ed72e1c1179d8e92bbbd7204205466b018259ac`
- `generate-real-ple-rows.py`: `0be236423d2963845ab9cfe3965e68f8ed776de4798dfffa40f7bb578393f21e`
- `input-manifest.json`: `0d417d61f7513a54de6365419e0d9d33a80ed5a2f6ad4b7073f862a6d51ea249`
- `rows.bin`: `da2e43ea28f297dd228414be9a5ae38b794a9da86f9603d037542b2e1317cdf5`
- `P2-get-rows-real` binary: `539cdcc03e2cf9883b2ad64ee53904f3c5a3eb5eeb2a64bf0ea9d208b9689b0f`
- `P2-real-get-rows.log`: `edf49c1b41156ef79aed8a32d0f27e3f37c3b997c78d48111c022ec1eafd6728`
- Loaded `libggml-cpu.so.0.25.1`: `c1fd8d19e54df03e97f1c71c8c95565626ec73f44f7d4f3e334bfe3e74f464fc`
- Loaded `libggml-base.so.0.25.1`: `4bb34fce3820138aa47dbddc7ecc75485701e2dbdf4b53429ae38cb9a9552a66`

The process maps in the log show the two runtime libraries came from `standard-opt-20261014/build-reuse/bin`. `ldd` shows only CPU/base GGML libraries and no CUDA library. The graph-compute log is the full test record.


## Corrected 32-worker run (authorized follow-up)

The root authorized one actual 32-worker run plus one warm repeat, CPU-only. This is the requested 32-worker measurement; the previous one-worker result remains separately labeled above.

For a new request beginning at sequence position zero, the selected real token slice `[49152:57344]` was treated as an independent prompt. Missing predecessors at positions 0/1 were converted to EOS exactly as `llm_graph_input_ple::set_input` does for `LLAMA_TOKEN_NULL`; no tokens from before the selected slice were used. This changes 24 of the first 32 raw PLE row IDs (token 0 and 1), with all later IDs unchanged. The shared `ple-hash-rows-bench --counts-only` collector independently agreed: 131,072 raw rows, 58,339 unique, 72,733 duplicates.

| Pass | scheduler graph wall | hits | misses | observed resident rows | direct rows | fallback | direct groups | direct bytes | direct 4 KiB pages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| First, `workers=32/32` | 12,219.7 ms | 70,319 | 60,753 | 46 | 60,699 | 0 | 60,699 | 254,009,344 | 62,014 |
| Warm, `workers=32/32` | 98.3294 ms | 130,605 | 467 | 1 | 466 | 0 | 466 | 1,953,792 | 477 |

The warm output matched the first output bitwise. Sixteen sampled rows / 2,560 floats matched direct IQ4_NL dequantization bitwise; max absolute error was zero and there were no non-finite values. The operation is still measured as the whole CPU scheduler graph compute; the row-cache API does not expose an isolated `read_many` duration counter.

The command explicitly set `GGML_CPU_PLE_GET_ROWS_THREADS=32`; log lines report `workers=32/32` for both graph operations. No cache eviction/advice was used. The run was CPU-only and loaded only `libggml-cpu` and `libggml-base`, not the CUDA library.

Corrected command:

```sh
numactl --cpunodebind=0 --membind=0 env \
  PLE_TEST_THREADS=32 \
  GGML_CPU_PLE_GET_ROWS_THREADS=32 \
  GGML_CPU_PLE_ROW_CACHE=1 \
  GGML_CPU_PLE_PAGE_BATCH=1 \
  GGML_CPU_PLE_PAGE_BATCH_LOG_PERIOD=1 \
  GGML_CPU_PLE_ROW_CACHE_ENTRIES=262144 \
  ./P2-get-rows-real \
  /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf \
  47039862208 320001536 90 160 ./rows-fresh.bin
```

Artifacts for this run:

- `generate-fresh-ple-rows.py` produced `rows-fresh.bin`, `input-manifest-fresh.json`, and the independent-prompt token slice.
- `P2-real-get-rows-32worker.log` contains both graph timings, per-pass counters, library mappings, and exactness checks.
