# V20: one-bank, four-lane P4 candidate

V20 is a new isolated harness for testing four concurrent sequences in one target context and one MTP draft context. It leaves the V19 harness and its results untouched. It preserves the original semantic cases, all 20 serial-versus-parallel exact-token pairs, both uncached 8K prefix probes, the full 8K prefill path, and the 4×256-token common-generation window.

## Fixed topology and KV capacity

The only accepted layout is `ROOT_KV_UNIFIED=0`, `ROOT_BANK_CTX=262144`, `ROOT_BANK_LANES=4`: one bank with four independent 65536-cell KV streams. The total 262144 cells is aggregate capacity; each request is limited to 65536 tokens. The runner independently checks actual target and MTP context sizes, sequence count, and KV stream count through public context queries after initialization. It does not claim 262144 tokens per request.

Each run loads one model and creates one target/draft context pair. The serial semantic and two-request references use the same four-lane-capable context with only their two active lanes; the P4 case places all four requests on bank 0. Four lanes do not mean four independent contexts. The saved output records each flow's bank mapping and requires all four P4 flows on bank 0.

## MTP shape and row budgets

The default MTP depth is D2. D4 can be selected explicitly after its separate qualification; D3 is rejected for four lanes by the engine lane-shape policy. Six fixed-prefix rows per lane means 24 padded input rows in the MTP draft batch. That is distinct from target verification capacity:

- D2: at most `4 × (2 + 1) = 12` target verify rows.
- D4: at most `4 × (4 + 1) = 20` target verify rows.
- MTP fixed-prefix input: `4 × 6 = 24` rows for either depth.

`n_batch` remains 8192, enough for each complete 8K prompt. `ROOT_UBATCH` accepts 512, 1024, or 2048; the runtime still uses its lazy workspace safety selector. The comparator records every `selected_ubatch` line from the actual `lazy-prefill-workspace` runtime logs into `results.json`, including when the selected tile is smaller than requested. The run requires both `LLAMA_PREFILL_LAZY_RESERVE=1` and `LLAMA_PREFILL_SEPARATE_STREAM_RESERVE=1`; no clamped tile is silently reported as the requested tile.

## Unchanged quality and runtime checks

- The frozen 16-case native semantic checker remains authoritative and never executes generated Python.
- All 20 prompt/output pairs must remain exact against the serial references; the 256-token parallel flows must finish in full.
- The cold and same-process warm-graph 8K probes retain their original 32-token prefix checks.
- D2/D4 unrolled activation, fixed-prefix padding, EP route, Meta context-view and GPU-span evidence, MTP fallback counts, actual KV shape, and the common generation window remain checked.
- Performance is diagnostic-only if CUPTI or prefill-logit dumps are enabled. A selected lower ubatch is logged as such, not described as a 2048-row run.

## Build and run

`compile-runtime-v20.sh` links a host runner against a frozen runtime closure (defaults to V56, but `ROOT_DYNAMIC_LIB_DIR` and `ROOT_BUILD_REPORT` may select a newer compatible frozen build). Linking does not load the model or touch GPU. `run-v20.sh` accepts only the V20 KV shape and MTP depth 2 or 4. Root supervises model runs and applies the existing GPU-memory and temperature guards.
