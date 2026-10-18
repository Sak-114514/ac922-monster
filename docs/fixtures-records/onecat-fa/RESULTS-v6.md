# Onecat FA fixture extension (V6 handoff)

Date: 2026-10-15. This was an operator-only GPU0 run using the already-patched Onecat object and frozen V5 ggml CUDA runtime. No core source or engine build was touched during this extension.

## New cases added

- QH24/KVH2, D256, T1, NKV2304, stream counts 1/4/8.
- Both Q8_0 and F16 K/V. K/V retain the same every-other-row gather view; Q remains the permuted query view; the F16 mask keeps a separate masked tail per stream.
- A/B/A CUDA-graph replay for S4→S8→S4 for both K/V types; A output remains byte-identical after B.
- 128 KiB post-allocation canary remains intact for every case.

## Comparison

Stock and Onecat ran in separate processes. All 14 output files pass the strict absolute comparison `--atol 0.004 --rtol 0` recorded in `compare-strict.txt`. The six new NKV2304 cases have:

| K/V | Streams | Max absolute error | RMS error |
|---|---:|---:|---:|
| Q8_0 | 1 | 0.000319 | 0.0000732 |
| Q8_0 | 4 | 0.000919 | 0.0002030 |
| Q8_0 | 8 | 0.000944 | 0.0002035 |
| F16 | 1 | 0.000247 | 0.0000686 |
| F16 | 4 | 0.000936 | 0.0002024 |
| F16 | 8 | 0.000936 | 0.0002019 |

Across all 14 cases, maximum absolute error is 0.003937 and maximum RMS error is 0.000924. The stricter compare output is under `results/onecat-fa-fixture-20261015-v6/compare-strict.txt`; hashes for raw outputs and the patched fixture object are in `sha256sums.txt`.

## Remaining limits

- No full-model request, server/MTP/multi-lane integration, semantic check, or performance measurement was run.
- No Compute Sanitizer was available. The allocator-adjacent canary passed.
- Broadcast-mask shapes were not exercised; Onecat rejects them and leaves them to stock FA.
