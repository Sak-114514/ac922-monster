# V215 F32 prefill indexer screen on V216

Runtime: frozen `runtime-perf-v216`  
Device: GPU3, Tesla V100-SXM2-16GB  
Scope: synthetic operator comparison; no model or GGUF load.

The test compares `llama_qsa_standard_headloop::build()` with `ggml_lightning_indexer()` on F32 pooled keys (`D=128`, `H=4`) and F32 queries, all-one weights, and zero F16 mask. Each arm gets two warmups and nine timed warm graph-compute-plus-synchronize samples. Score differences and selected top-k overlap remain `REPORT_ONLY`; no acceptance threshold was added.

## Results

| Case | Scores compared | Exact score match | Selected top-k position/set match | Standard median / workspace | Lightning median / workspace |
|---|---:|---:|---:|---:|---:|
| Q256 / K4096 | 1,048,576 | 100% (`bitdiff=0`) | 100% / 100% | 251.292 µs / 11,141,120 B | 231.914 µs / 8,916,992 B |
| Q512 / K8192 | 4,194,304 | 100% (`bitdiff=0`) | 100% / 100% | 797.406 µs / 39,059,456 B | 839.594 µs / 30,416,896 B |
| Q1024 / K8192 | 8,388,608 | 100% (`bitdiff=0`) | 100% / 100% | 1,475.312 µs / 73,924,608 B | 1,654.873 µs / 56,639,488 B |

All cases reported `max_abs=0`, `max_rel=0`, `rms_abs=0`, and zero non-finite scores. The stable four-cell-expanded selection matched in both order and set: 525,056/525,056 IDs for Q256, 1,050,112/1,050,112 for Q512, and 2,100,224/2,100,224 for Q1024.

Each CUDA run emitted `path=tile`, `type=f32`, and `F32 keys preserved=1 tile_keys=32 tile_tokens=8 shared_bytes=33152`. The Lightning graph used less workspace in all three cases. It was faster only for Q256/K4096; it was slower for Q512/K8192 and Q1024/K8192. These are warm isolated operator timings, not model-level performance claims.

## Setup note

The first Q256 attempt stopped before graph execution because the standard-headloop fixture arm allocated unused Lightning-only weights and mask, then tried to write those unconsumed tensors. The setup log is preserved as [`v215-v216-q256-k4096-setup-failure-preserved.log`](v215-v216-q256-k4096-setup-failure-preserved.log). The fixture now creates and initializes weights/mask only for the Lightning arm. The corrected Q256 run and the Q512/Q1024 runs each completed once; no CUDA exception occurred.

## Runtime artifacts

- Q256 raw log: [`v215-v216-q256-k4096-fixed-once-01.log`](v215-v216-q256-k4096-fixed-once-01.log)
- Q512 raw log: [`v215-v216-q512-k8192-once-01.log`](v215-v216-q512-k8192-once-01.log)
- Q1024 raw log: [`v215-v216-q1024-k8192-once-01.log`](v215-v216-q1024-k8192-once-01.log)
- Host link and V216 `ldd`: [`v215-hostlink-v216-setupfix.log`](v215-hostlink-v216-setupfix.log), [`v215-v216-ldd.txt`](v215-v216-ldd.txt)
- Binary, source, and loaded GGML library hashes: [`v215-v216-sha256.txt`](v215-v216-sha256.txt)
- GPU preflight/postflight: [`v215-v216-gpu-preflight.txt`](v215-v216-gpu-preflight.txt), [`v215-v216-gpu-postflight.txt`](v215-v216-gpu-postflight.txt)

Binary SHA-256: `a3ea5aae745edc7ea8a346b5a66f9e501f86f1dafc827bac1feaee9b3d51d6c8`  
Source SHA-256: `8b32d58c267003e54c54f12b381ea8400dd4afd88e6dedc9bc4e75083058f752`

Postflight showed no compute PIDs, 0% GPU utilization, 39–40°C, and 16,105–16,120 MiB free across all six cards.
