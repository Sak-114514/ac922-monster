# V25 run status — hierarchical EP6, 16K per request

**State: failed; supervisor stopped; no automatic relaunch.** This note records the only V25 model attempt and the preflight wrapper failure. It does not certify semantics, token parity, or performance.

## Runtime provenance

- Runtime closure: `/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v63`; build ID `20261016T013345.175233Z`; V63 build report SHA-256 `1dc9d03042c477ccdedef51f9abf0c9a53f7b148ba4be923a409ddcd24817c58`.
- Runner: copied byte-for-byte from the prior V24 PPC64LE host-linked executable; no runner rebuild. SHA-256 `8f25242b2c8bb8fe09eba2002386d5d98f5b8de08886e9e491a6499b76a5c44d`; ELF BuildID `8560fce4abedc573a22840fd81d9dfdf9dc34546`.
- V63 `ldd` check resolved all seven app libraries from `runtime-perf-v63/lib`. CUDA/NCCL runtime dependencies resolve from the installed system libraries. Evidence: `v63-reused-runner.{file,ldd,sha256,status}.txt`.
- Root-provided V63 nested PLE weight-view qualification: pass, log SHA-256 `f1116950904c3ba2d6481653b7fe8bf53df03c12cbe80b345694784b267aa59e`.
- Configuration retained: IQ3_XXS, all 512 experts, Q8_0 K/V, hierarchical EP6 (two 3-GPU islands), PLE 8 GiB over six GPU NUMA nodes, two banks × two lanes, D3 MTP, 16,384 context per request, target ubatch 1024, MTP ubatch 512. No reserve was lowered.
- Timing instrumentation: `GGML_META_TIMING=1`, `LLAMA_STEP_TIMING=1`, period 32. CUPTI and `LD_PRELOAD` were off. No synchronization shim was used.

## Attempts

1. `root-p4-v25-shortctx16k-hierarchical-ep6-d3-20261016t014451z`: wrapper exited 126 before model start because the newly copied `run-v25-shortctx.sh` lacked its executable bit. No model load or GPU work occurred. The launcher mode was corrected to 0755 before the next attempt. The original failed tag remains preserved.
2. `root-p4-v25-shortctx16k-hierarchical-ep6-d3-20261016t014524z`: model loaded once and the first diagnostic semantic scenario began, but the process exited 139 (SIGSEGV) during CUDA graph structural-signature construction. No completed native semantic, preclean full-F32, exact-token, or clean performance gate exists.

   Fatal record: `SIGSEGV`, `si_code=1`, `si_addr=0x57800`, TID 100419. Symbolized V63 CUDA stack begins `ggml_cuda_graph_tensor_input` → `ggml_cuda_graph_make_structural_signature` → `ggml_cuda_graph_get_key` → `ggml_backend_cuda_graph_compute`, then Meta scheduler / `llama_context::process_ubatch` / `llama_decode` / `bank::prefill_full_logical_prompts`.

   The PLE weight-view abort seen in V24 did not recur. This fault is in the CUDA graph structural-signature/input path; it is not evidence of a model quality result.

## Timing and graph evidence from the failed run

- One lane record exists for `capital-en`: `target_decode_sync_us=1921148`, `mtp_process_us=187116`, `diagnostic_dump_scan_us=81313`, `lane_prefill_total_us=2189584`. This is a diagnostic prefill lane record, not a throughput benchmark.
- No `step-timing` record reached the period-32 threshold. No `meta-timing` aggregate was emitted. V63 source at `ggml-backend-meta.cpp` increments `needs_rebuild`/rebuild-time accounting per completed Meta graph dispatch and prints only every 200 calls; the run ended before a completed summary interval. Therefore natural graph rebuild count and rebuild wall-time cannot be quantified from this run.
- Observed scheduler reserve shapes for target: PP/TG graph nodes `7375/7075` (later frame reserve `7379/7073`), two splits, six input objects, 18/15 input tensors, Meta compute buffer 171.78 MiB and CUDA_Host buffer 16.26 MiB. The corresponding reserve calls took 144.14/143.74 ms at context setup and 135.24 ms for the first scenario frame.
- Observed draft reserve: 160 graph nodes, two splits, four input objects, nine input tensors, Meta compute buffer 185.28 MiB and CUDA_Host buffer 41.28 MiB. Reserve calls took 64.22/65.11 ms at context setup and 78.81 ms for the first scenario frame. These are scheduler reserve timings, not per-frame graph rebuild costs.
- Runtime contained 333 physical-device `meta-gpu-span` records: 56 rows each for devices 0–2 and 55 rows each for devices 3–5; graph node counts were 40, 132, 141, 180, 193, and 232. Logged span durations ranged 3,828–221,754 µs (median 6,875 µs, p95 32,088 µs). The span logger explicitly includes stream gaps, so these are not kernel-busy measurements or graph-rebuild timings.
- Source condition: `needs_rebuild` is true when `cgraph->uid == 0` or differs from `backend_ctx->uid` (`ggml-backend-meta.cpp` around line 4513). The run did not produce the 200-call Meta timing summary needed to count or price natural inner-graph rebuilds.

## Supervisor telemetry

The supervisor recorded 15 samples from `2026-10-16T01:45:25Z` to `01:48:39Z`.

- GPU free memory ranged 3,888–16,128 MiB/card; GPU temperatures 37–40°C; sampled utilization peaked at 3%. After exit, every GPU returned to 16,125–16,126 MiB free and 0% utilization.
- BMC maximum temperature ranged 39–40°C; ambient 30.813–30.900°C. All three primary fans remained at target 3000 RPM: fan0 2843–2856, fan2 2853–2893, fan3 2845–2858 RPM. No alarm/fault samples; BMC masks remained valid.
- DDR node0+8 `MemFree` ranged 150.5–23,356.1 MiB and unit RSS peaked at 33,975 MiB. These are recorded telemetry only; no DDR stop threshold was applied.

Raw run artifacts are under the remote result directory for tag `...014524z`, including `runtime.log`, `gpu-telemetry.jsonl`, `unit-journal.txt`, and the start-time run manifests. The matching local monitor telemetry and unit journal are under `/Users/lyx/Desktop/ssh/monster-engine/standard-opt-20261014/notes/`.
