# V26 run status — hierarchical EP6, 16K per request

**State: failed with SIGSEGV; supervisor stopped; no auto-relaunch.** This is the single V26 full-model attempt; it produced no completed semantic or performance result.

## Provenance and launch configuration

- Runtime closure: `/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v64`; build ID `20261016T015610.271139Z`; build report SHA-256 `8818d4831179db364702a1720c849dfb04b836d6c37e7ae3e6671d9ed5aefc1e`.
- Runner: byte-for-byte reused V24/V25 PPC64LE binary; no runner or engine rebuild. SHA-256 `8f25242b2c8bb8fe09eba2002386d5d98f5b8de08886e9e491a6499b76a5c44d`; ELF BuildID `8560fce4abedc573a22840fd81d9dfdf9dc34546`. V64 `ldd` resolves all seven application libraries from `runtime-perf-v64/lib`; evidence is in `v64-reused-runner.{file,ldd,sha256,status}.txt`.
- Root reported the V64 opaque-extra qualification PASS (external-suffix sentinel/direct-node regression; fail-closed opaque graph path with raw-UID graph behavior and ABA/canary checks). That qualification did not prevent this full-model failure.
- Config: IQ3_XXS, all 512 experts, Q8_0 target/draft KV, hierarchical EP6, PLE 8 GiB over six GPU NUMA nodes, two banks × two lanes, D3 MTP, 16,384 context per request, target ubatch 1024, MTP ubatch 512. Existing native/F32/exact/diagnostic/clean and thermal/fan gates were retained; reserves were unchanged.
- `GGML_CUDA_GRAPH_STRUCTURAL_TRACE=1`; sparse Meta/step timing enabled (`GGML_META_TIMING=1`, `LLAMA_STEP_TIMING=1`, period 32); CUPTI and `LD_PRELOAD` off.

## Run

- Unit: `root-p4-v26-shortctx16k-hierarchical-ep6-d3-20261016t020607z`
- Output: `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v26-shortctx16k-hierarchical-20261016/results/root-p4-v26-shortctx16k-hierarchical-ep6-d3-20261016t020607z`
- One model load. The first diagnostic semantic scenario started (`serial-sem-00-0`, lane `capital-en`); no `scenario-complete`, semantic gate report, or completed result JSON exists.
- Fatal: `SIGSEGV`, `si_code=1`, `si_addr=0x87060f0c0c1c1818`, TID 104411; unit exit 139. V64 symbolization resolves the stack to `ggml_cuda_graph_tensor_input` → `ggml_cuda_graph_make_structural_signature` → `ggml_cuda_graph_get_key` → `ggml_backend_cuda_graph_compute` → scheduler → `llama_context::process_ubatch` / `llama_decode` / `bank::prefill_full_logical_prompts`.
- Structural trace was active on all six SM70 devices. The log records 18 `CUDA graph structural opaque external` entries and 33 `CUDA graph structural rejected graph` entries, including cycle, unsupported-external, and external-cap declines. Fail-closed declines were visible, but the full graph still faulted in input/signature handling.
- One lane timing record: `target_decode_sync_us=1843147`, `mtp_process_us=198387`, `diagnostic_dump_scan_us=81113`, `lane_prefill_total_us=2122657`. No 32-step `step-timing` or 200-dispatch `meta-timing` aggregate appeared before the crash. No throughput, semantic, full-F32, exact-token, or clean-family result is certified.

## Supervisor telemetry

The supervisor recorded 15 samples from `2026-10-16T02:06:08Z` to `02:09:20Z`.

- GPU free memory ranged 4,788–16,126 MiB/card; temperatures were 37–40°C and sampled utilization peaked at 5%. After exit, all cards returned to 16,125–16,126 MiB free and 0% utilization.
- BMC maximum temperature stayed at 40°C. Fan targets stayed at 3000 RPM: fan0 2838–2858, fan2 2869–2893, fan3 2847–2862 RPM. No fan alarms/faults; BMC masks stayed valid.
- DDR node0+8 MemFree ranged 226.1–23,594.8 MiB and unit RSS peaked at 34,542.5 MiB; DDR values were record-only, with no stop threshold.

Raw logs, telemetry, supervisor manifest, and unit journal remain under the result directory above. Matching local supervisor telemetry and journal are under `/Users/lyx/Desktop/ssh/monster-engine/standard-opt-20261014/notes/`.
