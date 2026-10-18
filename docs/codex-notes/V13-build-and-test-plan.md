# V13 shared descriptor guard and dual-context V7

2026-10-15. Private experiment only; shared monster source unchanged.

## Build

Frozen `/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v13`; incremental delta from V12: `ggml-backend-meta.cpp`, `ggml-backend.cpp`, `llama-context.cpp`, and diagnostics-only `ggml-cuda.cu`. Three C++ and one CUDA translation units compiled; all links succeeded. Private new header `ggml-meta-replay-guard.h` contains internal opt-in guard and scheduler-root/split readiness APIs.

## Gate and model sequence

1. Bounded real CUDA3+3 shared weights/two contexts fixture. Exact A/B/A and reset, cold capture exclusion and static descriptor lifetime checks.
2. Only after that passes: V7 CAPI two66K banks / two lanes each, same132096 total KV capacity, native550 runtime; outer broad submission guard off only with `GGML_META_REPLAY_GUARD=1`.
3. V7 additionally matches server draft KV rewind policy (proposal KV removed before verify, target hidden rows re-fed), enabling existing catchup+seed fusion. Serial first-use shape warmups remain.
4. Existing CHUNKS8 path + F32 collective wire; Onecat remains enabled. Fourteen short/long semantic checks before2/4x8K512 throughput. Preserve actual outputs, shared timestamps and maps.
5. Abort own unit at58C, repeated free<512MiB, CUDA error, semantic failure, or timeout; no reset/reboot/fan/system changes.

No performance claim until actual model timing. Goals300TG/6000PP remain unachieved; shutdown only after achievement and BMC confirmation.
