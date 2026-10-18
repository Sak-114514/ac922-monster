# V26 hierarchical EP6 16K-per-request candidate

Status: **V26 full-model attempt failed with SIGSEGV; supervisor stopped and no auto-relaunch**. V26 reuses the same PPC64LE runner binary proven in V24/V25 (SHA recorded in the plan) and switches only the complete dynamic closure to frozen V64. The binary keeps V24 internal scenario labels and preserves all native semantic, full-F32, exact-token, diagnostic and clean gates. This fixture does not modify core code. See `STATUS.md` for the failed run, V64 graph trace, safety telemetry, and remaining quality-gate status.

## Configuration held constant

IQ3_XXS with all 512 experts, Q8_0 K/V, hierarchical EP6, PLE 8 GiB across six GPU NUMA nodes, two banks × two lanes, D3 MTP, 16,384 context per request, target ubatch 1024, MTP ubatch 512. Reserves are unchanged.

V64-specific diagnostic: `GGML_CUDA_GRAPH_STRUCTURAL_TRACE=1`. Sparse timing remains enabled (`GGML_META_TIMING=1`, `LLAMA_STEP_TIMING=1`, period 32); CUPTI and `LD_PRELOAD` stay off.

## Gate and launch policy

The launcher requires `V26_SHORTCTX_TINY_GATE_PASSED=1` and `V26_SHORTCTX_GPU_WINDOW_AUTHORIZED=1`, a fresh GPU/BMC/fan gate, and the V64 whole-library hash/ldd checks. The supervisor uses `/bin/bash` to invoke the wrapper and has a 20-minute maximum. No DDR stop threshold is added; DDR telemetry remains record-only. There is no autorun or model startup from this preparation.

V25 predecessor status and evidence are recorded in `../dual-context-v25-shortctx16k-hierarchical-20261016/STATUS.md`. Its CUDA Graph structural-signature crash remains a known failure; V64's opaque-extra tiny gate is pending and must pass before root grants a V26 full-model run.
