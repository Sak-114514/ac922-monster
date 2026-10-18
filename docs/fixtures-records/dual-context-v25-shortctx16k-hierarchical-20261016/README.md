# V25 hierarchical EP6 16K-per-request run

Status: **V25 model attempt failed with SIGSEGV; supervisor stopped, no autorun**. V63's nested weight-view qualification passed after the V24 full-model attempt aborted on an unregistered `blk.1.ple_conv1d.weight` view. This fixture reuses the unchanged, previously host-linked V24 PPC64LE runner binary; it does not rebuild the runner. See `STATUS.md` for the failed attempt, fatal stack, telemetry, and measured graph-reserve evidence. The supervisor and output tag are V25-specific; sparse host timing is enabled at period 32. The frozen binary retains `v24-shortctx` internal scenario labels and continues to enforce the same 16K layout and all original semantic/F32/exact-token gates.

## Configuration

- IQ3_XXS, all 512 experts retained; Q8_0 target/draft KV.
- Hierarchical EP6 with two 3-GPU islands, 2 banks × 2 lanes, D3 MTP; 16,384 tokens per request, bank context 32,768.
- Target ubatch 1024, MTP ubatch 512, PLE 8 GiB across six GPU NUMA nodes; CPU lookup precision and reserves unchanged.
- V63 complete runtime closure; no CUPTI or `LD_PRELOAD`. `GGML_META_TIMING=1`, `LLAMA_STEP_TIMING=1`, `LLAMA_STEP_TIMING_PERIOD=32` add sparse host-side timing records without a sync shim.
- Single model load; diagnostic native/F32/exact gates, then clean measurement.

## Capacity

See `CAPACITY-REVIEW.md`. The 16K/sequence tracked lower bound is about 3.71 GiB/card before uncounted graph and draft-cache allocations. This is not a fit claim; the actual V63 loader and graph gates remain authoritative. No DDR stop threshold was added.

## Safety and artifacts

`run-v25-shortctx-supervisor.py --launch` requires a fresh GPU/BMC/fan gate plus explicit root tiny-gate and GPU-window flags. It has a 20-minute supervisor limit. The corrected V23 monitor accepts the known fan targets 3000/3500/4000/4300, handles a transient zero-RPM sample by requiring repeated low tach samples, and never alters BMC services. V24 result directories remain untouched.
