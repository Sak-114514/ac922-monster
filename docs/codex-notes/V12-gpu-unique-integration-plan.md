# V12 GPU uniqueness / pinned-output integration plan

Goal: preserve model precision and sampler behavior while avoiding full-logit D2H for exclusion-only greedy calls when the raw maximum is uniquely valid and remains allowed.

- Private graph emits raw IDs and a separate unique+finite bit. Default off. Only CUDA leaf output devices, including a Meta tree of CUDA members, may enable it. All other backends retain old raw graph.
- Preserve raw MTP feedback, seed/carry ownership and all dynamic/grammar/penalty/temperature guards. Bias list must contain only -Inf entries; CPU membership test uses the existing helper and includes model suppress tokens.
- Unique+finite GPU winner not excluded: return the same token as full chain without logits readback. Every unsupported/tied/nonfinite/excluded case uses the current CPU/full-chain route.
- Output flags must use context-owned pinned output storage, following the pending argmax-ID buffer fix. Wait before growing/freeing buffers; copy/read/reorder only after completion. Do not introduce a second pageable D2H.
- Root owns sampling.cpp, private llama-ext.h and context integration after the pinned-buffer owner hands off. Preserve the independently implemented default-off catchup+seed fusion.
- Gates: GPU-local old-v-new argmax, TP candidate aggregation and canaries/A-B-A; CPU sampler eligibility/reference fixtures; syntax checks; incremental private build/freeze; actual short/long concurrent semantics and 32 biased shadow comparisons. Only after these gates, compare one real 2/4-stream wave and ready-driven two-bank runner. No performance claim from fixtures.
