# V271 layer-GPU PLE and shared MTP embedding preparation

This fixture is a V271 copy of the **V259 R2** D2 lifecycle: one 64K single-lane context followed by four sequential 16K/512 one-lane contexts, with target layer split `[10,10,9,7,8,6]` and the model-default MTP owner on CUDA5. It retains the original single64K and four-bank native semantic scenario files.

V271 switches only the PLE/input placement mode from required hybrid PLE to `LLAMA_PLE_SEPARATE_LAYER=1` plus model-owned `LLAMA_MTP_SHARE_GPU_EMBEDDING=1`. Hybrid NUMA, required-hybrid, NUMA quotas, and legacy PLE shard flags are explicitly off/zero. The default model source and target weights remain the V268 runtime profile; no D3 arm is added.

The profile is preflight-only and has `launch_eligible=false`. Its local checker validates the D2 order, exact runtime/profile/source/header/checker hashes, split and owner, separate-layer PLE flags, semantic gate inputs, and all known pre-load checks. The hostlink script only links against frozen V268 DSOs and checks `ldd`; it does not load a model or initialize CUDA.
