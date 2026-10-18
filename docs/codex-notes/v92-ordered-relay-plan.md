# V92 ordered relay plan

Root owns one engine CUDA file. V39 remains immutable V91.

Goal: test a 2V cross-island relay against existing4V relay without changing per-element FP32 rank0..5 left-associative arithmetic, stage reserves, double-buffer epochs, or fixed64block communicator grid.

1. Gather1/2 into leader0 slots1/2; gather4/5 into leader3 slots4/5. Each producing thread system-fences then joins a block barrier before readiness is published.
2. Leader0 starts with x0 (no added zero), adds x1 thenx2, writes prefix to leader3slot0.
3. Leader3 waits local4/5 and prefix0, starts with prefix then adds x3,x4,x5 in order, writes final to local3slot3 and remote0slot3.
4. Leader3 system-fences every producer then publishes final readiness to all six ranks. Each island reads its leader slot3.
5. A call k+1 final requires all six arrivals, which occur only after each consumed call k final. Thus k+2 cannot overwrite a half still read by call k. Mixed-kind epochs and flags_seg phases retain monotonic counters.

DefaultOFF GGML_CUDA_P2P_AR_RELAY6_ORDERED; active only existing relay-admitted n6 sizes200KiB..configuredmax; no new reserve; preload kernel before peer spin; logactualdispatch/graph_append kind4, 2V.

Regression: exact original rankorder cancellation/signedzero/zero-rank0+3, A/B/A changedinputs, canaries, two communicators, mixedsizes including fallback, real graph_append plus capture/replay, bounded GPUwindow. Compare200KiB/10MiB hot latency, not model speed. If rejected leave OFF and record.

## Primary-source screening 2026-10-16

- NVIDIA NCCL device `runTreeSplit`: https://github.com/NVIDIA/nccl/blob/master/src/device/all_reduce.h — separates reduction/broadcast work, maps communication to topology; not a bit-exact rank-order drop-in.
- Upstream ggml dual-GPU pipeline: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/allreduce.cu — 2-GPU pinned-host/CE staging and per-call monotonic flags. Our direct6GPU P2P relay avoids that host transport; no direct code import.
- NV-Group author lab publication record: https://nowlab.cse.ohio-state.edu/projects/17/ — specifically dense NVLink/Summit systems and cooperative link-aware reduction. NSF paper endpoint search text available, full PDF fetch returned502 twice; no full-paper-derived performance claim.
- TurboPrefill official: https://github.com/sergey-automation/TurboPrefill — requires layer split and pipeline-eligible backend topology. Current flatMeta6/HEP6 topology remains ineligible; not a magic env switch, as existing V85 audit established.

These sources motivate topology-aware staging, not a claimed hardware speedup. New ordered relay is independently implemented for current exact arithmetic requirements.
