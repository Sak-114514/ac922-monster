# AC922 architecture-specific optimization research

2026-10-15. Pinned sources in source-refs/. Targets remain real-text shared decode300+/single prefill6000+, not achieved.

## Hardware/software admission

POWER9 ppc64le, six V100SM70, CUDA12.4 native driver550.54.15; two3GPU islands. Actual host DDR31.6GiB on0/8; GPU NUMA250..255 are not extra DDR. Existing checkpoint/full512 experts/mixed IQ and q8 KV unchanged. Existing hybrid PLE uses8GiBGPU-NUMA+18.8GiBDDR. Published x86_64 CUDA12.8 wheels cannot be installed here as substitutes for a native build.

## Primary sources and decisions

- [Exact AC922 V100 fork](https://github.com/1115714829/llama.cpp-v100/blob/9c3e19b5f8e1fdbe0b6c70d193bd5e58515b6faa/docs/v100/moe-2.0.0.md): EP experts+TP dense; NUMA mapped-host tiers; per-tile Volta TC prefill kernels. Their published MoE contract is Q4 with96GBDDR, not ours. Pinned sources show fused TC decoders support Q4_K/Q5_K/Q5_1/Q8_0, not our7types. Adapt byte-safe per-K-tile IQ readers rather than repeat whole-F16-dequant trial. Optional fused F16SwiGLU excluded. GPL3v2code, keep notices if transplanted; original Strata provenance evaluated separately.
- [1Cat upstream](https://github.com/1CatAI/1Cat-vLLM): #966 pairedIQ gate/up, #1203 staged expert prefill, #992HC TP4 exchange, #993coalesced projection. Actual local dimensions/quant formats,3/6rank topologies and CUDA12.4 capability gates required. No claimed direct port from headlines. See 1cat-vllm-sm70-portability-20261015.md.
- [vLLM DBO](https://github.com/vllm-project/vllm/blob/main/docs/design/dbo.md): useful persistent microbatch owners and overlap scheduling concepts; actual implementation targets DP+EP and full graphs, not a flag usable in our GGML island model. Transfer ownership/event design only.
- [Sarathi Serve](https://github.com/microsoft/sarathi-serve) and [paper](https://arxiv.org/abs/2403.02310): bounded chunked-prefill with pipeline-aware scheduling. ExistingFIFO onlysnapshotsinputs; needs actualframe scratch and immutable memory-context snapshots, not stockpipeline_parallel toggle.
- [NVIDIA mapped memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): mapped/pinned host data still travels acrossPCIe/NVLink; UVA does not prove cheap local HBM access. Our ATS/pageable-memory access capabilities must be respected, no assumption from POWER9 name alone.

## Active implementation paths

1. Constant host owner thread per model bank (newV8runner). V7 used fresh std::async threads every step, discarding TLS state; rootfoundthisdirectlyinsource. Preserve oneinflight,exceptions,drain. CPU regression gatebeforemodel.
2. Opt-in fourlaneD2 unrolled MTP graph, with depth-owned masks/slots/inputs, GPU hidden/token edges and explicit KV write/read dependencies. Coreowner kernel_semantic_audit; fixture/syntax beforemodel.
3. Stateful island prefill frames: two scratch slots share physical per-layer KV/GDN but snapshot metadata. First cold Nsight fixture exact butzeroGPUoverlap; hotcapture/replay variant pending. Eventordering/hostenqueue nonblocking must beprovedbeforeintegration.
4. SevenIQtype streaming tile-to-TC readers. Storeweights unchanged; no globalF16weightcache; compare decoder-RNE and explicit numeric gates plusmodelsemantics. Not promoted/engineintegrated yet.
5. EP3/EP6 layout borrowing: needs LOCAL_PARTIAL nonlinear semantics (gate/up cannot be allreduced beforeSiLU), localID masking and exactlyoncefinalexpert sum. Loader-onlyaxis2change is incorrect. RootauditingcurrentMeta andnewfork implementation; no deployment.

No source/API result above is a local model speed claim. No hardware/system modifications.
