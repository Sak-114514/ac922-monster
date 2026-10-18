# V100 grouped MoE and Tensor Core path audit

Date: 2026-10-15
Scope: read-only inspection of the private CUDA source and the frozen V9b server log. No source, build, model, or GPU state was changed.

## Conclusion

The V9b 512/1024-token MoE prefill is already dispatched through the compact grouped MMQ path for every logged MUL_MAT_ID invocation. On SM70, that path executes the quantized DP4A MMQ family, not Tensor Core MMA. The name ggml_cuda_mmq_get_config_ampere is only the tile/config selection reached for Volta; it does not enable the MMA data layout. The device-side MMQ utility selector enables that layout only for Turing/AMD, otherwise quantized IQ/Q types select DP4A dot and writeback.

There is no direct opt-in that makes these seven quantized weight formats use the existing F16 Tensor Core matrix path. The existing grouped MUL_MAT_ID WMMA implementation is reusable only after weights are available as F16 (or another supported non-quantized type); ggml_cuda_should_use_mmf() explicitly rejects quantized weights. It also rejects the current 1024-token dimensions even if they were F16. A new path would therefore need (1) quantized-weight dequantization to an F16 device shard or an on-the-fly fused dequant+WMMA kernel and (2) an opt-in MMF policy for the large-batch shapes.

The current grouped route has already removed the per-expert CPU-synchronizing GEMM chain: it compacts expert IDs, builds tile offsets, quantizes/scatters activations to Q8_1, then launches one grouped MMQ kernel per weight matrix on the CUDA stream. V9b logged 5,085/5,085 grouped-path selections on devices 0–2 and 4,851/4,851 on devices 3–5. A new F16 route would not remove that synchronization cost, because it is absent on the currently selected path.

## Evidence

### Model formats and per-expert sharding

The header-only model-row fixture at fixtures/volta-tc-real/manifest.json reads the actual GGUF directory and samples expert tensors with these seven formats:

- IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S
- IQ4_NL, Q2_0

Its sampled gate/up matrices have shape [K=2560, N=640, E=512]; down matrices have shape [K=640, N=2560, E=512]. The grouped MMQ switch supports those seven formats, and the adaptive-J switch explicitly lists the same seven in ggml/src/ggml-cuda/mmq-moe-grouped.cu:39-43,70-86. The source MMQ utility maps their IQ/Q formats to DP4A kernels at ggml/src/ggml-cuda/mmq.cuh:544-669.

The Qwen4EXP tensor-split callback places gate/up expert weights on split axis 1 and down expert weights on split axis 0 (src/llama-model.cpp:325-337). In a 3-device island, that means each gate/up GPU keeps full K=2560 and a local output width near 640/3; each down GPU keeps full N=2560 and a local K near 640/3. The actual split boundaries honor block granularity and rotate the remainder across devices. For the two down-weight block sizes in the manifest examples:

- IQ4_NL (block length 32): local widths are 192/224/224.
- Q2_0 (block length 64): local widths are 192/192/256.

Thus the per-expert local MMF shapes to target are gate/up [2560, 192..256, 512], down [192..256, 2560, 512], with the 192/224/256 row assignments rotated by layer. Source granularity is selected by get_split_granularity() at src/llama-model.cpp:456-480,562-579; the layer/device rotation is at src/llama-model.cpp:202-210,662-690.

### Current SM70 arithmetic path

- ggml/src/ggml-cuda/mmq.cuh:188-201: MMA data layout is enabled only for AMD MMA or Turing MMA. SM70 is not one of those cases.
- ggml/src/ggml-cuda/mmq.cuh:252-258,277-285: the host/device config chooses the Ampere config for a compiled Volta-capable binary. This is a configuration table and is separate from the arithmetic-layout predicate above.
- ggml/src/ggml-cuda/mmq.cuh:544-558,628-669: when MMA layout is false, quantized IQ/Q types route to DP4A dot and DP4A writeback.
- ggml/src/ggml-cuda/mmq.cu:342-348: V100 MoE explicitly chooses MMQ at any batch when compiled architecture is below Turing, because the per-expert fallback is a host-sync plus many small GEMMs.
- ggml/src/ggml-cuda/mmvq.cuh:3 sets the MMVQ batch maximum to 8. mmvq.cu:315-325 returns that limit for Volta, so 512/1024-token expert products are not MMVQ.

The V9b log corroborates that dispatch: results/20261015T044252Z/server.log reports all grouped calls selected on each device. Adaptive-J is selected for 2,985/5,085 calls on devices 0–2 (58.7%) and 2,919/4,851 on devices 3–5 (60.2%). The remainder is logged under a combined shape/type/shared-memory/override fallback reason, so the counts do not isolate each reason. Grouped MMQ still handles those calls.

### Existing grouped MMF/WMMA reuse path

ggml/src/ggml-cuda/mmf.cu:72-97 already compacts MUL_MAT_ID expert IDs on device for batches above 16; mmf.cuh:305-345 contains a single grouped mul_mat_f_ids kernel grid across experts. On Volta, the F16 branch uses the Volta MMA tile path (mmf.cuh:317-333). It consumes F32 activations and does not require one host launch/synchronization per expert. This is a useful execution skeleton for an F16-weight experiment.

However, ggml/src/ggml-cuda/mmf.cu:133-137 immediately returns false for quantized types. Its MUL_MAT_ID size guards at lines 162-167 reject the present 1024-token workload even after conversion: gate/up has N=640 and batch 1024 (>512); down has N=2560 and batch 1024 (>128). The normal ggml_cuda_mul_mat_id dispatcher (ggml/src/ggml-cuda/ggml-cuda.cu:2840-2865) reaches MMF only if this predicate accepts the weight type and shape. There is no direct quantized-weight-dequant-to-F16 branch in the current MUL_MAT_ID dispatch.

The existing MMF path is a custom WMMA kernel, not a grouped cuBLAS call. The ordinary cuBLAS route is a separate path and does not supply the compact expert-grouped behavior needed here.

## Frozen V9b measurements

From results/20261015T044252Z/server.log:

| Concurrent 8K prompts | Per-stream prompt throughput | Aggregate prompt throughput |
|---:|---:|---:|
| 1 | 992.57 tok/s | 992.57 tok/s |
| 2 | 601.05 and 529.08 tok/s | about 1,130 tok/s |
| 4 | 313.08–313.60 tok/s | about 1,253 tok/s |

The four-stream aggregate is approximately 4.8× below the 6,000 tok/s goal. This log alone does not attribute that remaining gap to MoE arithmetic: it contains no kernel-time breakdown. Since adaptive J improved the local MoE fixture by roughly 1.5× but the model result moved little (parent-provided evidence), further MMQ tile tuning is unlikely to close the whole gap.

## Feasibility and recommended next test

A full F16 cache is not viable: one full expert matrix contains 838,860,800 values and occupies 1.5625 GiB in F16; the 3 routed-expert matrices across 36 layers would be about 168.75 GiB before runtime buffers. A per-layer/per-matrix scratch is more plausible: the local TP shards above are roughly 480–640 MiB F16 per active GPU for one matrix. That scratch has to fit alongside the current graph/KV buffers, and each matrix would be dequantized again unless a much smaller reusable cache is justified.

The main risk is bandwidth and extra conversion work. At 1024 prefill tokens and top-10 routing, average occupancy is 20 tokens per each of 512 experts, so nearly the entire expert set is touched. Dequantizing to a larger F16 representation may create substantial write/read traffic and erase Tensor Core gains. The existing MMF kernel also consumes F32 activations and converts its input tile for MMA, avoiding the grouped MMQ Q8_1 scatter but introducing F16 rounding. Both weight and activation rounding can alter logits; this path is not “quality unchanged” until it passes exact-token/logit and semantic gates.

Recommended bounded test if the parent elects to proceed:

1. Reuse MMF’s grouped ID compaction and per-expert grid, but keep current quantized grouped MMQ as default and fallback.
2. In a fixture only, add a dequant-to-F16 shard for one representative gate/up tensor and one down tensor, using the actual 3-way local shapes above. Compare full-precision dequant reference against MMF output and record rounding/error; do not repeat the prior integer-Tensor-Core prototypes in fixtures/volta-tc*.
3. Only if fixture accuracy is acceptable, profile dequant+MMF versus grouped DP4A MMQ at 512 and 1024 tokens with CUDA events. Count conversion, scatter/compaction, MMF, and scratch separately. Keep CUDA graph compatibility by having all offsets/counts stay device-side as the current grouped MMQ does.
4. If scratch or bandwidth makes this slower, stop the route. If it wins and quality gates pass, then evaluate one real 8K prompt before combining it.

This is a candidate for a small isolated experiment, not a credible 5× prefill solution by source inspection alone.
