# Volta integer operands through FP16 Tensor Core — root design

Status: hardware arithmetic fixture passed; no production kernel or model gain yet.

## Evidence

`fixtures/volta-integer-mma.cu` ran on native CUDA12.4/driver550, V100 SM70. 100 cases, 25,600 outputs exactly match CPU int64 dot32 cast to FP32, including ±2048 weight coefficients and -128/127 activations. CUDA Graph replay used. No model conversion. Actual capabilities: pageable=0, ATS=0, directManagedMemAccessFromHost=1. The independent anonymous GPU-NUMA CPU-read experiment remains valid; do not claim GPU ATS.

Actual expert types: IQ2_XXS/XS/S, IQ3_XXS/S for gate/up; IQ4_NL/Q2_0 for down. Current grouped prefill already quantizes activations to Q8 D4, one FP32 scale per32 values, and unpacks weights to signed integer codes plus FP32 scales. New kernel must reuse this exact representation, without F16 rounding of dequantized values or another activation quantizer.

## Integration route

Reuse compact expert live-tile routing and existing `ggml_cuda_mmq_load_tiles_*` decoders. SM70 MMQ uses unpacked signed bytes and per32 scales (IQ2_XS/S per16); these integers are exactly representable in half. WMMA integer-dot K16/K32 accumulators remain below2^24 and therefore exact. Multiply original FP32 weight/activation scales after each original scale group, retaining arithmetic grouping. Do not copy MMVQ integer division/truncation semantics into prefill MMQ.

Place opt-in grouped TC implementation in a new header included only by mmq-moe-grouped.cu. Avoid editing mmq.cuh/common headers that would rebuild170CUDA objects. Restrict first implementation to SM70, token count>32, all seven supported formats, validated dimensions; else explicit logged DP4A fallback. Existing decode path unchanged. No full F16 expert bank or new weight format.

Potential body: fixedJ16 tile withI128/nwarps8 where supported. Existing packed-x tile loader and sorted-y tile loading remain unchanged. Per warp16x16 output tile; convert signed byte inputs to half, WMMA exact integer dot per scale group, apply original scale, keep FP32 accumulators. Preserve ids_dst writeback and tail masks. Shared scratch must fit96KiB with existing tile allocation; inspect actual config rather than assumeI128/nwarps8. ConsiderK16 staging to limit scratch. No claimed speedup until actual mixed-type fixture with existing grouped MMQ reference and model semantic gate.

## Gates

1. CPU coefficient/scales exactness for all seven formats from actual GGUF expert blocks.
2. GPU fixture against existing grouped MMQ on actual expert matrices and identical original Q8 activation representation; nonfinite checks, max/relative/RMS error, scaled reduction ordering.
3. Only then model semantics, slot reuse and 8K timing. New quantization/expert pruning prohibited.
