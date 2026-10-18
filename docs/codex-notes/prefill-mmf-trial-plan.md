# Prefill MMF trial plan (fixture only)

Date: 2026-10-15
Owner: Codex
Status: preparation; no GPU execution until root explicitly releases a test window.

## Scope and constraints

- Own only `fixtures/prefill-mmf-trial-20261015/*` and `notes/prefill-mmf-trial-plan.md` / `notes/prefill-mmf-trial-results.md`.
- Do not edit engine/core code or rebuild it. Use the frozen V10 engine/runtime only as reference metadata; do not stop, query, or interfere with its GPU workload.
- No model load. The fixture extracts only the required expert weight shards from the existing GGUF file using the header-derived tensor offsets/strides.
- First candidates: IQ3_XXS gate/up and IQ4_NL down, 512 and 1024 input tokens, top-10 expert IDs, balanced-uniform and skewed route distributions.
- Test only the local TP shard shapes: gate/up [K=2560,N=224,E=512]; down [K=224,N=2560,E=512].

## Path under test

1. Extract each real quantized local shard from layer 35 of the existing IQ3_XXS-MTP-Q8 GGUF. Keep original quant bytes.
2. On GPU, call the existing `ggml_get_to_fp16_cuda(type)` on the flattened contiguous local shard to produce temporary F16 weights. The non-contiguous selector does not support IQ types, so the fixture uses the existing contiguous converter. Total element counts for both shards are divisible by 256. IQ4_NL is stored as independent 32-value blocks; its converter processes 256-value chunks that can cross row boundaries safely, so the real K=224 TP shard remains unpadded.
3. Generate deterministic F32 activations and genuine-sized top-10 expert ID matrices. The uniform route balances assignments across 512 experts; the skewed route concentrates most assignments in a hot expert subset while retaining a nonzero tail.
4. Reuse the existing grouped MMF Volta kernel (`mul_mat_f_ids<half2,32,16,nwarps>`) directly from its internal header in the fixture. Use the existing device-side `ggml_cuda_launch_mm_ids_helper` to compute expert bounds/maps so captured execution has no host-built counts.
5. Run the current grouped MMQ kernel as the production-format reference on the same quantized weights and inputs, with device-side expert compaction, tile offsets, Q8_1 scatter, and grouped MMQ.
6. Capture candidate and reference pipelines in CUDA Graphs with persistent pointers. Replay at the same shape after changing the ID buffer to check device-side counts/maps remain dynamic.

## Accuracy gates

- CPU `ggml_get_type_traits(type)->to_float` is the FP32 dequantized-weight reference for sampled weights. Report F16 conversion max absolute error, RMS error, and relative L2 error.
- For output correctness, compare MMF against a direct FP32 dequantized dot reference on a deterministic bounded sample of routed assignments and output rows, then compare full-batch MMF against grouped MMQ. Do not require bit equality against MMQ: MMQ quantizes activations to Q8_1 while MMF casts F32 activations to F16 in its tile.
- Initial numerical go/no-go for a full test: finite outputs; F16 weights vs FP32 dequant weight relative L2 <= 1e-3; sampled output relative L2 <= 2e-3; and no systematic NaN/Inf or output-layout mismatch. These are fixture screening thresholds, not a full-model semantic-quality certification.
- Preserve the original quantized tensors and grouped-MMQ result. No weight conversion is persisted or promoted.

## Performance and memory accounting

Record separate CUDA-event timings for weight dequantization, ID compaction, MMQ offset/Q8_1 scatter, MMQ core, MMF core, and graph-replayed combined path. Record allocation peak and exact shard bytes.

Expected temporary F16 sizes per local matrix:
- gate/up: 2560 x 224 x 512 x 2 bytes = 560 MiB;
- down: 224 x 2560 x 512 x 2 bytes = 560 MiB.

Quantized shard sizes and transfer/upload time are reported separately. The device run must fit conversion scratch, input/output buffers, MMQ reference buffers, and graph storage together. No model weights are retained in F16 after the fixture exits.

## Stop conditions

- If the contiguous CUDA converter cannot dequantize the flattened local shards with the source block ordering intact, stop and report; do not pad or change engine code without a source-backed need.
- If compile-only validation fails, fix only the fixture; do not touch engine files.
- If the MMF kernel path fails to compile for SM70 or cannot be captured with device-side ID compaction, stop before GPU execution.
- Request a GPU window from root only after fixture compile/link, host-side ID/shape assertions, and extraction-script dry-run are ready. No kernel run occurs before explicit release.
