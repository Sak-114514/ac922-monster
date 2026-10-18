# Current priority: execution and scheduling, not routine expert kernels

User steering: V100/current expert-format kernel marginal return is low. This is a prioritization decision, not a proof of the absolute arithmetic ceiling.

1. Finish V335: V329 device-tail submission, independently created ordinary draft UB1024, target UB512/original fastest layer split/PLE18 unchanged. Six-device actual GGML exact/lifetime gate passed, full-model result pending.
2. If submission benefit materializes, port the bounded bridge to an explicit decode eligibility mode and test the existing fastest four-bank genericD3 recipe. Do not replace it with the slower dedicatedD3 result.
3. Draft prefill scheduling: persistent single owner, stage original token/position/sequence and shifted h rows into owned existing batch, release target only after complete consumption, overlap next target chunk with prior draft prefill. Depth1 and drain before reuse/end. First inspect active input costs; unused mask is already unallocated and not filled, so mask metadata cleanup is deprioritized.
4. Revisit layer balance only jointly with a proven reduction in submission overhead. Existing V327 balanced split/admission succeeded but was slower; do not report static balance as a performance win.

Metrics remain whole-request real uncached64K prefill, target-only prefill as secondary, four-lane shared steady and inclusive decode separately. Native1699 is a canary, not full quality acceptance. No model weights, experts, quantization, BMC/fan/system controls changed by these candidates.
