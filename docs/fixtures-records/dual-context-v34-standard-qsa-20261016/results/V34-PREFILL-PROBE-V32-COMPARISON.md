# V34 diagnostic single-8K prefill probe vs V32

Both records use `single8k-pp-probe-32`, an uncached 8,192-token prompt with target UB1024/MTP UB512, full-logit diagnostic dumps and `GGML_META_GPU_SPANS` period 1. V34 had no Nsight or CUPTI instrumentation. These are diagnostic-family measurements, not clean performance results.

| Field | V32 diagnostic | V34 diagnostic | V34 − V32 |
|---|---:|---:|---:|
| Target prefill sum | 17.693653 s | 16.086226 s | −1.607427 s (−9.1%) |
| MTP prefill sum | 30.918281 s | 1.554533 s | −29.363748 s (−95.0%) |
| Prefill scenario wall | 48.702603 s | 17.764786 s | −30.937817 s (−63.5%) |
| Whole scenario wall | 52.936730 s | 20.810087 s | −32.126643 s (−60.7%) |
| Target-only input rate field | 462.991 tok/s | 509.256 tok/s | +10.0% |
| Aggregate input rate field | 168.205 tok/s | 461.137 tok/s | +174.2% |

V34 logged `qsa-standard-sparse requested=1 selected=1`, including the 8K probe (`streams=1`, `queries_per_stream=1024`, 2,051 selected cells). The V34 record’s `shared_tg` value is a decode-rate field and is intentionally excluded from this prefill comparison.

## Quality and stop status

The 12 native semantic cases passed. The run then began `real8k-p4-2x2`, but the supervisor stopped the unit when GPU free memory fell below the 512 MiB/card safety floor (lowest observed 256 MiB). The unit exited successfully during that stop, but the F32 preclean gate, exact-token/auxiliary checks, and clean-family run did not complete. No performance claim is quality-certified.

Raw V34 records are in `root-p4-v34-target_pad_only-16k-hierarchical-ep6-d3-20261016t064541z/copy1/diagnostic/results.json`. The V32 comparison record is in `dual-context-v32-parent-walkfast-2bank2-20261016/results/root-p4-v32-parent-walkfast-shortctx16k-hierarchical-ep6-d3-20261016t052412z/copy1/diagnostic/results.json`.
