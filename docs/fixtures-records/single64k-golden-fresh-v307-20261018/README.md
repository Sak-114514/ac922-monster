# V307 golden D2 fresh-exec baseline

This local fixture is derived from the qualified V237 single-request, per-sequence 64K route. The C++ source is byte-identical to the V237/V233 runner except for one admission-clause deletion: it no longer requires `LLAMA_FULL_STEP_TIMING=1`. This permits the exact profile overlay to use `LLAMA_FULL_STEP_TIMING=0`; all D2/dedicated-unroll, PLE placement, UBatch256→512, context, KV, scenario labels, and native prompt bytes remain as qualified.

The profile pins `runtime-perf-v299`, `GGML_CUDA_GRAPH_FRESH_EXEC=1`, shared MTP embedding, and four pinned scheduler host slots. Full-step logs and graph miss tracing are off; `LLAMA_QSA_INDEXER_TRACE` is unset. No build, runner launch, model load, or GPU run has occurred. The one hostlink is held until the root-v306 model unit is inactive.

`native-checker-read-closure.json` records the unchanged V50 checker’s transitive source reads and offline checks of both saved V237 Paris results. The V118 fresh-exec summary is retained as a bounded kernel/Meta3 acceptance reference, not a model speed claim.
