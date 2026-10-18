# V118: explicit fresh CUDA graph executable and host-stage diagnosis

Observed V54 abort: graph_exec_update returned a status other than success or cudaErrorGraphExecUpdateFailure; generic assertion erased the status. Original status is unknown; do not label it OOM, illegal-address, or a QSA arithmetic bug. No full-model replay is needed to reproduce the assertion before improving this path.

Add opt-in GGML_CUDA_GRAPH_FRESH_EXEC=1: after a new graph capture, destroy the old executable and instantiate the freshly captured graph directly. Keep CUDA graphs/replay, all structural and cohort gates, and all model/kernel arithmetic. Existing update-failure path already uses this reconstruction. Default remains the original update path, but unexpected errors now use CUDA_CHECK with a named CUDA error rather than a value-less assert. A fresh-instantiation failure still fails explicitly; no suppressed CUDA failure/eager fallback.

Combine the prepared optional LLAMA_FULL_STEP_TIMING host-stage spans from V114, adding THREAD CPU deltas and named reserve/memory/graphbuild/alloc/input/backend-submit phases. No new GPU synchronization/events and no model/weights/topology/quantization change. Nested spans are inclusive and cannot be blindly summed. Current user allows combined routes; this is a combined stability/diagnostic revision, not an isolated gain claim.

Back up both source files, compile only one CUDA+one C++ consumer, freeze immutable V118. Verify real native QSA 4096/8192 A/B/A with fresh-exec mode and an existing Meta3 cold-cohort replay fixture if available. Then retarget the unchanged PLE8 runner bytes under a truthful library-reuse record and one new tag; no same-revision failed model rerun. V113 virtual TopK remains standalone until its GPU operator gate.

No driver/kernel/BMC/fan/power/reset changes. Keep all prior safety limits. Goals remain unmet; do not shut down.
