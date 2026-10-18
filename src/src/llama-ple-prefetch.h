#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bounded asynchronous PLE row prefetch. The model must outlive the worker.
// Returns NULL unless LLAMA_PLE_PREFETCH=1 and the matching encoded row cache
// is enabled with GGML_CPU_PLE_ROW_CACHE=1.
struct llama_ple_prefetcher;
LLAMA_API struct llama_ple_prefetcher * llama_ple_prefetcher_init(const struct llama_model * model);
LLAMA_API void llama_ple_prefetcher_free(struct llama_ple_prefetcher * p);

// Submit a fresh prompt or a continuation with its exact available predecessor
// suffix in oldest-to-newest order. history_complete=false safely skips work.
// request_id must be unique while outstanding; only one pending job is kept.
LLAMA_API void llama_ple_prefetch_submit(struct llama_ple_prefetcher * p, uint64_t request_id,
                               const llama_token * tokens, int32_t n_tokens,
                               const llama_token * history, int32_t n_history,
                               bool history_complete);
LLAMA_API void llama_ple_prefetch_cancel(struct llama_ple_prefetcher * p, uint64_t request_id);

// Compatibility helper for callers that know the token list starts a fresh
// sequence. Server integrations should use submit/cancel with task IDs.
LLAMA_API void llama_ple_prefetch_tokens(struct llama_ple_prefetcher * p,
                               const llama_token * tokens, int32_t n_tokens);

#ifdef __cplusplus
}
#endif
