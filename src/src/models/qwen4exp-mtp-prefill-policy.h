#pragma once

#include <cstdint>

namespace qwen4exp_mtp {

// KV-only MTP prefill is valid only when this graph is storing the MTP layer's
// attention cache. Any output row or GPU-chain-owned input needs the full graph.
constexpr bool kv_only_prefill_eligible(
        bool enabled,
        bool is_mtp_context,
        uint32_t n_mtp_layers,
        uint32_t n_outputs,
        int64_t n_tokens,
        uint32_t qsa_compress_ratio,
        bool has_gpu_chain_inputs,
        bool requests_embeddings,
        bool embeddings_nextn,
        bool embeddings_nextn_masked) noexcept {
    return enabled &&
        is_mtp_context &&
        n_mtp_layers == 1 &&
        n_outputs == 0 &&
        n_tokens > 0 &&
        qsa_compress_ratio == 0 &&
        !has_gpu_chain_inputs &&
        !requests_embeddings &&
        (!embeddings_nextn || embeddings_nextn_masked);
}

} // namespace qwen4exp_mtp
