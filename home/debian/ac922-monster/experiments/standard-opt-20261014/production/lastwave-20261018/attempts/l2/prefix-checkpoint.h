#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ac922_prefix_checkpoint {

// One entry belongs to one existing bank. It never owns/copies attention KV
// or QSA indexer KV; those remain resident in that bank's live contexts.
struct bank_checkpoint {
    uint32_t bank_id = 0;
    uint64_t context_epoch = 0;
    uint64_t profile_key = 0;

    // Position is a token count: this checkpoint represents positions [0, position).
    size_t position = 0;
    std::vector<int32_t> prefix_tokens;

    // Exact LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY payloads. The target contains
    // Qwen4Exp recurrent/GDN and PLE convolution state; the MTP draft context
    // may legitimately have an empty recurrent payload. Neither field may
    // contain full attention/QSA KV state.
    std::vector<uint8_t> target_recurrent_partial;
    std::vector<uint8_t> draft_recurrent_partial;

    // One exact F32 MTP pending_h row captured at the same clean prompt boundary.
    std::vector<float> mtp_pending_h;

    // The caller sets these only after source/runtime checks succeed.
    bool after_prompt_begin = false;
    bool gpu_chain_inactive = false;
    bool mtp_transients_clean = false;
    bool target_kv_live_in_bank = false;
    bool draft_kv_live_in_bank = false;
};

} // namespace ac922_prefix_checkpoint
