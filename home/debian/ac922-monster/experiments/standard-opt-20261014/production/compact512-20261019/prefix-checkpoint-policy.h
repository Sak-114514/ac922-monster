#pragma once

#include "prefix-checkpoint.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace ac922_prefix_checkpoint {

constexpr size_t kMaxCheckpointBytes = size_t(512) * 1024 * 1024;
constexpr size_t kCheckpointAccountingSlack = 64 * 1024;

enum class miss_reason {
    none,
    invalid_checkpoint,
    context_epoch_changed,
    profile_changed,
    bank_changed,
    unsafe_state,
    payload_over_budget,
    token_prefix_mismatch,
    no_replay_token,
};

struct restore_plan {
    bool hit = false;
    miss_reason reason = miss_reason::invalid_checkpoint;
    size_t exact_lcp = 0;
    size_t restore_position = 0;
    size_t replay_tokens = 0;
};

inline size_t saturating_add(size_t a, size_t b) {
    return b > std::numeric_limits<size_t>::max() - a
            ? std::numeric_limits<size_t>::max() : a + b;
}

inline size_t checkpoint_accounted_bytes(const bank_checkpoint & checkpoint) {
    size_t bytes = kCheckpointAccountingSlack;
    bytes = saturating_add(bytes, checkpoint.prefix_tokens.capacity() * sizeof(int32_t));
    bytes = saturating_add(bytes, checkpoint.target_recurrent_partial.capacity());
    bytes = saturating_add(bytes, checkpoint.draft_window_state.capacity());
    bytes = saturating_add(bytes, checkpoint.mtp_pending_h.capacity() * sizeof(float));
    return bytes;
}

inline size_t total_accounted_bytes(const std::vector<const bank_checkpoint *> & checkpoints) {
    size_t bytes = 0;
    for (const bank_checkpoint * checkpoint : checkpoints) {
        if (checkpoint != nullptr) {
            bytes = saturating_add(bytes, checkpoint_accounted_bytes(*checkpoint));
        }
    }
    return bytes;
}

inline bool within_budget(size_t total_bytes, size_t budget_bytes = kMaxCheckpointBytes) {
    return total_bytes <= budget_bytes;
}

inline size_t exact_token_lcp(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    const size_t limit = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < limit && a[i] == b[i]) {
        ++i;
    }
    return i;
}

inline restore_plan plan_restore(
        const bank_checkpoint & checkpoint,
        const std::vector<int32_t> & request_tokens,
        uint32_t current_bank_id,
        uint64_t current_context_epoch,
        uint64_t current_profile_key,
        size_t other_checkpoint_bytes, // resident checkpoints excluding this bank's entry
        size_t budget_bytes = kMaxCheckpointBytes) {
    restore_plan result;
    if (checkpoint.position == 0 || checkpoint.prefix_tokens.size() != checkpoint.position ||
            checkpoint.target_recurrent_partial.empty() || checkpoint.draft_window_state.empty() ||
            checkpoint.mtp_pending_h.empty()) {
        result.reason = miss_reason::invalid_checkpoint;
        return result;
    }
    if (checkpoint.bank_id != current_bank_id) {
        result.reason = miss_reason::bank_changed;
        return result;
    }
    if (checkpoint.context_epoch != current_context_epoch) {
        result.reason = miss_reason::context_epoch_changed;
        return result;
    }
    if (checkpoint.profile_key != current_profile_key) {
        result.reason = miss_reason::profile_changed;
        return result;
    }
    if (!checkpoint.after_prompt_begin || !checkpoint.gpu_chain_inactive ||
            !checkpoint.mtp_transients_clean || !checkpoint.target_kv_live_in_bank ||
            !checkpoint.draft_kv_live_in_bank) {
        result.reason = miss_reason::unsafe_state;
        return result;
    }
    const size_t checkpoint_bytes = checkpoint_accounted_bytes(checkpoint);
    if (checkpoint_bytes > budget_bytes || other_checkpoint_bytes > budget_bytes - checkpoint_bytes) {
        result.reason = miss_reason::payload_over_budget;
        return result;
    }

    result.exact_lcp = exact_token_lcp(checkpoint.prefix_tokens, request_tokens);
    if (result.exact_lcp < checkpoint.position) {
        result.reason = miss_reason::token_prefix_mismatch;
        return result;
    }
    // The checkpoint stores memory but not target logits. Require at least one
    // real token to be replayed after restore before accepting the cache hit.
    if (request_tokens.size() <= checkpoint.position) {
        result.reason = miss_reason::no_replay_token;
        return result;
    }

    result.hit = true;
    result.reason = miss_reason::none;
    result.restore_position = checkpoint.position;
    result.replay_tokens = request_tokens.size() - checkpoint.position;
    return result;
}

} // namespace ac922_prefix_checkpoint
