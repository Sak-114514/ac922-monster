#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace server_prompt_cache_policy {

inline size_t add_bytes(size_t total, size_t bytes) {
    return bytes > static_cast<size_t>(-1) - total ? static_cast<size_t>(-1) : total + bytes;
}

inline size_t capacity_bytes(size_t capacity, size_t element_size) {
    if (element_size == 0) {
        return 0;
    }
    return capacity > static_cast<size_t>(-1) / element_size ? static_cast<size_t>(-1) : capacity * element_size;
}

inline size_t owned_bytes(
        size_t fixed_bytes,
        size_t token_bytes,
        size_t target_capacity_bytes,
        size_t draft_capacity_bytes,
        size_t speculative_capacity_bytes,
        size_t checkpoint_bytes) {
    return add_bytes(
            add_bytes(
                    add_bytes(
                            add_bytes(
                                    add_bytes(fixed_bytes, token_bytes),
                                    target_capacity_bytes),
                            draft_capacity_bytes),
                    speculative_capacity_bytes),
            checkpoint_bytes);
}

inline bool aligned_boundary(size_t boundary, size_t block_size) {
    return block_size <= 1 || boundary % block_size == 0;
}

template<typename Checkpoints, typename CheckpointUsable>
size_t deepest_restorable_boundary(
        size_t cached_token_count,
        size_t common_prefix_tokens,
        bool endpoint_usable,
        const Checkpoints & checkpoints,
        CheckpointUsable checkpoint_usable,
        size_t block_size = 1) {
    size_t best = endpoint_usable && cached_token_count > 0 && cached_token_count <= common_prefix_tokens &&
                    aligned_boundary(cached_token_count, block_size)
            ? cached_token_count
            : 0;

    for (const auto & checkpoint : checkpoints) {
        if (checkpoint.n_tokens <= 0) {
            continue;
        }

        const size_t boundary = static_cast<size_t>(checkpoint.n_tokens);
        if (boundary > cached_token_count || boundary > common_prefix_tokens || boundary <= best ||
                !aligned_boundary(boundary, block_size)) {
            continue;
        }

        if (checkpoint_usable(checkpoint)) {
            best = boundary;
        }
    }

    return best;
}

template<typename CachedTokens, typename RequestTokens, typename Checkpoint>
bool checkpoint_usable(
        const CachedTokens & cached_tokens,
        const RequestTokens & request_tokens,
        const Checkpoint & checkpoint,
        size_t common_prefix_tokens,
        bool require_draft,
        bool require_spec,
        size_t block_size = 1,
        size_t min_replay = 1) {
    if (checkpoint.data_tgt.empty() ||
            (require_draft && checkpoint.data_dft.empty()) ||
            (require_spec && checkpoint.data_spec.empty()) ||
            checkpoint.pos_min < 0 ||
            checkpoint.pos_max < checkpoint.pos_min ||
            checkpoint.n_tokens <= 0 ||
            static_cast<size_t>(checkpoint.n_tokens) > common_prefix_tokens ||
            static_cast<size_t>(checkpoint.n_tokens) > cached_tokens.size() ||
            !aligned_boundary(static_cast<size_t>(checkpoint.n_tokens), block_size) ||
            (require_spec && (static_cast<size_t>(checkpoint.n_tokens) >= request_tokens.size() ||
                    request_tokens.size() - static_cast<size_t>(checkpoint.n_tokens) < min_replay))) {
        return false;
    }

    const int64_t n_tokens = checkpoint.n_tokens;
    if (cached_tokens.pos_next(n_tokens) != request_tokens.pos_next(n_tokens)) {
        return false;
    }

    // A checkpoint cannot seed replay if its state contains the first token to replay.
    const auto boundary_pos = request_tokens.pos_next(n_tokens);
    const auto prefix_end_pos = request_tokens.pos_next(static_cast<int64_t>(common_prefix_tokens));
    return checkpoint.pos_max < boundary_pos && checkpoint.pos_max < prefix_end_pos;
}

template<typename Prompt, typename Tokens>
size_t reusable_prefix(
        const Prompt & prompt,
        const Tokens & tokens,
        bool endpoint_state_available,
        bool require_draft,
        bool require_spec,
        size_t block_size = 1,
        size_t min_replay = 1) {
    if (prompt.tokens.has_mtmd || tokens.has_mtmd) {
        return 0;
    }

    const size_t common_prefix = prompt.tokens.get_common_prefix(tokens);

    const bool endpoint_usable = endpoint_state_available &&
            !prompt.tokens.empty() &&
            prompt.tokens.size() <= common_prefix &&
            server_prompt_cache_policy::aligned_boundary(prompt.tokens.size(), block_size) &&
            (!require_spec || (tokens.size() > prompt.tokens.size() &&
                    tokens.size() - prompt.tokens.size() >= min_replay)) &&
            prompt.tokens.pos_next(static_cast<int64_t>(prompt.tokens.size())) ==
                    tokens.pos_next(static_cast<int64_t>(prompt.tokens.size()));

    const auto is_checkpoint_usable = [&](const auto & checkpoint) {
        return server_prompt_cache_policy::checkpoint_usable(
                prompt.tokens, tokens, checkpoint, common_prefix, require_draft, require_spec,
                block_size, min_replay);
    };

    return deepest_restorable_boundary(
            prompt.tokens.size(),
            common_prefix,
            endpoint_usable,
            prompt.checkpoints,
            is_checkpoint_usable,
            block_size);
}

template<typename ExistingPrompt, typename IncomingPrompt>
bool checkpoint_boundaries_cover(
        const ExistingPrompt & existing,
        const IncomingPrompt & incoming,
        bool require_draft,
        bool require_spec,
        size_t block_size = 1) {
    const size_t endpoint_tokens = incoming.tokens.size();
    if (incoming.tokens.has_mtmd || existing.tokens.has_mtmd ||
            endpoint_tokens == 0 ||
            existing.tokens.size() != endpoint_tokens ||
            existing.tokens.get_common_prefix(incoming.tokens) != endpoint_tokens) {
        return false;
    }

    for (const auto & incoming_checkpoint : incoming.checkpoints) {
        const int64_t n_tokens = incoming_checkpoint.n_tokens;
        // The full endpoint already covers this boundary for extension requests.
        if (n_tokens <= 0 || static_cast<size_t>(n_tokens) >= endpoint_tokens ||
                !checkpoint_usable(incoming.tokens, incoming.tokens, incoming_checkpoint,
                        endpoint_tokens, require_draft, require_spec, block_size)) {
            continue;
        }

        bool covered = false;
        for (const auto & existing_checkpoint : existing.checkpoints) {
            if (existing_checkpoint.n_tokens != incoming_checkpoint.n_tokens ||
                    existing_checkpoint.pos_min != incoming_checkpoint.pos_min ||
                    existing_checkpoint.pos_max != incoming_checkpoint.pos_max) {
                continue;
            }
            if (checkpoint_usable(existing.tokens, incoming.tokens, existing_checkpoint,
                    endpoint_tokens, require_draft, require_spec, block_size)) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            return false;
        }
    }

    return true;
}

template<typename Entry, typename Prompt, typename EndpointCompleteFn, typename ReusableBoundaryFn>
bool hybrid_duplicate(
        const Entry & entry,
        const Prompt & prompt,
        bool require_draft,
        bool require_spec,
        EndpointCompleteFn endpoint_complete_fn,
        ReusableBoundaryFn reusable_boundary_fn,
        size_t block_size = 1) {
    if (!endpoint_complete_fn(entry)) {
        return false;
    }

    const bool same_endpoint = entry.prompt.tokens.size() == prompt.tokens.size() &&
            entry.prompt.tokens.get_common_prefix(prompt.tokens) == prompt.tokens.size();
    if (same_endpoint) {
        return checkpoint_boundaries_cover(entry.prompt, prompt, require_draft, require_spec, block_size);
    }

    return prompt.tokens.size() > 0 &&
            reusable_boundary_fn(entry, prompt.tokens) >= prompt.tokens.size();
}

template<typename Entry, typename Prompt, typename EndpointCompleteFn>
bool hybrid_endpoint_needs_upgrade(
        const Entry & entry,
        const Prompt & prompt,
        bool require_draft,
        bool require_spec,
        EndpointCompleteFn endpoint_complete_fn,
        size_t block_size = 1) {
    const bool same_endpoint = entry.prompt.tokens.size() == prompt.tokens.size() &&
            entry.prompt.tokens.get_common_prefix(prompt.tokens) == prompt.tokens.size();
    return same_endpoint && (!endpoint_complete_fn(entry) ||
            !checkpoint_boundaries_cover(entry.prompt, prompt, require_draft, require_spec, block_size));
}

template<typename Entries, typename Request, typename BoundaryFn>
auto select_deepest(Entries & entries, const Request & request, size_t current_boundary, BoundaryFn boundary_fn) {
    auto best = entries.end();
    size_t best_boundary = current_boundary;

    for (auto it = entries.begin(); it != entries.end(); ++it) {
        const size_t boundary = boundary_fn(*it, request);
        if (boundary > best_boundary) {
            best_boundary = boundary;
            best = it;
        }
    }

    return std::make_pair(best, best_boundary);
}

template<typename Entries, typename CurrentTokens, typename RequestTokens, typename EligibleFn>
auto select_legacy_lcp(
        Entries & entries,
        const CurrentTokens & current_tokens,
        const RequestTokens & request_tokens,
        EligibleFn eligible_fn) {
    const size_t lcp_live = current_tokens.get_common_prefix(request_tokens);
    float f_keep_best = current_tokens.size() > 0 ? float(lcp_live) / current_tokens.size() : -1.0f;
    float f_sim_best = request_tokens.size() > 0 ? float(lcp_live) / request_tokens.size() : -1.0f;

    auto best = entries.end();
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (!eligible_fn(*it)) {
            continue;
        }
        const size_t lcp = it->prompt.tokens.get_common_prefix(request_tokens);
        const float f_keep = it->prompt.tokens.size() > 0 ? float(lcp) / it->prompt.tokens.size() : 0.0f;
        const float f_sim = request_tokens.size() > 0 ? float(lcp) / request_tokens.size() : 0.0f;

        if (f_keep < 0.25f) {
            continue;
        }
        if (f_keep_best < f_keep && f_sim_best < f_sim) {
            f_keep_best = f_keep;
            f_sim_best = f_sim;
            best = it;
        }
    }
    return best;
}

template<typename Entry, typename Prompt>
bool legacy_lcp_duplicate(const Entry & entry, const Prompt & prompt) {
    return entry.prompt.tokens.get_common_prefix(prompt.tokens) == prompt.tokens.size();
}

template<typename Entries, typename Predicate>
void erase_if(Entries & entries, Predicate predicate) {
    for (auto it = entries.begin(); it != entries.end();) {
        if (predicate(*it)) {
            it = entries.erase(it);
        } else {
            ++it;
        }
    }
}

inline bool reject_multimodal(bool hybrid_policy, bool has_multimodal) {
    return hybrid_policy && has_multimodal;
}

template<typename Entries, typename Iterator>
void touch_mru(Entries & entries, Iterator it) {
    entries.splice(entries.end(), entries, it);
}

template<typename Entries, typename Iterator>
void finish_hit(Entries & entries, Iterator it, bool retain_snapshot) {
    if (retain_snapshot) {
        touch_mru(entries, it);
    } else {
        entries.erase(it);
    }
}

template<typename Entries, typename Prompt, typename ReusableFn>
bool has_reusable_entry(const Entries & entries, const Prompt & prompt, ReusableFn reusable_fn) {
    for (const auto & entry : entries) {
        if (reusable_fn(entry, prompt)) {
            return true;
        }
    }

    return false;
}

enum class admission_result {
    admit,
    duplicate,
    oversized,
};

template<typename Entries, typename Prompt, typename ReusableFn, typename CurrentBytesFn, typename EvictFn, typename BeforeRoomFn>
admission_result prepare_admission(
        Entries & entries,
        const Prompt & prompt,
        size_t new_entry_bytes,
        size_t byte_limit,
        ReusableFn reusable_fn,
        CurrentBytesFn current_bytes_fn,
        EvictFn evict_oldest_fn,
        BeforeRoomFn before_room_fn) {
    if (has_reusable_entry(entries, prompt, reusable_fn)) {
        return admission_result::duplicate;
    }

    if (byte_limit > 0 && new_entry_bytes > byte_limit) {
        return admission_result::oversized;
    }

    before_room_fn();

    if (byte_limit == 0) {
        return admission_result::admit;
    }

    while (!entries.empty()) {
        const size_t current_bytes = current_bytes_fn();
        if (current_bytes < byte_limit && new_entry_bytes <= byte_limit - current_bytes) {
            break;
        }
        evict_oldest_fn(entries.front());
    }

    return admission_result::admit;
}

template<typename State, typename RestoreTargetFn, typename RestoreDraftFn, typename OnFailureFn>
bool restore_entry(
        const State & state,
        bool draft_required,
        std::vector<uint8_t> * speculative_out,
        RestoreTargetFn restore_target,
        RestoreDraftFn restore_draft,
        OnFailureFn on_failure) {
    if (speculative_out != nullptr && state.data.spec.empty()) {
        speculative_out->clear();
        on_failure();
        return false;
    }

    if (!restore_target(state.data.main) ||
            (draft_required && !restore_draft(state.data.drft))) {
        if (speculative_out != nullptr) {
            speculative_out->clear();
        }
        on_failure();
        return false;
    }

    if (speculative_out != nullptr) {
        try {
            *speculative_out = state.data.spec;
        } catch (...) {
            speculative_out->clear();
            on_failure();
            return false;
        }
    }

    return true;
}

template<typename ReserveFn, typename RestoreFn>
bool restore_after_checkpoint_reservation(
        size_t checkpoint_bytes,
        ReserveFn reserve_fn,
        RestoreFn restore_fn,
        bool & safe_miss) {
    safe_miss = false;
    try {
        if (!reserve_fn(checkpoint_bytes)) {
            safe_miss = true;
            return false;
        }
    } catch (...) {
        safe_miss = true;
        return false;
    }

    return restore_fn();
}

} // namespace server_prompt_cache_policy
