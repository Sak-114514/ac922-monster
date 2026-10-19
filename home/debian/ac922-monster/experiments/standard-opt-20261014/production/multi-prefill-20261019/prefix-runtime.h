#pragma once
#include "prefix-checkpoint-policy.h"
#include <atomic>
#include <mutex>

// Added in the private libllama-common C++ delta, not the generic snapshot API.
bool common_speculative_prefix_capture(common_speculative *, int, std::vector<float> &);
bool common_speculative_prefix_restore(common_speculative *, int, const std::vector<float> &);

class same_bank_prefix {
    using entry_type = ac922_prefix_checkpoint::bank_checkpoint;
    entry_type entry_;
    size_t charged_ = 0;
    uint32_t bank_id_ = 0;
    uint64_t epoch_ = 0;
    uint64_t profile_ = 0;
    static std::mutex & budget_mutex() { static std::mutex mutex; return mutex; }
    static size_t & used_bytes() { static size_t used = 0; return used; }
public:
    same_bank_prefix() = default;
    same_bank_prefix(const same_bank_prefix &) = delete;
    same_bank_prefix & operator=(const same_bank_prefix &) = delete;
    ~same_bank_prefix() { forget(); }
    void bind(uint32_t id, uint32_t capacity, uint32_t ubatch) {
        static std::atomic<uint64_t> next{0};
        bank_id_ = id;
        epoch_ = ++next;
        profile_ = (uint64_t(capacity) << 32) | ubatch;
    }
    bool valid() const { return charged_ != 0 && entry_.position != 0; }
    size_t bytes() const { return charged_; }
    size_t target_bytes() const { return entry_.target_recurrent_partial.size(); }
    size_t draft_bytes() const { return entry_.draft_window_state.size(); }
    void forget() {
        {
            std::lock_guard<std::mutex> lock(budget_mutex());
            if (charged_ <= used_bytes()) used_bytes() -= charged_;
            charged_ = 0;
        }
        entry_ = entry_type{};
    }
    size_t reusable_tokens(const llama_tokens & tokens) const {
        if (!valid()) return 0;
        const auto plan = ac922_prefix_checkpoint::plan_restore(
                entry_, tokens, bank_id_, epoch_, profile_, 0);
        return plan.hit ? plan.restore_position : 0;
    }
    bool capture(llama_context * target, llama_context * draft, common_speculative * spec,
            const llama_tokens & tokens, size_t position) {
        if (position == 0 || position >= tokens.size() || !draft || !spec) return false;
        std::vector<float> hidden;
        if (!common_speculative_prefix_capture(spec, 0, hidden)) return false;
        constexpr auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        const size_t target_size = llama_state_seq_get_size_ext(target, 0, flags);
        const size_t draft_size = llama_state_seq_get_size_ext(draft, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t estimate = target_size + draft_size + position*sizeof(int32_t) +
                hidden.capacity()*sizeof(float) + 2*ac922_prefix_checkpoint::kCheckpointAccountingSlack;
        if (target_size == 0 || draft_size == 0 || draft_size > size_t(64)*1024*1024 ||
                estimate > ac922_prefix_checkpoint::kMaxCheckpointBytes) return false;
        forget();
        {
            std::lock_guard<std::mutex> lock(budget_mutex());
            if (used_bytes() > ac922_prefix_checkpoint::kMaxCheckpointBytes-estimate) return false;
            used_bytes() += estimate;
            charged_ = estimate;
        }
        try {
            entry_.bank_id = bank_id_;
            entry_.context_epoch = epoch_;
            entry_.profile_key = profile_;
            entry_.position = position;
            entry_.prefix_tokens.assign(tokens.begin(), tokens.begin()+position);
            entry_.mtp_pending_h = std::move(hidden);
            entry_.target_recurrent_partial.resize(target_size);
            entry_.draft_window_state.resize(draft_size);
            const size_t got_target = llama_state_seq_get_data_ext(target,
                    entry_.target_recurrent_partial.data(), target_size, 0, flags);
            const size_t got_draft = draft_size ? llama_state_seq_get_data_ext(draft,
                    entry_.draft_window_state.data(), draft_size, 0, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
            if (got_target != target_size || got_draft != draft_size) { forget(); return false; }
            entry_.after_prompt_begin = entry_.gpu_chain_inactive = entry_.mtp_transients_clean = true;
            entry_.target_kv_live_in_bank = entry_.draft_kv_live_in_bank = true;
            const size_t actual = ac922_prefix_checkpoint::checkpoint_accounted_bytes(entry_);
            if (actual > charged_) { forget(); return false; }
            {
                std::lock_guard<std::mutex> lock(budget_mutex());
                used_bytes() -= charged_-actual;
                charged_ = actual;
            }
            return true;
        } catch (...) {
            forget();
            throw;
        }
    }
    size_t restore(llama_context * target, llama_context * draft, common_speculative * spec,
            const llama_tokens & tokens) {
        const size_t position = reusable_tokens(tokens);
        if (!position || !draft || !spec) return 0;
        constexpr auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        llama_synchronize(target);
        llama_synchronize(draft);
        bool ok = llama_state_seq_set_data_ext(target, entry_.target_recurrent_partial.data(),
                entry_.target_recurrent_partial.size(), 0, flags) == entry_.target_recurrent_partial.size();
        if (ok && !entry_.draft_window_state.empty()) {
            ok = llama_state_seq_set_data_ext(draft, entry_.draft_window_state.data(),
                    entry_.draft_window_state.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) == entry_.draft_window_state.size();
        }
        ok = ok && llama_memory_seq_rm(llama_get_memory(target), 0, llama_pos(position), -1);
        ok = ok && llama_memory_seq_rm(llama_get_memory(draft), 0, llama_pos(position), -1);
        ok = ok && common_speculative_prefix_restore(spec, 0, entry_.mtp_pending_h);
        if (!ok) {
            // Any partial restore failure invalidates both live memories before
            // the original cold prefill path can observe either one.
            llama_memory_clear(llama_get_memory(target), true);
            llama_memory_clear(llama_get_memory(draft), true);
            forget();
            return 0;
        }
        return position;
    }
};
