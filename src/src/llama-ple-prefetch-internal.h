#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace llama_ple_prefetch_detail {

using token = int32_t;
constexpr token token_null = -1;
constexpr size_t max_ngram = 8;
constexpr size_t max_heads = 64;

struct hash_params {
    int ngram = 0;
    int heads_per_gram = 0;
    int n_heads = 0;
    token eos = token_null;
    uint64_t table_rows = 0;
    std::array<uint64_t, max_ngram> multipliers{};
    std::array<uint32_t, max_heads> head_offsets{};
    std::array<uint32_t, max_heads> head_vocab_sizes{};
};

struct limits {
    size_t max_tokens = 8192;
    size_t max_rows = 16384;
    size_t max_io_bytes = 64 * 1024 * 1024;
};

// Hashes the same newest-first predecessor windows as qwen4exp::set_input.
// history is ordered oldest-to-newest and must contain the exact available
// predecessor suffix (or be empty for a known-fresh sequence).
template<class Cancelled>
bool collect_rows(const hash_params & hp,
                  const token * tokens, size_t n_tokens,
                  const token * history, size_t n_history,
                  bool history_complete, const limits & bound,
                  Cancelled cancelled, std::vector<uint64_t> & output) {
    output.clear();
    if (!tokens || !n_tokens || (n_history && !history) || !history_complete || hp.ngram < 2 || hp.ngram > int(max_ngram) ||
            hp.heads_per_gram <= 0 || hp.n_heads <= 0 || hp.n_heads > int(max_heads) ||
            hp.heads_per_gram * (hp.ngram - 1) != hp.n_heads || !hp.table_rows ||
            !bound.max_tokens || !bound.max_rows) return false;

    for (int h = 0; h < hp.n_heads; ++h) {
        if (!hp.head_vocab_sizes[h] || hp.head_offsets[h] >= hp.table_rows ||
                uint64_t(hp.head_offsets[h]) + hp.head_vocab_sizes[h] > hp.table_rows) return false;
    }

    const size_t tokens_to_hash = std::min(n_tokens, bound.max_tokens);
    std::unordered_set<uint64_t> rows;
    rows.reserve(std::min(bound.max_rows, tokens_to_hash * size_t(hp.n_heads)));

    for (size_t i = 0; i < tokens_to_hash; ++i) {
        if (cancelled()) return false;
        std::array<token, max_ngram> ctx{};
        ctx[0] = tokens[i];
        bool cut = false;
        for (size_t s = 1; s < size_t(hp.ngram); ++s) {
            const int64_t prev_index = int64_t(n_history) + int64_t(i) - int64_t(s);
            const token previous = prev_index < 0 ? token_null :
                (prev_index < int64_t(n_history) ? history[prev_index] : tokens[size_t(prev_index) - n_history]);
            cut = cut || previous < 0 || previous == hp.eos;
            ctx[s] = cut ? hp.eos : previous;
        }
        for (int n = 2; n <= hp.ngram; ++n) {
            uint64_t mixed = uint64_t(ctx[0]) * hp.multipliers[0];
            for (int j = 1; j < n; ++j) mixed ^= uint64_t(ctx[size_t(j)]) * hp.multipliers[size_t(j)];
            const int base = (n - 2) * hp.heads_per_gram;
            for (int g = 0; g < hp.heads_per_gram; ++g) {
                const int h = base + g;
                const uint64_t row = mixed % hp.head_vocab_sizes[h] + hp.head_offsets[h];
                if (row >= hp.table_rows) return false;
                rows.insert(row);
                if (rows.size() >= bound.max_rows) goto collected;
            }
        }
    }

collected:
    if (cancelled()) return false;
    output.assign(rows.begin(), rows.end());
    std::sort(output.begin(), output.end());
    return !output.empty();
}

struct prefetch_job {
    uint64_t request_id = 0;
    std::vector<token> tokens;
    std::vector<token> history;
    std::atomic<bool> cancelled{false};
};

enum class queue_submit_result { queued, replaced_pending, stopped };

// One in-flight job plus one replaceable pending job. Replacing a pending job
// cancels and releases it; explicit request cancellation reaches both slots.
class bounded_queue {
    std::mutex mutex_;
    std::condition_variable cv_;
    std::shared_ptr<prefetch_job> pending_;
    std::shared_ptr<prefetch_job> active_;
    bool stopping_ = false;
public:
    queue_submit_result submit(std::shared_ptr<prefetch_job> job) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            job->cancelled.store(true, std::memory_order_release);
            return queue_submit_result::stopped;
        }
        const bool replaced = bool(pending_);
        if (pending_) pending_->cancelled.store(true, std::memory_order_release);
        pending_ = std::move(job);
        cv_.notify_one();
        return replaced ? queue_submit_result::replaced_pending : queue_submit_result::queued;
    }

    std::shared_ptr<prefetch_job> take() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stopping_ || bool(pending_); });
        if (stopping_) return {};
        active_ = std::move(pending_);
        return active_;
    }

    void finish(const std::shared_ptr<prefetch_job> & job) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_ == job) active_.reset();
    }

    void cancel(uint64_t request_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_ && pending_->request_id == request_id) {
            pending_->cancelled.store(true, std::memory_order_release);
            pending_.reset();
        }
        if (active_ && active_->request_id == request_id) {
            active_->cancelled.store(true, std::memory_order_release);
        }
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        if (pending_) pending_->cancelled.store(true, std::memory_order_release);
        if (active_) active_->cancelled.store(true, std::memory_order_release);
        pending_.reset();
        cv_.notify_all();
    }
};

inline bool enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] == '1' && value[1] == '\0';
}

} // namespace llama_ple_prefetch_detail
