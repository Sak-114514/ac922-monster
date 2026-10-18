// Bounded asynchronous PLE row-cache warmer. Prefetch is a hint only: the
// normal GET_ROWS path remains the source of truth and produces unchanged data.
#include "llama-ple-prefetch.h"
#include "llama-ple-prefetch-internal.h"

#include "llama-model.h"

#include "ggml.h"

#if defined(__linux__)
#include "../ggml/src/ggml-cpu/ple-row-cache-api.h"
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace {

using namespace llama_ple_prefetch_detail;

struct prefetcher {
    hash_params params;
    limits bound;
#if defined(__linux__)
    std::shared_ptr<ggml_ple_io::row_cache> cache;
#endif
    bounded_queue jobs;
    std::thread worker;
    std::atomic<uint64_t> next_id{1};
    std::atomic<uint64_t> submitted{0}, skipped{0}, replaced{0}, hashed_rows{0}, warmed_rows{0};

    ~prefetcher() {
        jobs.stop();
        if (worker.joinable()) worker.join();
#if defined(__linux__)
        if (cache) {
            std::fprintf(stderr, "ple-row-cache: capacity=%zu hits=%llu resident=%llu direct=%llu fallback=%llu io_bytes=%llu\n",
                    cache->row_capacity(), (unsigned long long) cache->hit_count(),
                    (unsigned long long) cache->resident_read_count(), (unsigned long long) cache->direct_read_count(),
                    (unsigned long long) cache->fallback_read_count(), (unsigned long long) cache->io_byte_count());
        }
#endif
        std::fprintf(stderr, "ple-prefetch: stopped requests=%llu skipped=%llu replaced_pending=%llu planned_rows=%llu warmed_rows=%llu\n",
            (unsigned long long) submitted.load(std::memory_order_relaxed),
            (unsigned long long) skipped.load(std::memory_order_relaxed),
            (unsigned long long) replaced.load(std::memory_order_relaxed),
            (unsigned long long) hashed_rows.load(std::memory_order_relaxed),
            (unsigned long long) warmed_rows.load(std::memory_order_relaxed));
    }

    void start_worker() {
        worker = std::thread([this] {
            for (;;) {
                auto job = jobs.take();
                if (!job) return;
                std::vector<uint64_t> rows;
                const bool planned = collect_rows(params, job->tokens.data(), job->tokens.size(),
                    job->history.empty() ? nullptr : job->history.data(), job->history.size(),
                    true, bound,
                    [&] { return job->cancelled.load(std::memory_order_acquire); }, rows);
                if (planned && !job->cancelled.load(std::memory_order_acquire)) {
                    hashed_rows.fetch_add(rows.size(), std::memory_order_relaxed);
#if defined(__linux__)
                    const auto result = cache->warm_rows(rows, bound.max_io_bytes, &job->cancelled);
                    warmed_rows.fetch_add(result.rows_warmed, std::memory_order_relaxed);
                    std::fprintf(stderr,
                        "ple-prefetch: request=%llu planned=%zu warmed=%zu cached=%zu direct_reads=%llu direct_bytes=%llu budget_stop=%d cancelled=%d\n",
                        (unsigned long long) job->request_id, rows.size(), result.rows_warmed, result.rows_cached,
                        (unsigned long long) result.direct_reads, (unsigned long long) result.direct_bytes,
                        int(result.budget_exhausted), int(job->cancelled.load(std::memory_order_acquire)));
#endif
                } else {
                    std::fprintf(stderr, "ple-prefetch: request=%llu skipped=%s\n",
                        (unsigned long long) job->request_id,
                        job->cancelled.load(std::memory_order_acquire) ? "cancelled" : "invalid-or-empty-plan");
                }
                jobs.finish(job);
            }
        });
    }
};

bool init_from_model(const llama_model * model, prefetcher & p) {
    if (!enabled("LLAMA_PLE_PREFETCH")) {
        std::fprintf(stderr, "ple-prefetch: disabled by LLAMA_PLE_PREFETCH\n");
        return false;
    }
    if (!model) {
        std::fprintf(stderr, "ple-prefetch: no model\n");
        return false;
    }
#if !defined(__linux__)
    std::fprintf(stderr, "ple-prefetch: row-cache prefetch requires Linux\n");
    return false;
#else
    if (!enabled("GGML_CPU_PLE_ROW_CACHE")) {
        std::fprintf(stderr, "ple-prefetch: disabled because GGML_CPU_PLE_ROW_CACHE is off\n");
        return false;
    }
    const auto & hp = model->hparams;
    if (hp.ple_ngram_size < 2 || hp.ple_ngram_size > max_ngram || hp.ple_n_heads == 0 ||
            hp.ple_n_heads > max_heads || hp.ple_heads_per_ngram == 0 ||
            hp.ple_heads_per_ngram * (hp.ple_ngram_size - 1) != hp.ple_n_heads) {
        std::fprintf(stderr, "ple-prefetch: invalid PLE hparams (ngram=%u heads=%u per_gram=%u)\n",
            hp.ple_ngram_size, hp.ple_n_heads, hp.ple_heads_per_ngram);
        return false;
    }
    const ggml_tensor * table = model->get_tensor("per_layer_token_embd.weight");
    if (!table || !table->data || table->type != GGML_TYPE_IQ4_NL || table->ne[0] != 160 ||
            table->ne[2] != 1 || table->ne[3] != 1 || !ggml_is_contiguous(table) ||
            std::strcmp(table->name, "per_layer_token_embd.weight") != 0 ||
            ggml_nbytes(table) <= 64 * 1024 * 1024 || table->ne[1] < 2) {
        std::fprintf(stderr, "ple-prefetch: PLE table is not supported by the row cache\n");
        return false;
    }
    p.cache = ggml_ple_io::acquire(table);
    if (!p.cache || p.cache->row_width() != table->nb[1] ||
            p.cache->n_rows() < uint64_t(table->ne[1]) || p.cache->row_width() > 128) {
        std::fprintf(stderr, "ple-prefetch: row-cache identity or row geometry mismatch\n");
        p.cache.reset();
        return false;
    }
    p.params.ngram = hp.ple_ngram_size;
    p.params.heads_per_gram = hp.ple_heads_per_ngram;
    p.params.n_heads = hp.ple_n_heads;
    p.params.eos = (llama_token) hp.ple_eos_token_id;
    p.params.table_rows = table->ne[1];
    for (int i = 0; i < p.params.ngram; ++i) p.params.multipliers[i] = hp.ple_layer_multipliers[i];
    for (int h = 0; h < p.params.n_heads; ++h) {
        p.params.head_offsets[h] = hp.ple_head_offsets[h];
        p.params.head_vocab_sizes[h] = hp.ple_head_vocab_sizes[h];
        if (!p.params.head_vocab_sizes[h] || p.params.head_offsets[h] >= p.params.table_rows ||
                uint64_t(p.params.head_offsets[h]) + p.params.head_vocab_sizes[h] > p.params.table_rows) {
            std::fprintf(stderr, "ple-prefetch: invalid head row range h=%d offset=%u vocab=%u rows=%llu\n",
                h, p.params.head_offsets[h], p.params.head_vocab_sizes[h],
                (unsigned long long) p.params.table_rows);
            p.cache.reset();
            return false;
        }
    }
    p.start_worker();
    std::fprintf(stderr,
        "ple-prefetch: bounded row-cache warmer active (row=%zu bytes, rows/job=%zu, direct-I/O budget/job=%zu MiB, tokens/job=%zu)\n",
        p.cache->row_width(), p.bound.max_rows, p.bound.max_io_bytes / (1024 * 1024), p.bound.max_tokens);
    return true;
#endif
}

} // namespace

struct llama_ple_prefetcher * llama_ple_prefetcher_init(const struct llama_model * model) {
    auto * p = new prefetcher();
    if (!init_from_model(model, *p)) {
        delete p;
        return nullptr;
    }
    return reinterpret_cast<struct llama_ple_prefetcher *>(p);
}

void llama_ple_prefetcher_free(struct llama_ple_prefetcher * p) {
    delete reinterpret_cast<prefetcher *>(p);
}

void llama_ple_prefetch_submit(struct llama_ple_prefetcher * p, uint64_t request_id,
        const llama_token * tokens, int32_t n_tokens,
        const llama_token * history, int32_t n_history, bool history_complete) {
    auto * pf = reinterpret_cast<prefetcher *>(p);
    if (!pf) return;
    if (!history_complete) {
        if (request_id) pf->jobs.cancel(request_id);
        pf->skipped.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "ple-prefetch: request=%llu skipped=incomplete-history\n",
            (unsigned long long) request_id);
        return;
    }
    if (!request_id || !tokens || n_tokens <= 0 || n_history < 0 || (n_history > 0 && !history)) {
        pf->skipped.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "ple-prefetch: request=%llu skipped=invalid-input\n",
            (unsigned long long) request_id);
        return;
    }
    const size_t n_predecessors = size_t(pf->params.ngram - 1);
    if (size_t(n_history) > n_predecessors) {
        history += size_t(n_history) - n_predecessors;
        n_history = int32_t(n_predecessors);
    }
    auto job = std::make_shared<prefetch_job>();
    job->request_id = request_id;
    const size_t n_copy = std::min<size_t>(size_t(n_tokens), pf->bound.max_tokens);
    job->tokens.assign(tokens, tokens + n_copy);
    if (n_history) job->history.assign(history, history + n_history);
    pf->submitted.fetch_add(1, std::memory_order_relaxed);
    const auto outcome = pf->jobs.submit(std::move(job));
    if (outcome == queue_submit_result::stopped) {
        pf->skipped.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "ple-prefetch: request=%llu skipped=worker-stopping\n",
            (unsigned long long) request_id);
    } else if (outcome == queue_submit_result::replaced_pending) {
        pf->replaced.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "ple-prefetch: request=%llu replaced-older-pending-job\n",
            (unsigned long long) request_id);
    }
}

void llama_ple_prefetch_cancel(struct llama_ple_prefetcher * p, uint64_t request_id) {
    auto * pf = reinterpret_cast<prefetcher *>(p);
    if (pf && request_id) pf->jobs.cancel(request_id);
}

void llama_ple_prefetch_tokens(struct llama_ple_prefetcher * p,
                               const llama_token * tokens, int32_t n_tokens) {
    auto * pf = reinterpret_cast<prefetcher *>(p);
    if (!pf || !tokens || n_tokens <= 0) return;
    const uint64_t id = pf->next_id.fetch_add(1, std::memory_order_relaxed);
    llama_ple_prefetch_submit(p, id, tokens, n_tokens, nullptr, 0, true);
}
