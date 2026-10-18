#pragma once

#include <array>
#include <cstdint>

template<class SeqCount, class SeqId>
bool llama_target_fifo_rows_eligible(uint32_t n_tokens, uint32_t n_outputs,
        uint32_t n_seqs, uint32_t n_unique, uint32_t tokens_per_seq,
        uint32_t max_seqs, bool equal_seqs, SeqCount count, SeqId seq) {
    if (n_tokens <= 64 || n_outputs != 0 || !equal_seqs ||
            n_seqs == 0 || n_seqs > 8 || n_seqs != n_unique ||
            max_seqs == 0 || max_seqs > 8 || tokens_per_seq == 0 ||
            uint64_t(n_seqs) * tokens_per_seq != n_tokens) return false;
    std::array<bool, 8> seen{};
    for (uint32_t lane = 0; lane < n_seqs; ++lane) {
        const uint32_t first = lane * tokens_per_seq;
        if (count(first) != 1) return false;
        const int32_t id = seq(first);
        if (id < 0 || uint32_t(id) >= max_seqs || seen[id]) return false;
        seen[id] = true;
        for (uint32_t row = first; row < first + tokens_per_seq; ++row) {
            if (count(row) != 1 || seq(row) != id) return false;
        }
    }
    return true;
}
