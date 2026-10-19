#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace kv_layout_policy_v221 {

constexpr uint32_t kPromptTokens = 16384;
constexpr uint32_t kOutputTokens = 512;
constexpr uint32_t kBankContext = 17152;
constexpr uint32_t kMtpPadRows = 6;
constexpr uint32_t kGenericGpuChainCapacity = 32;

struct layout {
    const char * variant;
    bool unified;
    uint32_t bank_context;
    uint32_t context_per_sequence;
    uint32_t sequence_count;
    uint32_t kv_streams;
};

inline bool family_depth_allowed(uint32_t depth) {
    return depth >= 2 && depth <= 4;
}

inline bool request_and_pad_fit(uint32_t depth, uint32_t prompt = kPromptTokens,
        uint32_t output = kOutputTokens, uint32_t context = kBankContext,
        uint32_t pad_rows = kMtpPadRows) {
    return family_depth_allowed(depth) && depth <= kGenericGpuChainCapacity &&
            uint64_t(prompt) + output + depth + 1 <= context && depth + 1 <= pad_rows;
}

// Fixed-prefix unrolled rows=6 are supported by the current core only through
// D4. V221 tests only D3 on the generic single-context chain; valid policy depths are D2 through D4.
inline bool fixed_prefix_unrolled_supported(uint32_t depth, uint32_t rows = kMtpPadRows) {
    return rows == 6 && depth >= 2 && depth <= 4;
}

inline bool generic_chain_only(uint32_t depth, bool unrolled_requested) {
    return family_depth_allowed(depth) && !unrolled_requested;
}

inline bool parse(const char * unified_text, const char * lanes_text,
        const char * per_sequence_text, const char * bank_context_text, layout & out) {
    if (!unified_text || !lanes_text || !per_sequence_text || !bank_context_text ||
            std::strcmp(unified_text, "0") != 0) {
        return false;
    }
    char * end = nullptr;
    const unsigned long lanes = std::strtoul(lanes_text, &end, 10);
    if (end == lanes_text || !end || *end != '\0') return false;
    end = nullptr;
    const unsigned long per_sequence = std::strtoul(per_sequence_text, &end, 10);
    if (end == per_sequence_text || !end || *end != '\0') return false;
    end = nullptr;
    const unsigned long bank_context = std::strtoul(bank_context_text, &end, 10);
    if (end == bank_context_text || !end || *end != '\0') return false;

    if (lanes != 1 || per_sequence != kBankContext || bank_context != kBankContext) return false;
    out = {"per_seq_16k_plus_mtp_pad6_x1", false, kBankContext, kBankContext, 1, 1};
    return true;
}

inline bool matches_actual(const layout & expected, uint32_t context_total,
        uint32_t context_per_sequence, uint32_t sequence_count) {
    if (!context_total || !context_per_sequence || context_total % context_per_sequence != 0) {
        return false;
    }
    return context_total == expected.bank_context &&
            context_per_sequence == expected.context_per_sequence &&
            sequence_count == expected.sequence_count &&
            context_total / context_per_sequence == expected.kv_streams;
}

} // namespace kv_layout_policy_v221
