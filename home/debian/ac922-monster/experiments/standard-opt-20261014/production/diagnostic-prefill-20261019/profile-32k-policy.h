#pragma once

#include <cstddef>
#include <string_view>

namespace ac922_diag_prefill32 {

inline constexpr std::string_view kWaveId = "diag-current-raw32k";
inline constexpr std::size_t kInputTokens = 32768;

inline bool should_arm_live_wave(bool opted_in, bool collector_completed) noexcept {
    return opted_in && !collector_completed;
}

struct candidate {
    bool opted_in = false;
    bool prefill_phase = false;
    std::string_view wave_id;
    std::size_t request_count = 0;
    int bank_id = -1;
    std::size_t input_tokens = 0;
    std::size_t cached_tokens = 0;
    std::size_t chunk_index = 0;
    std::size_t chunk_tokens = 0;
    bool target_decode = false;
    bool mtp_available = false;
};

inline bool should_capture(const candidate & value) noexcept {
    return value.opted_in && value.prefill_phase && value.wave_id == kWaveId &&
            value.request_count == 1 && value.bank_id == 0 &&
            value.input_tokens == kInputTokens && value.cached_tokens == 0 &&
            value.chunk_index == 0 && value.chunk_tokens == kInputTokens &&
            value.target_decode && value.mtp_available;
}

} // namespace ac922_diag_prefill32
