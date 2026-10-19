#pragma once

#include <cstdint>

// The selectable profiles are intentionally closed: the relaxed mode only
// moves the soft pacing trigger/resume points. The 65 C hard stop is invariant.
namespace prefill_thermal_policy {

enum class profile_id : uint8_t {
    default_guarded,
    relaxed,
};

enum class phase : uint8_t {
    normal,
    cooling,
};

enum class action : uint8_t {
    proceed,
    wait,
    hard_stop,
};

struct thresholds {
    int trigger_c;
    int resume_c;
    int hard_stop_c;
};

constexpr profile_id select_profile(bool relaxed_enabled) noexcept {
    return relaxed_enabled ? profile_id::relaxed : profile_id::default_guarded;
}

constexpr thresholds get_thresholds(profile_id profile) noexcept {
    return profile == profile_id::relaxed ? thresholds{61, 57, 65} : thresholds{59, 54, 65};
}

constexpr const char * profile_name(profile_id profile) noexcept {
    return profile == profile_id::relaxed ? "relaxed" : "default";
}

constexpr action classify(int peak_c, profile_id profile, phase current_phase) noexcept {
    const thresholds limits = get_thresholds(profile);
    if (peak_c >= limits.hard_stop_c) {
        return action::hard_stop;
    }
    if (current_phase == phase::cooling) {
        return peak_c > limits.resume_c ? action::wait : action::proceed;
    }
    return peak_c >= limits.trigger_c ? action::wait : action::proceed;
}

} // namespace prefill_thermal_policy
