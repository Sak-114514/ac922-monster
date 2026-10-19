#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace stage_event_diagnostic {

enum class window {
    none,
    prefill32,
    decode4,
};

enum class reject_reason {
    none,
    disabled,
    not_live_wave,
    unknown_id,
    already_consumed,
    another_window_active,
    semantic_gate_not_passed,
    owners_not_drained,
    contexts_not_ready,
    topology_not_plain_cuda_layer6,
    wrong_request_count,
    wrong_geometry,
    cache_prompt_enabled,
    predicted_cache_hit,
    bank_collision,
};

inline const char * to_string(reject_reason reason) {
    switch (reason) {
        case reject_reason::none: return "none";
        case reject_reason::disabled: return "diagnostic_opt_in_off";
        case reject_reason::not_live_wave: return "not_live_wave";
        case reject_reason::unknown_id: return "unknown_wave_id";
        case reject_reason::already_consumed: return "wave_id_already_consumed";
        case reject_reason::another_window_active: return "another_window_active";
        case reject_reason::semantic_gate_not_passed: return "semantic_gate_not_passed";
        case reject_reason::owners_not_drained: return "owners_not_drained";
        case reject_reason::contexts_not_ready: return "contexts_not_ready";
        case reject_reason::topology_not_plain_cuda_layer6: return "topology_not_plain_cuda_layer6";
        case reject_reason::wrong_request_count: return "wrong_request_count";
        case reject_reason::wrong_geometry: return "wrong_geometry";
        case reject_reason::cache_prompt_enabled: return "cache_prompt_enabled";
        case reject_reason::predicted_cache_hit: return "predicted_cache_hit";
        case reject_reason::bank_collision: return "bank_collision";
    }
    return "unknown_reject_reason";
}

struct request_shape {
    uint32_t prompt_tokens = 0;
    uint32_t max_tokens = 0;
    uint32_t predicted_cached_tokens = 0;
    uint32_t context_capacity = 0;
    uint32_t ubatch = 0;
    uint8_t bank = 0;
    bool cache_prompt = true;
};

struct wave_shape {
    const char * id = nullptr;
    bool opt_in = false;
    bool live_wave = false;
    bool semantic_gate_passed = false;
    bool owners_drained = false;
    bool contexts_ready = false;
    bool plain_cuda_layer6 = false;
    size_t request_count = 0;
    std::array<request_shape, 4> requests{};
};

struct decision {
    window selected = window::none;
    reject_reason reason = reject_reason::none;
    const char * id = nullptr;

    explicit operator bool() const { return selected != window::none; }
};

class policy {
public:
    decision admit(const wave_shape & wave) {
        if (!wave.opt_in) return {window::none, reject_reason::disabled, wave.id};
        if (!wave.live_wave) return {window::none, reject_reason::not_live_wave, wave.id};
        if (wave.id == nullptr) return {window::none, reject_reason::unknown_id, nullptr};

        window candidate = window::none;
        bool * consumed = nullptr;
        if (std::string(wave.id) == "diag-event-prefill32") {
            candidate = window::prefill32;
            consumed = &prefill32_consumed_;
        } else if (std::string(wave.id) == "diag-event-decode4") {
            candidate = window::decode4;
            consumed = &decode4_consumed_;
        } else {
            return {window::none, reject_reason::unknown_id, wave.id};
        }

        if (*consumed) return {window::none, reject_reason::already_consumed, wave.id};
        // Exact diagnostic IDs are one-shot even if their geometry or admission
        // preconditions are malformed. This prevents retrying the same label
        // later with a different request shape.
        *consumed = true;

        if (active_ != window::none) {
            return {window::none, reject_reason::another_window_active, wave.id};
        }
        if (!wave.semantic_gate_passed) {
            return {window::none, reject_reason::semantic_gate_not_passed, wave.id};
        }
        if (!wave.owners_drained) return {window::none, reject_reason::owners_not_drained, wave.id};
        if (!wave.contexts_ready) return {window::none, reject_reason::contexts_not_ready, wave.id};
        if (!wave.plain_cuda_layer6) {
            return {window::none, reject_reason::topology_not_plain_cuda_layer6, wave.id};
        }

        const reject_reason geometry = candidate == window::prefill32 ?
                validate_prefill32(wave) : validate_decode4(wave);
        if (geometry != reject_reason::none) return {window::none, geometry, wave.id};

        active_ = candidate;
        return {candidate, reject_reason::none, wave.id};
    }

    // Called by the runtime scope guard after the synchronous live-wave runner
    // returns or unwinds. It is safe to call more than once.
    bool finish(window selected) {
        if (selected == window::none || active_ != selected) return false;
        active_ = window::none;
        return true;
    }

    bool active() const { return active_ != window::none; }
    window active_window() const { return active_; }

private:
    static reject_reason validate_uncached_request(const request_shape & request) {
        if (request.cache_prompt) return reject_reason::cache_prompt_enabled;
        if (request.predicted_cached_tokens != 0) return reject_reason::predicted_cache_hit;
        if (request.prompt_tokens == 0 || request.max_tokens == 0 || request.ubatch == 0 ||
                request.context_capacity < request.prompt_tokens + request.max_tokens + 4) {
            return reject_reason::wrong_geometry;
        }
        return reject_reason::none;
    }

    static reject_reason validate_prefill32(const wave_shape & wave) {
        if (wave.request_count != 1) return reject_reason::wrong_request_count;
        const request_shape & request = wave.requests[0];
        const reject_reason uncached = validate_uncached_request(request);
        if (uncached != reject_reason::none) return uncached;
        if (request.prompt_tokens != 32768 || request.max_tokens != 8) {
            return reject_reason::wrong_geometry;
        }
        return reject_reason::none;
    }

    static reject_reason validate_decode4(const wave_shape & wave) {
        if (wave.request_count != 4) return reject_reason::wrong_request_count;
        std::array<bool, 4> seen_banks{};
        for (size_t i = 0; i < wave.request_count; ++i) {
            const request_shape & request = wave.requests[i];
            const reject_reason uncached = validate_uncached_request(request);
            if (uncached != reject_reason::none) return uncached;
            if (request.max_tokens != 512 || request.bank >= seen_banks.size()) {
                return reject_reason::wrong_geometry;
            }
            if (seen_banks[request.bank]) return reject_reason::bank_collision;
            seen_banks[request.bank] = true;
        }
        return reject_reason::none;
    }

    bool prefill32_consumed_ = false;
    bool decode4_consumed_ = false;
    window active_ = window::none;
};

} // namespace stage_event_diagnostic
