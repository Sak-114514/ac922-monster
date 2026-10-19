#pragma once

#include "bank-retention-policy.h"
#include "production-shape-policy.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

// CPU-only initial-wave request-to-bank planning. The caller supplies exact
// token-prefix reuse lengths from each existing bank; this policy neither
// hashes prompts nor moves checkpoint/KV state between banks.
namespace bank_affinity_policy {

constexpr size_t bank_count = 4;
using capacities = production_shape::capacities;
using ubatches = bank_retention_policy::ubatches;
using hit_matrix = std::array<std::array<uint32_t, bank_count>, bank_count>;

enum class status {
    ready,
    invalid_input,
    owners_active,
    pool_too_small,
};

struct input {
    // Request-indexed arrays. required_capacity is already aligned and includes
    // the caller's KV/MTP safety allowance.
    capacities required_capacity{};
    capacities actual_prompt_tokens{};
    // L1's input_work_size used only to select the wide-single UBatch.
    capacities input_work_tokens{};
    capacities current_capacity{};
    ubatches current_ubatch{};
    hit_matrix reusable_tokens{};
    size_t request_count = 0;
    uint32_t pool_capacity = 0;
    bool affinity_enabled = false;
    // The caller sets this only after its pre-allocation free-memory admission.
    bool allow_single_existing_cache = false;
    bool owners_quiescent = true;
};

struct plan_result {
    status state = status::invalid_input;
    std::array<size_t, bank_count> request_to_bank{bank_count, bank_count, bank_count, bank_count};
    capacities needed_by_bank{};
    capacities desired_capacity{};
    ubatches desired_ubatch{};
    capacities retain_ceiling{};
    bank_retention_policy::plan_result retention{};
    capacities request_hit_tokens{};
    uint64_t total_cached_tokens = 0;
    size_t cache_hit_count = 0;
    size_t rebuild_count = 0;
    size_t displacement_count = 0;
    bool single_prefill = false;
    bool single_cached_affinity = false;
    bool used_affinity = false;

    bool accepted() const noexcept {
        return state == status::ready && retention.accepted();
    }
};

namespace detail {

inline bool identity_mapping(const std::array<size_t, bank_count> & mapping,
        size_t request_count) noexcept {
    for (size_t i = 0; i < request_count; ++i) {
        if (mapping[i] != i) return false;
    }
    return true;
}

inline status map_retention_status(bank_retention_policy::status value) noexcept {
    switch (value) {
        case bank_retention_policy::status::ready: return status::ready;
        case bank_retention_policy::status::owners_active: return status::owners_active;
        case bank_retention_policy::status::pool_too_small: return status::pool_too_small;
        case bank_retention_policy::status::invalid_input: return status::invalid_input;
    }
    return status::invalid_input;
}

inline plan_result evaluate(const input & source,
        const std::array<size_t, bank_count> & mapping,
        bool single_prefill) {
    plan_result result;
    result.request_to_bank = mapping;
    result.single_prefill = single_prefill;

    std::array<bool, bank_count> assigned{};
    for (size_t request = 0; request < source.request_count; ++request) {
        const size_t bank = mapping[request];
        if (bank >= bank_count || assigned[bank]) {
            result.state = status::invalid_input;
            return result;
        }
        assigned[bank] = true;
        result.needed_by_bank[bank] = source.required_capacity[request];
    }

    result.desired_ubatch.fill(256);
    if (single_prefill) {
        // Preserve the original long-single-request profile and its bank-0
        // pinning. The single predicate uses actual prompt tokens; the separate
        // input-work size selects the old 49-52K UBatch threshold.
        const uint32_t input_work = source.input_work_tokens[0];
        result.desired_ubatch[0] = input_work <= 49152 ? 1024 : 512;
        result.needed_by_bank[0] = std::max<uint32_t>(result.needed_by_bank[0], 66048);
    } else {
        for (size_t bank = 0; bank < bank_count; ++bank) {
            if (result.needed_by_bank[bank] > 66048) result.desired_ubatch[bank] = 128;
        }
    }

    try {
        result.desired_capacity = production_shape::choose(
                result.needed_by_bank, source.pool_capacity);
    } catch (const std::runtime_error &) {
        result.state = status::pool_too_small;
        return result;
    }
    if (single_prefill) {
        // This is the existing single-prefill profile: keep the unused contexts
        // small after choose() has applied its shared-pool check.
        for (size_t bank = 1; bank < bank_count; ++bank) result.desired_capacity[bank] = 1024;
    }

    for (size_t bank = 0; bank < bank_count; ++bank) {
        if (uint64_t(source.current_capacity[bank]) >
                2 * uint64_t(result.desired_capacity[bank])) {
            result.retain_ceiling[bank] = result.desired_capacity[bank];
        }
    }
    if (single_prefill) {
        const uint32_t idle_ceiling = result.desired_ubatch[0] == 512 ? 1024 : 17152;
        for (size_t bank = 1; bank < bank_count; ++bank) {
            result.retain_ceiling[bank] = idle_ceiling;
        }
    }

    result.retention = bank_retention_policy::make_plan(
            source.current_capacity, source.current_ubatch,
            result.desired_capacity, result.desired_ubatch,
            source.pool_capacity, source.owners_quiescent,
            result.retain_ceiling, result.desired_capacity);
    result.state = map_retention_status(result.retention.result);
    if (!result.accepted()) return result;

    result.rebuild_count = result.retention.create_count;
    for (size_t request = 0; request < source.request_count; ++request) {
        const size_t bank = mapping[request];
        const uint32_t cached = source.reusable_tokens[request][bank];
        // A hit is valid only if the exact bank context/profile remains live
        // and it can hold this request without resizing.
        if (cached != 0 && result.retention.retain[bank] &&
                result.retention.target_capacity[bank] == source.current_capacity[bank] &&
                result.desired_ubatch[bank] == source.current_ubatch[bank] &&
                source.required_capacity[request] <= source.current_capacity[bank]) {
            result.request_hit_tokens[request] = cached;
            result.total_cached_tokens += cached;
            ++result.cache_hit_count;
        }
    }
    for (size_t request = 0; request < source.request_count; ++request) {
        result.displacement_count += mapping[request] != request;
    }
    return result;
}

inline bool better(const plan_result & candidate, const plan_result & current,
        size_t request_count) noexcept {
    if (candidate.total_cached_tokens != current.total_cached_tokens) {
        return candidate.total_cached_tokens > current.total_cached_tokens;
    }
    if (candidate.cache_hit_count != current.cache_hit_count) {
        return candidate.cache_hit_count > current.cache_hit_count;
    }
    if (candidate.rebuild_count != current.rebuild_count) {
        return candidate.rebuild_count < current.rebuild_count;
    }
    if (candidate.displacement_count != current.displacement_count) {
        return candidate.displacement_count < current.displacement_count;
    }
    for (size_t i = 0; i < request_count; ++i) {
        if (candidate.request_to_bank[i] != current.request_to_bank[i]) {
            return candidate.request_to_bank[i] < current.request_to_bank[i];
        }
    }
    return false;
}

inline bool supported_single_cache_ubatch(uint32_t ubatch) noexcept {
    return ubatch == 128 || ubatch == 256 || ubatch == 512 || ubatch == 1024;
}

inline bool make_single_cached_plan(const input & source, plan_result & output) {
    // Reuse is allowed only if all four existing context objects and profiles
    // survive unchanged. This plan allocates nothing and cannot move state.
    capacities unchanged_ceiling = source.current_capacity;
    auto retention = bank_retention_policy::make_plan(
            source.current_capacity, source.current_ubatch,
            source.current_capacity, source.current_ubatch,
            source.pool_capacity, source.owners_quiescent, unchanged_ceiling,
            source.current_capacity);
    if (!retention.accepted()) return false;
    for (size_t bank = 0; bank < bank_count; ++bank) {
        if (!retention.retain[bank] ||
                retention.target_capacity[bank] != source.current_capacity[bank]) {
            return false;
        }
    }

    const uint64_t actual_prompt = source.actual_prompt_tokens[0];
    size_t selected_bank = bank_count;
    uint32_t selected_cached = 0;
    for (size_t bank = 0; bank < bank_count; ++bank) {
        const uint32_t cached = source.reusable_tokens[0][bank];
        if (cached == 0 || cached > actual_prompt ||
                uint64_t(cached) * 4 < actual_prompt * 3 ||
                actual_prompt - cached > 8192 ||
                !supported_single_cache_ubatch(source.current_ubatch[bank]) ||
                source.current_capacity[bank] == 0 ||
                source.required_capacity[0] > source.current_capacity[bank]) {
            continue;
        }
        if (selected_bank == bank_count || cached > selected_cached ||
                (cached == selected_cached && bank < selected_bank)) {
            selected_bank = bank;
            selected_cached = cached;
        }
    }
    if (selected_bank == bank_count) return false;

    output = plan_result{};
    output.state = status::ready;
    output.request_to_bank[0] = selected_bank;
    output.needed_by_bank[selected_bank] = source.required_capacity[0];
    output.desired_capacity = source.current_capacity;
    output.desired_ubatch = source.current_ubatch;
    output.retain_ceiling = unchanged_ceiling;
    output.retention = retention;
    output.request_hit_tokens[0] = selected_cached;
    output.total_cached_tokens = selected_cached;
    output.cache_hit_count = 1;
    output.single_prefill = false;
    output.single_cached_affinity = true;
    output.used_affinity = selected_bank != 0;
    output.displacement_count = selected_bank != 0;
    output.rebuild_count = 0;
    return true;
}

} // namespace detail

inline plan_result make_plan(const input & source) {
    plan_result invalid;
    if (!source.owners_quiescent) {
        invalid.state = status::owners_active;
        return invalid;
    }
    if (source.request_count == 0 || source.request_count > bank_count ||
            source.pool_capacity == 0 ||
            source.pool_capacity % bank_retention_policy::cell_alignment != 0) {
        invalid.state = status::invalid_input;
        return invalid;
    }
    for (size_t request = 0; request < source.request_count; ++request) {
        if (source.required_capacity[request] == 0 ||
                !bank_retention_policy::valid_capacity(source.required_capacity[request]) ||
                source.actual_prompt_tokens[request] == 0) {
            invalid.state = status::invalid_input;
            return invalid;
        }
    }

    const bool single_prefill = source.request_count == 1 &&
            source.actual_prompt_tokens[0] >= 8192;
    std::array<size_t, bank_count> identity{0, 1, 2, 3};
    const plan_result legacy = detail::evaluate(source, identity, single_prefill);
    // Every singleton remains bank0/cold unless the caller explicitly admits a
    // zero-rebuild existing-checkpoint reuse. The historical long prefill stays
    // on its original single-profile path when that candidate does not qualify.
    if (source.request_count == 1) {
        if (source.affinity_enabled && source.allow_single_existing_cache) {
            plan_result cached;
            if (detail::make_single_cached_plan(source, cached)) return cached;
        }
        return legacy;
    }
    if (!source.affinity_enabled) return legacy;

    plan_result best;
    bool have_hit_candidate = false;
    std::array<size_t, bank_count> mapping{bank_count, bank_count, bank_count, bank_count};
    std::array<bool, bank_count> bank_used{};
    const auto enumerate = [&](auto && self, size_t request) -> void {
        if (request == source.request_count) {
            const auto candidate = detail::evaluate(source, mapping, false);
            if (!candidate.accepted() || candidate.total_cached_tokens == 0) return;
            if (!have_hit_candidate || detail::better(candidate, best, source.request_count)) {
                best = candidate;
                have_hit_candidate = true;
            }
            return;
        }
        for (size_t bank = 0; bank < bank_count; ++bank) {
            if (bank_used[bank]) continue;
            bank_used[bank] = true;
            mapping[request] = bank;
            self(self, request + 1);
            bank_used[bank] = false;
        }
    };
    enumerate(enumerate, 0);
    if (have_hit_candidate) {
        best.used_affinity = !detail::identity_mapping(best.request_to_bank, source.request_count);
        return best;
    }
    return legacy;
}

} // namespace bank_affinity_policy
