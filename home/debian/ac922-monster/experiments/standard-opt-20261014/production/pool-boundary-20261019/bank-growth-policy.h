#pragma once

#include "bank-affinity-policy.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

// Optional, CPU-only context reserve layered over an already accepted affinity
// plan. The exact plan remains the complete fallback on every rejection.
namespace bank_growth_policy {

using capacities = bank_affinity_policy::capacities;
using gpu_free_vector = std::array<uint64_t, 6>;

constexpr uint32_t bucket_cells = 8192;
constexpr uint32_t maximum_capacity = 40960;
constexpr uint64_t fixed_margin_mib = 256;
constexpr uint64_t per_growth_unit_mib = 128;

struct result {
    bank_affinity_policy::plan_result plan{};
    bool applied = false;
    uint64_t growth_units = 0;
    uint64_t required_free_mib = 0;
    capacities candidate_capacity{};
    const char * reason = "not_evaluated";
};

struct recovery_result {
    result choice{};
    std::array<bool, bank_affinity_policy::bank_count> recreate{};
    bool rollback = false;
};

inline uint32_t round_up_bucket(uint32_t capacity) noexcept {
    return static_cast<uint32_t>(
            (uint64_t(capacity) + bucket_cells - 1) / bucket_cells * bucket_cells);
}

inline result apply(const bank_affinity_policy::input & source,
        const bank_affinity_policy::plan_result & exact,
        bool enabled,
        const gpu_free_vector & gpu_free_mib,
        uint64_t floor_mib = 256) noexcept {
    result output;
    output.plan = exact;
    output.candidate_capacity = exact.retention.target_capacity;

    if (!enabled) {
        output.reason = "disabled";
        return output;
    }
    if (!exact.accepted()) {
        output.reason = "exact_plan_rejected";
        return output;
    }
    if (exact.rebuild_count != exact.retention.create_count) {
        output.reason = "exact_plan_inconsistent";
        return output;
    }
    for (size_t bank = 0; bank < bank_affinity_policy::bank_count; ++bank) {
        if (exact.retention.rebuild[bank] == exact.retention.retain[bank]) {
            output.reason = "exact_plan_inconsistent";
            return output;
        }
    }
    if (!source.owners_quiescent) {
        output.reason = "owners_active";
        return output;
    }
    if (source.request_count < 2 || source.request_count > bank_affinity_policy::bank_count) {
        output.reason = "request_count_out_of_range";
        return output;
    }
    if (source.pool_capacity == 0 ||
            source.pool_capacity % bank_retention_policy::cell_alignment != 0) {
        output.reason = "invalid_pool";
        return output;
    }

    capacities candidate = exact.retention.target_capacity;
    capacities desired = exact.desired_capacity;
    std::array<bool, bank_affinity_policy::bank_count> eligible{};
    bool any_growth_slack = false;

    for (size_t bank = 0; bank < bank_affinity_policy::bank_count; ++bank) {
        const uint32_t needed = exact.needed_by_bank[bank];
        const uint32_t current = source.current_capacity[bank];
        const uint32_t exact_target = exact.retention.target_capacity[bank];

        if (needed == 0 || current == 0 ||
                source.current_ubatch[bank] != 256 || exact.desired_ubatch[bank] != 256 ||
                needed <= current || !exact.retention.rebuild[bank] ||
                exact.retention.retain[bank]) {
            continue;
        }
        // Do not pad an inconsistent or profile-driven plan. The accepted C3
        // plan must already cover true demand and rebuild solely on growth.
        if (exact_target < needed || exact_target <= current) {
            output.reason = "exact_target_inconsistent";
            return output;
        }
        if (exact_target > maximum_capacity) continue;

        const uint32_t rounded = round_up_bucket(exact_target);
        if (rounded > maximum_capacity || rounded <= exact_target) continue;

        eligible[bank] = true;
        candidate[bank] = rounded;
        desired[bank] = rounded;
        any_growth_slack = true;
    }

    if (!any_growth_slack) {
        output.reason = "no_eligible_growth";
        return output;
    }

    // Preserve the C3 mapping/hit contract: only banks already scheduled for
    // growth may change, and none of their requests may claim a retained hit.
    std::array<bool, bank_affinity_policy::bank_count> mapped{};
    for (size_t request = 0; request < source.request_count; ++request) {
        const size_t bank = exact.request_to_bank[request];
        if (bank >= bank_affinity_policy::bank_count || mapped[bank]) {
            output.reason = "invalid_exact_mapping";
            return output;
        }
        mapped[bank] = true;
        if (eligible[bank] && exact.request_hit_tokens[request] != 0) {
            output.reason = "growth_bank_has_cached_hit";
            return output;
        }
    }
    for (size_t bank = 0; bank < bank_affinity_policy::bank_count; ++bank) {
        if (eligible[bank] && !mapped[bank]) {
            output.reason = "invalid_exact_mapping";
            return output;
        }
    }

    output.candidate_capacity = candidate;
    if (bank_retention_policy::total(candidate) > source.pool_capacity) {
        output.reason = "candidate_exceeds_pool";
        return output;
    }

    for (size_t bank = 0; bank < bank_affinity_policy::bank_count; ++bank) {
        const uint64_t positive_growth = candidate[bank] > source.current_capacity[bank]
                ? uint64_t(candidate[bank]) - source.current_capacity[bank]
                : 0;
        if (positive_growth != 0) {
            output.growth_units +=
                    (positive_growth + bucket_cells - 1) / bucket_cells;
        }
    }
    output.required_free_mib = floor_mib + fixed_margin_mib +
            per_growth_unit_mib * output.growth_units;
    if (std::any_of(gpu_free_mib.begin(), gpu_free_mib.end(), [&](uint64_t free_mib) {
            return free_mib < output.required_free_mib;
        })) {
        output.reason = "insufficient_gpu_headroom";
        return output;
    }

    // Bound every context by the intended candidate vector. If retention would
    // trim or otherwise reshape any bank, reject the whole optional reserve.
    const auto retention = bank_retention_policy::make_plan(
            source.current_capacity, source.current_ubatch,
            desired, exact.desired_ubatch, source.pool_capacity,
            source.owners_quiescent, candidate, desired);
    if (!retention.accepted() || retention.target_capacity != candidate) {
        output.reason = "candidate_retention_mismatch";
        return output;
    }
    for (size_t bank = 0; bank < bank_affinity_policy::bank_count; ++bank) {
        if (eligible[bank]) {
            if (!retention.rebuild[bank] || retention.retain[bank]) {
                output.reason = "growth_bank_flags_mismatch";
                return output;
            }
        } else if (retention.rebuild[bank] != exact.retention.rebuild[bank] ||
                retention.retain[bank] != exact.retention.retain[bank]) {
            output.reason = "unchanged_bank_flags_mismatch";
            return output;
        }
    }

    output.plan = exact;
    output.plan.desired_capacity = desired;
    output.plan.retain_ceiling = candidate;
    output.plan.retention = retention;
    output.plan.rebuild_count = retention.create_count;
    output.applied = true;
    output.reason = "applied";
    return output;
}

inline recovery_result post_allocation_fallback(const result & selected,
        const bank_affinity_policy::plan_result & exact,
        bool floor_ok) noexcept {
    recovery_result recovery;
    recovery.choice = selected;
    if (floor_ok || !selected.applied) {
        return recovery;
    }

    // Keep the attempted bucket and preflight budget in choice for audit, but
    // replace the executable plan with the exact C3 decision. The caller must
    // recreate precisely the contexts whose capacities differ.
    recovery.choice.plan = exact;
    recovery.choice.applied = false;
    recovery.choice.reason = "postallocation_exact_fallback";
    recovery.rollback = true;
    for (size_t bank = 0; bank < bank_affinity_policy::bank_count; ++bank) {
        recovery.recreate[bank] = selected.plan.retention.target_capacity[bank] !=
                exact.retention.target_capacity[bank];
    }
    return recovery;
}

} // namespace bank_growth_policy
