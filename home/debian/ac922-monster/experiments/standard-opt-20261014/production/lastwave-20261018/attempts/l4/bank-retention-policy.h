#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

// Pure CPU-side planning for the four production bank contexts. A desired
// capacity <= idle_cells marks an inactive bank. Inactive contexts can retain
// their larger existing capacity while the shared pool has room.
namespace bank_retention_policy {

constexpr size_t bank_count = 4;
constexpr uint32_t idle_cells = 1024;
constexpr uint32_t warm_cells = 17152;
constexpr uint32_t cell_alignment = 256;

using capacities = std::array<uint32_t, bank_count>;
using ubatches = std::array<uint32_t, bank_count>;
using bank_order = std::array<size_t, bank_count>;

enum class status {
    ready,
    owners_active,
    invalid_input,
    pool_too_small,
};

struct plan_result {
    status result = status::invalid_input;
    capacities target_capacity{};
    std::array<bool, bank_count> retain{};
    std::array<bool, bank_count> rebuild{};
    bank_order release_order{};
    size_t release_count = 0;
    bank_order create_order{};
    size_t create_count = 0;

    bool accepted() const noexcept {
        return result == status::ready;
    }
};

inline uint64_t total(const capacities & values) noexcept {
    uint64_t sum = 0;
    for (uint32_t value : values) {
        sum += value;
    }
    return sum;
}

inline bool valid_capacity(uint32_t value) noexcept {
    return value % cell_alignment == 0;
}

inline plan_result make_plan(
        const capacities & current_capacity,
        const ubatches & current_ubatch,
        const capacities & desired_capacity,
        const ubatches & desired_ubatch,
        uint32_t pool_capacity,
        bool owners_quiescent,
        const capacities & maximum_retained_capacity = capacities{}) noexcept {
    plan_result result;
    if (!owners_quiescent) {
        result.result = status::owners_active;
        return result;
    }
    if (pool_capacity == 0 || !valid_capacity(pool_capacity)) {
        result.result = status::invalid_input;
        return result;
    }

    capacities floor{};
    capacities target{};
    for (size_t i = 0; i < bank_count; ++i) {
        const bool no_context = current_capacity[i] == 0;
        if (!valid_capacity(current_capacity[i]) || !valid_capacity(desired_capacity[i]) ||
                desired_ubatch[i] == 0 ||
                (no_context && current_ubatch[i] != 0) ||
                (!no_context && current_ubatch[i] == 0)) {
            result.result = status::invalid_input;
            return result;
        }

        const bool inactive = desired_capacity[i] <= idle_cells;
        floor[i] = inactive ? idle_cells : std::max(warm_cells, desired_capacity[i]);
        // UBatch changes can materially alter compute-buffer reservation. Do
        // not carry a large old context into a different UBatch family.
        target[i] = current_ubatch[i] != desired_ubatch[i] ? floor[i] :
                std::max(current_capacity[i], floor[i]);
        if (maximum_retained_capacity[i] != 0) {
            target[i] = std::max(floor[i],std::min(target[i],maximum_retained_capacity[i]));
        }
    }

    if (total(floor) > pool_capacity) {
        result.result = status::pool_too_small;
        return result;
    }

    // Retain existing capacity first. If growth would exceed the shared pool,
    // reclaim unused capacity from inactive banks before trimming active banks.
    uint64_t planned_total = total(target);
    for (int pass = 0; pass < 2 && planned_total > pool_capacity; ++pass) {
        while (planned_total > pool_capacity) {
            size_t victim = bank_count;
            uint32_t largest_slack = 0;
            for (size_t i = 0; i < bank_count; ++i) {
                const bool inactive = desired_capacity[i] <= idle_cells;
                if ((pass == 0) != inactive) {
                    continue;
                }
                const uint32_t slack = target[i] - floor[i];
                if (slack > largest_slack) {
                    largest_slack = slack;
                    victim = i;
                }
            }
            if (victim == bank_count) {
                break;
            }
            const uint64_t excess = planned_total - pool_capacity;
            const uint32_t reclaim = static_cast<uint32_t>(
                    std::min<uint64_t>(largest_slack, excess));
            target[victim] -= reclaim;
            planned_total -= reclaim;
        }
    }
    if (planned_total > pool_capacity) {
        result.result = status::pool_too_small;
        return result;
    }

    result.target_capacity = target;
    for (size_t i = 0; i < bank_count; ++i) {
        result.rebuild[i] = current_capacity[i] != target[i] ||
                current_ubatch[i] != desired_ubatch[i];
        result.retain[i] = !result.rebuild[i];
        if (result.rebuild[i]) {
            result.release_order[result.release_count++] = i;
            result.create_order[result.create_count++] = i;
        }
    }

    // Release the largest existing contexts first, then create the smallest
    // targets first. Ties remain in bank-index order for deterministic plans.
    std::sort(result.release_order.begin(),
            result.release_order.begin() + result.release_count,
            [&](size_t left, size_t right) {
                if (current_capacity[left] != current_capacity[right]) {
                    return current_capacity[left] > current_capacity[right];
                }
                return left < right;
            });
    std::sort(result.create_order.begin(),
            result.create_order.begin() + result.create_count,
            [&](size_t left, size_t right) {
                if (target[left] != target[right]) {
                    return target[left] < target[right];
                }
                return left < right;
            });

    result.result = status::ready;
    return result;
}

} // namespace bank_retention_policy
