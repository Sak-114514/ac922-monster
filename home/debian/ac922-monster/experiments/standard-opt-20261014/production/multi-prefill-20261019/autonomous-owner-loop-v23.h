#pragma once

#include <exception>
#include <utility>

namespace dual_context_v23 {

struct owner_loop_result {
    unsigned long frames = 0;
};

// Run complete per-bank steps on that bank's single long-lived owner thread.
// run_one_step must include target submission/drain and the corresponding MTP
// process/accept/rollback/hidden-state collection before returning.
template <typename IsActive, typename RunOneStep, typename DrainOwnedContexts>
owner_loop_result drive_owner_until_done(
        IsActive is_active, RunOneStep run_one_step, DrainOwnedContexts drain_owned_contexts) {
    owner_loop_result result;
    try {
        while (is_active()) {
            run_one_step();
            ++result.frames;
        }
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        try {
            drain_owned_contexts();
        } catch (...) {
            // Preserve the frame's original failure; the owning bank has
            // already attempted a drain and the coordinator drains the wave.
        }
        std::rethrow_exception(failure);
    }
    return result;
}

// Wait for every submitted bank owner loop before propagating the first error.
// The caller's failure path then drains all bank contexts after no owner is
// still mutating its state.
template <typename Banks, typename JoinOne>
void join_all_owner_loops(const Banks & banks, JoinOne join_one) {
    std::exception_ptr first_failure;
    for (const auto bank : banks) {
        try {
            join_one(bank);
        } catch (...) {
            if (!first_failure) {
                first_failure = std::current_exception();
            }
        }
    }
    if (first_failure) {
        std::rethrow_exception(first_failure);
    }
}

} // namespace dual_context_v23
