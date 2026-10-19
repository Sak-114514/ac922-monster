#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace dual_context_v14 {

struct shared_window {
    int64_t start_us = 0;
    int64_t end_us = 0;
    bool valid = false;
};

// Always invokes append_flow for every lane. Shared-window eligibility is a
// separate concern: a lane with no post-start token invalidates the common
// window, but must never suppress its output/semantic record.
template <typename Banks, typename AppendFlow>
shared_window append_all_flows_and_measure_window(
        const Banks & banks, int64_t parallel_begin, AppendFlow append_flow) {
    int64_t overlap_start = parallel_begin;
    int64_t overlap_end = std::numeric_limits<int64_t>::max();
    size_t lane_count = 0;
    bool all_lanes_active = true;

    for (auto * b : banks) {
        for (const auto & lane : b->lanes) {
            ++lane_count;
            const auto first_parallel = std::find_if(lane.emitted.begin(), lane.emitted.end(),
                    [parallel_begin](int64_t time) { return time >= parallel_begin; });
            if (first_parallel == lane.emitted.end()) {
                all_lanes_active = false;
            } else {
                overlap_start = std::max(overlap_start, *first_parallel);
                overlap_end = std::min(overlap_end, lane.emitted.back());
            }

            append_flow(*b, lane);
        }
    }

    if (lane_count == 0 || !all_lanes_active || overlap_end <= overlap_start) {
        return {parallel_begin, parallel_begin, false};
    }
    return {overlap_start, overlap_end, true};
}

} // namespace dual_context_v14
