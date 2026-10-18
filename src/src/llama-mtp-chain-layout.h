#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_mtp_gpu_chain_lane {
    int32_t seq_id = -1;
    int32_t seed_pos = -1;
    int32_t first_pos = -1;
};

// Validate the explicit lane-major batch layout expected by compact MTP chaining.
// Step zero is a one-row seed per lane; accepted catch-up rows are decoded before
// the chain starts. Later steps contain exactly one row per lane.
template <class SeqAt, class PosAt, class OutputAt>
inline bool llama_mtp_gpu_chain_validate_rows(
        const std::vector<llama_mtp_gpu_chain_lane> & lanes,
        int32_t step,
        size_t n_rows,
        SeqAt seq_at,
        PosAt pos_at,
        OutputAt output_at) {
    if (lanes.size() < 2 || lanes.size() > 8 || step < 0) {
        return false;
    }

    size_t row = 0;
    for (const auto & lane : lanes) {
        if (lane.seq_id < 0 || lane.seed_pos < 0 || lane.first_pos < 0 || lane.first_pos > lane.seed_pos) {
            return false;
        }

        const size_t lane_rows = step == 0 ? (size_t) (lane.seed_pos - lane.first_pos + 1) : 1;
        const int32_t first = step == 0 ? lane.first_pos : lane.seed_pos + step;
        for (size_t local = 0; local < lane_rows; ++local, ++row) {
            if (row >= n_rows || seq_at(row) != lane.seq_id || pos_at(row) != first + (int32_t) local) {
                return false;
            }
            const bool want_output = step != 0 || local + 1 == lane_rows;
            if ((output_at(row) != 0) != want_output) {
                return false;
            }
        }
    }
    return row == n_rows;
}

inline bool llama_mtp_gpu_chain_reorder_step_major(
        const int32_t * src,
        int32_t steps,
        int32_t lanes,
        int32_t * dst,
        size_t capacity) {
    if (steps < 0 || lanes < 2 || lanes > 8 || (steps > 0 && (!src || !dst)) ||
            (size_t) steps * lanes > capacity) {
        return false;
    }
    for (int32_t step = 0; step < steps; ++step) {
        for (int32_t lane = 0; lane < lanes; ++lane) {
            dst[(size_t) lane * steps + step] = src[(size_t) step * lanes + lane];
        }
    }
    return true;
}

// Only clean prefill snapshots are reusable with the compact GPU chain. Its
// device buffers are transient: begin() resets the cursor and the first seed
// graph overwrites every active lane before device feedback can be read.
inline bool llama_mtp_gpu_chain_prefill_snapshot_safe(
        bool enabled, bool active, bool tokens_empty, bool positions_empty,
        bool hidden_empty, bool draft_pending, int32_t keep) {
    return enabled && !active && tokens_empty && positions_empty && hidden_empty &&
        !draft_pending && keep == -1;
}

inline bool llama_mtp_gpu_chain_catchup_allowed(int32_t maximum, size_t rows) {
    return maximum < 0 || rows <= static_cast<size_t>(maximum);
}
