#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace target_verify_pad {

constexpr int32_t kDraftDepth = 3;
constexpr int32_t kMinSupportedDraftDepth = 2;
constexpr int32_t kMaxSupportedDraftDepth = 5;

enum class row_kind : uint8_t {
    seed,
    draft,
    pad,
};

struct lane_input {
    size_t lane_index = 0;
    int32_t sequence_id = -1;
    int64_t seed_position = 0;
    int32_t seed_token = 0;
    std::vector<int32_t> draft_tokens;
    bool active = true;
};

struct target_row {
    size_t row_index = 0;
    size_t lane_index = 0;
    int32_t sequence_id = -1;
    int64_t position = 0;
    int32_t token = 0;
    row_kind kind = row_kind::seed;
    bool request_output = true;
};

struct batch_plan {
    std::vector<target_row> rows;
    // Indexed by the original lane_input vector; padding rows never appear here.
    std::vector<std::vector<size_t>> score_rows;
    std::vector<std::vector<size_t>> pad_rows;
    std::vector<size_t> active_lanes;
};

inline bool parse_opt_in(const char * value) {
    if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument("ROOT_TARGET_VERIFY_PAD_PER_ACTIVE_LANE must be exactly 0 or 1");
}

inline batch_plan make_plan(const std::vector<lane_input> & lanes,
        bool pad_active_lanes, int32_t max_draft_depth = kDraftDepth,
        size_t max_target_rows = 8) {
    if (max_draft_depth < kMinSupportedDraftDepth || max_draft_depth > kMaxSupportedDraftDepth) {
        throw std::invalid_argument("per-active-lane target padding supports draft depths 2 through 5");
    }

    batch_plan plan;
    plan.score_rows.resize(lanes.size());
    plan.pad_rows.resize(lanes.size());

    std::vector<int32_t> seen_sequences;
    std::vector<bool> seen_lane_indices(lanes.size(), false);
    for (size_t lane_slot = 0; lane_slot < lanes.size(); ++lane_slot) {
        const lane_input & lane = lanes[lane_slot];
        if (!lane.active) {
            // A completed/EOS lane may retain stale proposals. It must not add
            // target rows to this plan; lane compaction stays a separate change.
            continue;
        }
        if (lane.sequence_id < 0 || lane.seed_position < 0) {
            throw std::invalid_argument("active target lane has an invalid sequence or position");
        }
        if (lane.lane_index >= lanes.size() || seen_lane_indices[lane.lane_index]) {
            throw std::invalid_argument("active target lane indexes must be unique and in range");
        }
        if (std::find(seen_sequences.begin(), seen_sequences.end(), lane.sequence_id) != seen_sequences.end()) {
            throw std::invalid_argument("active target lane sequence IDs must be unique");
        }
        if (lane.draft_tokens.size() > (size_t) max_draft_depth) {
            throw std::invalid_argument("real draft rows exceed the selected draft depth");
        }
        seen_lane_indices[lane.lane_index] = true;
        seen_sequences.push_back(lane.sequence_id);
        plan.active_lanes.push_back(lane_slot);

        auto append = [&](int32_t token, int64_t position, row_kind kind, bool score) {
            const size_t row = plan.rows.size();
            plan.rows.push_back({ row, lane.lane_index, lane.sequence_id, position,
                    token, kind, true });
            (score ? plan.score_rows[lane_slot] : plan.pad_rows[lane_slot]).push_back(row);
        };

        append(lane.seed_token, lane.seed_position, row_kind::seed, true);
        for (size_t i = 0; i < lane.draft_tokens.size(); ++i) {
            append(lane.draft_tokens[i], lane.seed_position + 1 + (int64_t) i,
                    row_kind::draft, true);
        }

        if (pad_active_lanes) {
            const int32_t pad_token = lane.draft_tokens.empty()
                    ? lane.seed_token : lane.draft_tokens.back();
            for (size_t i = lane.draft_tokens.size(); i < (size_t) max_draft_depth; ++i) {
                append(pad_token, lane.seed_position + 1 + (int64_t) i,
                        row_kind::pad, false);
            }
        }
    }

    if (plan.rows.size() > max_target_rows) {
        throw std::length_error("target verification plan exceeds its output/row budget");
    }
    return plan;
}

struct accept_result {
    std::vector<int32_t> emitted;
    size_t accepted_drafts = 0;
    size_t hidden_target_row = 0;
    size_t kept_target_rows = 0;
    size_t discarded_target_rows = 0;
    int64_t retained_prompt_size = 0;
    int64_t kv_remove_from_position = 0;
    int32_t recurrent_rollback_index = 0;
    int32_t recurrent_snapshot_slot = 0;
};

inline accept_result score_real_prefix(const batch_plan & plan, size_t lane_slot,
        const lane_input & lane, const std::vector<int32_t> & target_tokens,
        int32_t n_rs_seq) {
    if (lane_slot >= plan.score_rows.size() || !lane.active) {
        throw std::invalid_argument("cannot score an inactive or unplanned lane");
    }
    const auto & rows = plan.score_rows[lane_slot];
    if (rows.size() != lane.draft_tokens.size() + 1 || rows.empty()) {
        throw std::invalid_argument("scoring rows do not match seed plus real draft tokens");
    }
    if (n_rs_seq < 0) {
        throw std::invalid_argument("negative recurrent rollback budget");
    }

    accept_result result;
    for (size_t i = 0; i < rows.size(); ++i) {
        const size_t row = rows[i];
        if (row >= plan.rows.size() || row >= target_tokens.size() ||
                plan.rows[row].lane_index != lane.lane_index ||
                plan.rows[row].sequence_id != lane.sequence_id ||
                plan.rows[row].kind == row_kind::pad || !plan.rows[row].request_output) {
            throw std::invalid_argument("real scoring row is missing or points at padding");
        }
        result.emitted.push_back(target_tokens[row]);
        if (i == lane.draft_tokens.size() || target_tokens[row] != lane.draft_tokens[i]) {
            break;
        }
    }
    result.accepted_drafts = result.emitted.size() - 1;
    result.hidden_target_row = rows[result.accepted_drafts];
    result.kept_target_rows = result.accepted_drafts + 1; // current seed plus accepted drafts
    // Count this lane's real and pad rows, independent of interleaved lanes.
    result.discarded_target_rows = rows.size() + plan.pad_rows[lane_slot].size() - result.kept_target_rows;
    result.retained_prompt_size = lane.seed_position + (int64_t) result.kept_target_rows;
    result.kv_remove_from_position = result.retained_prompt_size;
    result.recurrent_rollback_index = (int32_t) result.discarded_target_rows;
    // Snapshot slot 0 is newest; rollback R selects state R tokens before the
    // fixed verify tail, matching llama_memory_recurrent::s_copy/SSM snapshot rows.
    result.recurrent_snapshot_slot = result.recurrent_rollback_index;
    if (result.recurrent_rollback_index > n_rs_seq) {
        throw std::invalid_argument("acceptance tail exceeds the recurrent rollback budget");
    }
    return result;
}

} // namespace target_verify_pad
