#pragma once

namespace warmup_policy_v14 {

inline bool valid_requested_frames(int requested_frames) {
    return requested_frames >= 1 && requested_frames <= 8;
}

inline bool submit_more_serial_warmup_frames(int requested_frames, int completed_frames, bool bank_active) {
    return valid_requested_frames(requested_frames) && bank_active && completed_frames < requested_frames;
}

} // namespace warmup_policy_v14
