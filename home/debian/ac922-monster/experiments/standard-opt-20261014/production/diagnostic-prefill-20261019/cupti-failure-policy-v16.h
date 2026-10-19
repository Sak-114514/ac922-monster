#pragma once

namespace cupti_failure_policy_v16 {

inline bool defer_stop_failure(int stop_status, bool deferred_mode) {
    return stop_status != 0 && deferred_mode;
}

} // namespace cupti_failure_policy_v16
