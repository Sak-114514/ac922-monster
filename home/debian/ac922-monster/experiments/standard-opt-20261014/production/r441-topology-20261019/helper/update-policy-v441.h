#pragma once

namespace ac922_v441 {

enum class update_decision {
    commit_shared,
    use_per_key_fallback,
    propagate_error,
};

// Keep this policy CUDA-header-free so tests can inject status values without
// poisoning a CUDA context or manufacturing a device fault.
constexpr update_decision classify_update_status(
        int status, int success_status, int graph_update_failure_status) noexcept {
    return status == success_status ? update_decision::commit_shared :
            status == graph_update_failure_status ? update_decision::use_per_key_fallback :
            update_decision::propagate_error;
}

} // namespace ac922_v441
