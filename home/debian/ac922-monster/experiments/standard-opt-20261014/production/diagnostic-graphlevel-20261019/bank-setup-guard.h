#pragma once

#include <mutex>

// The fallback shares one process-wide mutex. Parallel setup is admitted only
// with the separately validated per-device Onecat readiness protocol.
class bank_setup_guard {
    static std::mutex & mutex() {
        static std::mutex value;
        return value;
    }
    std::unique_lock<std::mutex> lock_;
    bool parallel_;
public:
    bank_setup_guard(bool requested, bool onecat_ready)
        : lock_(mutex(), std::defer_lock), parallel_(requested && onecat_ready) {
        if (!parallel_) lock_.lock();
    }
    bool parallel() const noexcept { return parallel_; }
};
