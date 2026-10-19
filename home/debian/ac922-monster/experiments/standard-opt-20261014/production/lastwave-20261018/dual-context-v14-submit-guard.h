#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>

// Runner-only guard: serialize host submission/build calls while allowing the
// queued device work to drain after release.
class dual_context_submit_guard {
public:
    using clock = std::chrono::steady_clock;

    explicit dual_context_submit_guard(bool enabled)
        : enabled_(enabled), requested_(clock::now()), lock_(shared_mutex(), std::defer_lock) {
        if (enabled_) {
            lock_.lock();
            acquired_ = clock::now();
        }
    }

    dual_context_submit_guard(const dual_context_submit_guard &) = delete;
    dual_context_submit_guard & operator=(const dual_context_submit_guard &) = delete;

    ~dual_context_submit_guard() {
        release();
    }

    int64_t wait_us() const {
        if (!enabled_) return 0;
        return std::chrono::duration_cast<std::chrono::microseconds>(acquired_ - requested_).count();
    }

    // Unlock explicitly before stream draining. The returned duration only
    // counts time held by the CPU submission section.
    int64_t release() {
        if (!enabled_ || !lock_.owns_lock()) return hold_us_;
        hold_us_ = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - acquired_).count();
        lock_.unlock();
        return hold_us_;
    }

    int64_t hold_us() const {
        return hold_us_;
    }

private:
    static std::mutex & shared_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    const bool enabled_;
    const clock::time_point requested_;
    clock::time_point acquired_{};
    std::unique_lock<std::mutex> lock_;
    int64_t hold_us_ = 0;
};
