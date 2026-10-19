#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace ac922_v441 {

// Allows operations that acquired a context before teardown to finish, while
// rejecting late users before teardown releases any of the context's slots.
class context_gate {
public:
    explicit context_gate(uint64_t generation) noexcept : generation_(generation) {}

    context_gate(const context_gate &) = delete;
    context_gate & operator=(const context_gate &) = delete;

    bool enter() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closing_) return false;
        ++active_;
        return true;
    }

    void leave() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_ > 0) --active_;
        if (closing_ && active_ == 0) idle_.notify_all();
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
        if (active_ == 0) idle_.notify_all();
    }

    void wait_until_idle() {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_.wait(lock, [&] { return active_ == 0; });
    }

    bool is_closing() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closing_;
    }

    uint64_t generation() const noexcept { return generation_; }

private:
    const uint64_t generation_;
    mutable std::mutex mutex_;
    std::condition_variable idle_;
    size_t active_ = 0;
    bool closing_ = false;
};

class context_guard {
public:
    explicit context_guard(context_gate & gate, bool enabled = true)
        : gate_(&gate), enabled_(enabled), entered_(!enabled || gate.enter()) {}
    context_guard(const context_guard &) = delete;
    context_guard & operator=(const context_guard &) = delete;
    ~context_guard() { if (enabled_ && entered_) gate_->leave(); }
    explicit operator bool() const noexcept { return entered_; }

private:
    context_gate * gate_;
    bool enabled_;
    bool entered_;
};

} // namespace ac922_v441
