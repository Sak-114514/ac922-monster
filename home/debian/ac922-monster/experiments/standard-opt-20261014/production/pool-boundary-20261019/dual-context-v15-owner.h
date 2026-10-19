#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <cstdlib>
#include <cstring>

#include "fatal-backtrace-v15.h"

// One model bank, one stable host/CUDA owner. No unbounded task queue.
class dual_context_owner {
public:
    dual_context_owner() : worker_([this] { run(); }) {}
    dual_context_owner(const dual_context_owner &) = delete;
    dual_context_owner & operator=(const dual_context_owner &) = delete;

    ~dual_context_owner() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_one();
        worker_.join();
    }

    std::future<void> submit(std::function<void()> function) {
        auto task = std::make_unique<work>();
        task->function = std::move(function);
        auto result = task->completion.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ || busy_) {
                throw std::logic_error("owner stopped or already has an in-flight task");
            }
            busy_ = true;
            pending_ = std::move(task);
        }
        ready_.notify_one();
        return result;
    }

private:
    struct work {
        std::function<void()> function;
        std::promise<void> completion;
    };

    void run() {
        const char * fatal_trace = std::getenv("ROOT_FATAL_BACKTRACE");
        if (fatal_trace != nullptr && std::strcmp(fatal_trace, "1") == 0 &&
                !fatal_backtrace_v15::install_current_thread_altstack()) {
            static constexpr char marker[] = "[fatal-backtrace-v15] owner-thread altstack install failed\n";
            (void) ::write(STDERR_FILENO, marker, sizeof(marker) - 1);
        }
        for (;;) {
            std::unique_ptr<work> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || pending_ != nullptr; });
                if (!pending_) return;
                task = std::move(pending_);
            }
            std::exception_ptr failure;
            try { task->function(); } catch (...) { failure = std::current_exception(); }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                busy_ = false;
            }
            // A ready future must imply that a subsequent submission is legal.
            if (failure) task->completion.set_exception(failure);
            else task->completion.set_value();
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::unique_ptr<work> pending_;
    bool stopping_ = false;
    bool busy_ = false;
    std::thread worker_;
};
