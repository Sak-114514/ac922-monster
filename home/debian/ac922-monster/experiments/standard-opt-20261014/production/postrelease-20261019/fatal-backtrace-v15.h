#pragma once

// Best-effort fatal-signal diagnostics for a process that cannot safely
// recover from SIGSEGV/SIGBUS. The handler only writes fixed markers and
// integer fields with write(); glibc's backtrace unwinder is prewarmed before
// CUDA/model initialization and is intentionally best-effort in signal context.

#include <cstddef>
#include <cstdint>
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#endif
#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace fatal_backtrace_v15 {

constexpr size_t kFrameLimit = 64;
constexpr size_t kAltStackBytes = 128 * 1024;
inline volatile sig_atomic_t handling_signal = 0;

inline void write_bytes(const char * data, size_t size) {
    while (size != 0) {
        const ssize_t written = ::write(STDERR_FILENO, data, size);
        if (written <= 0) return;
        data += written;
        size -= static_cast<size_t>(written);
    }
}

inline void write_text(const char * text) {
    size_t size = 0;
    while (text[size] != '\0') ++size;
    write_bytes(text, size);
}

inline void write_unsigned(uint64_t value, unsigned base) {
    char digits[32];
    size_t count = 0;
    do {
        const unsigned digit = static_cast<unsigned>(value % base);
        digits[count++] = static_cast<char>(digit < 10 ? '0' + digit : 'a' + digit - 10);
        value /= base;
    } while (value != 0 && count < sizeof(digits));
    while (count != 0) write_bytes(&digits[--count], 1);
}

inline uint64_t current_thread_id() {
#if defined(__linux__)
    return static_cast<uint64_t>(::syscall(SYS_gettid));
#elif defined(__APPLE__)
    uint64_t tid = 0;
    (void) pthread_threadid_np(nullptr, &tid);
    return tid;
#else
    return 0;
#endif
}

inline const char * signal_name(int signal_number) {
    if (signal_number == SIGSEGV) return "SIGSEGV";
    if (signal_number == SIGBUS) return "SIGBUS";
    return "UNKNOWN";
}

inline void fatal_signal_handler(int signal_number, siginfo_t * info, void *) {
    if (handling_signal) {
        write_text("\n[fatal-backtrace-v15] recursive fatal signal; exiting\n");
        ::_exit(128 + signal_number);
    }
    handling_signal = 1;

    write_text("\n[fatal-backtrace-v15] BEGIN signal=");
    write_text(signal_name(signal_number));
    write_text(" signo=");
    write_unsigned(static_cast<uint64_t>(signal_number), 10);
    write_text(" si_code=");
    write_unsigned(static_cast<uint64_t>(info ? info->si_code : 0), 10);
    write_text(" si_addr=0x");
    write_unsigned(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(info ? info->si_addr : nullptr)), 16);
    write_text(" tid=");
    write_unsigned(current_thread_id(), 10);
    write_text("\n[fatal-backtrace-v15] backtrace follows\n");

    void * frames[kFrameLimit];
    const int frame_count = ::backtrace(frames, static_cast<int>(kFrameLimit));
    if (frame_count > 0) ::backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);

    write_text("[fatal-backtrace-v15] END exiting=");
    write_unsigned(static_cast<uint64_t>(128 + signal_number), 10);
    write_text("\n");
    ::_exit(128 + signal_number);
}

inline bool install_current_thread_altstack() {
    alignas(16) static thread_local unsigned char altstack[kAltStackBytes];
    stack_t stack{};
    stack.ss_sp = altstack;
    stack.ss_size = sizeof(altstack);
    stack.ss_flags = 0;
    return ::sigaltstack(&stack, nullptr) == 0;
}

inline bool prewarm_unwinder() {
    void * frames[8];
    const int frame_count = ::backtrace(frames, static_cast<int>(sizeof(frames) / sizeof(frames[0])));
    const int null_fd = ::open("/dev/null", O_WRONLY);
    if (frame_count > 0 && null_fd >= 0) ::backtrace_symbols_fd(frames, frame_count, null_fd);
    if (null_fd >= 0) (void) ::close(null_fd);
    return frame_count > 0;
}

inline bool install_handlers() {
    struct sigaction action{};
    action.sa_sigaction = fatal_signal_handler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;
    if (sigemptyset(&action.sa_mask) != 0) return false;
    if (::sigaction(SIGSEGV, &action, nullptr) != 0) return false;
    if (::sigaction(SIGBUS, &action, nullptr) != 0) return false;
    return true;
}

inline bool prepare_current_thread() {
    return install_current_thread_altstack() && prewarm_unwinder() && install_handlers();
}

} // namespace fatal_backtrace_v15
