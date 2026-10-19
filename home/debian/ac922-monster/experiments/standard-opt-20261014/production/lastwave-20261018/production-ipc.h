#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <poll.h>
#include <set>
#include <string>
#include <thread>
#include <stdexcept>
#include <unistd.h>

// Only the coordinator changes context geometry. Bank owners emit tokens.
class production_ipc {
    int fd_ = -1;
    std::mutex mutex_, output_mutex_;
    std::condition_variable cv_;
    std::deque<json> waves_;
    std::deque<json> refills_;
    std::set<std::string> cancelled_;
    std::string active_wave_id_;
    bool worker_closing_wave_ = false;
    bool close_requested_ = false;
    std::atomic<bool> closing_{false};
    std::thread reader_;
public:
    bool serving = false;
    ~production_ipc() {
        closing_ = true;
        cv_.notify_all();
        if (reader_.joinable()) reader_.join();
    }
    bool enabled() const { return fd_ >= 0; }
    void configure() {
        const char * value = std::getenv("AC922_API_EVENT_FD");
        if (!value) return;
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        require(end != value && *end == '\0' && parsed >= 3 && parsed < 1048576,
                "invalid production event fd");
        fd_ = (int) parsed;
    }
    void emit(const json & event) {
        if (!enabled()) return;
        const auto wire = event.dump(-1, ' ', false, json::error_handler_t::replace) + "\n";
        std::lock_guard<std::mutex> lock(output_mutex_);
        size_t sent = 0;
        while (sent < wire.size()) {
            const auto n = ::write(fd_, wire.data() + sent, wire.size() - sent);
            if (n < 0 && errno == EINTR) continue;
            require(n > 0, "production event pipe closed");
            sent += (size_t) n;
        }
    }
    void start(int max_context, int pool_tokens) {
        serving = true;
        const char * prefix_cache_env = std::getenv("AC922_PREFIX_CACHE");
        const bool prefix_cache_enabled = prefix_cache_env != nullptr &&
                prefix_cache_env[0] == '1' && prefix_cache_env[1] == '\0';
        reader_ = std::thread([this] {
            std::string pending;
            while (!closing_) {
                pollfd pfd{STDIN_FILENO, POLLIN, 0};
                const int status = ::poll(&pfd, 1, 100);
                if (status < 0 && errno == EINTR) continue;
                if (status == 0) continue;
                if (status < 0) break;
                char buffer[8192];
                const auto count = ::read(STDIN_FILENO, buffer, sizeof(buffer));
                if (count <= 0) break;
                pending.append(buffer, (size_t) count);
                if (pending.size() > 8 * 1024 * 1024) break;
                size_t newline;
                while ((newline = pending.find('\n')) != std::string::npos) {
                    const std::string line = pending.substr(0, newline);
                    pending.erase(0, newline + 1);
                    try {
                        auto command = json::parse(line);
                        const auto type = command.at("type").get<std::string>();
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (type == "cancel") {
                            if (cancelled_.size() < 64) cancelled_.insert(command.at("id").get<std::string>());
                        } else if (type == "wave" && active_wave_id_.empty() && waves_.empty()) {
                            waves_.push_back(std::move(command));
                            cv_.notify_all();
                        } else if (type == "refill") {
                            const auto wave_id = command.at("id").get<std::string>();
                            require(!active_wave_id_.empty() && wave_id == active_wave_id_ &&
                                    !close_requested_ && refills_.size() < 16,
                                    "refill does not match the open worker wave");
                            refills_.push_back(std::move(command));
                            cv_.notify_all();
                        } else if (type == "close_wave") {
                            const auto wave_id = command.at("id").get<std::string>();
                            require(!active_wave_id_.empty() && wave_id == active_wave_id_ &&
                                    worker_closing_wave_ && !close_requested_,
                                    "close_wave does not match a closing worker wave");
                            // Earlier refill commands are already in refills_ because
                            // this reader consumes stdin lines in order. The coordinator
                            // drains them before close_requested() becomes true.
                            close_requested_ = true;
                            cv_.notify_all();
                        } else if (type == "shutdown") {
                            closing_ = true;
                            cv_.notify_all();
                        } else {
                            throw std::runtime_error("invalid or overlapping worker command");
                        }
                    } catch (const std::exception &) {
                        closing_ = true;
                        cv_.notify_all();
                    }
                }
            }
            closing_ = true;
            cv_.notify_all();
        });
        emit({{"type", "ready"}, {"model", "flash-next-iq3-xxs"},
              {"refill_supported", true},
              {"prefix_cache_enabled", prefix_cache_enabled},
              {"max_parallel", 4}, {"max_context", max_context}, {"pool_tokens", pool_tokens}});
    }
    bool next(json & wave) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return closing_ || !waves_.empty(); });
        if (closing_) return false;
        wave = std::move(waves_.front());
        waves_.pop_front();
        active_wave_id_ = wave.at("id").get<std::string>();
        worker_closing_wave_ = false;
        close_requested_ = false;
        refills_.clear();
        return true;
    }
    // Announces that all currently-owned banks have drained. The API either
    // sends more refills first or acknowledges close_wave; its stdin ordering
    // is preserved by the reader thread above.
    void request_wave_close(const std::string & wave_id) {
        bool shutting_down = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            require(!active_wave_id_.empty() && wave_id == active_wave_id_ && !worker_closing_wave_,
                    "cannot close an inactive or already-closing wave");
            worker_closing_wave_ = true;
            shutting_down = closing_.load();
        }
        if (shutting_down) return;
        emit({{"type", "wave_closing"}, {"id", wave_id}});
    }
    // Always drain queued refills before consulting close_requested(). This is
    // the ordering guarantee that closes the bank_idle / wave_closing race.
    bool poll_refill(const std::string & wave_id, json & command) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_wave_id_.empty() || wave_id != active_wave_id_ || refills_.empty()) return false;
        command = std::move(refills_.front());
        refills_.pop_front();
        cv_.notify_all();
        return true;
    }
    // True only after close_wave arrived and all preceding refill commands
    // have been handed to the coordinator.
    bool close_requested(const std::string & wave_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return !active_wave_id_.empty() && wave_id == active_wave_id_ &&
                refills_.empty() && (closing_.load() || close_requested_);
    }
    bool cancelled(const std::string & id) {
        if (!serving) return false;
        if (closing_) return true;
        std::lock_guard<std::mutex> lock(mutex_);
        return cancelled_.count(id) != 0;
    }
    void finish_request(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_.erase(id);
    }
    void finish_wave(const std::string & id, const json & timing=json::object()) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            require(!active_wave_id_.empty() && active_wave_id_ == id && refills_.empty(),
                    "finish_wave before matching wave drain or refill drain");
            if (!worker_closing_wave_) {
                // Legacy clients omit refill=true, so the worker may finish a
                // wave without the new close handshake when no refill is used.
                require(!close_requested_, "legacy wave has unexpected close acknowledgement");
            } else {
                require(close_requested_ || closing_.load(),
                        "refill-capable wave finished before close acknowledgement");
            }
            active_wave_id_.clear();
            worker_closing_wave_ = false;
            close_requested_ = false;
            // No owner or refill remains, and the next wave cannot yet be
            // admitted under this mutex. Retire any late cancel for completed
            // requests so repeated client disconnects cannot fill the bound.
            cancelled_.clear();
        }
        emit({{"type", "wave_done"}, {"id", id}, {"timings", timing}});
    }
    void token(const std::string & id, const llama_vocab * vocab, llama_token tok) {
        if (!serving || llama_vocab_is_eog(vocab, tok)) return;
        const auto piece = common_token_to_piece(vocab, tok, false);
        std::vector<uint8_t> bytes(piece.begin(), piece.end());
        emit({{"type", "token"}, {"id", id}, {"token_id", tok}, {"piece_bytes", bytes}});
    }
};
static production_ipc production;
