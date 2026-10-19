#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <poll.h>
#include <set>
#include <unistd.h>

// Only the coordinator changes context geometry. Bank owners emit tokens.
class production_ipc {
    int fd_ = -1;
    std::mutex mutex_, output_mutex_;
    std::condition_variable cv_;
    std::deque<json> waves_;
    std::set<std::string> cancelled_;
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
                        } else if (type == "wave" && waves_.empty()) {
                            waves_.push_back(std::move(command));
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
              {"max_parallel", 4}, {"max_context", max_context}, {"pool_tokens", pool_tokens}});
    }
    bool next(json & wave) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return closing_ || !waves_.empty(); });
        if (closing_) return false;
        wave = std::move(waves_.front());
        waves_.pop_front();
        return true;
    }
    bool cancelled(const std::string & id) {
        if (!serving) return false;
        if (closing_) return true;
        std::lock_guard<std::mutex> lock(mutex_);
        return cancelled_.count(id) != 0;
    }
    void finish_wave(const std::string & id, const json & timing=json::object()) {
        emit({{"type", "wave_done"}, {"id", id}, {"timings", timing}});
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_.clear();
    }
    void token(const std::string & id, const llama_vocab * vocab, llama_token tok) {
        if (!serving || llama_vocab_is_eog(vocab, tok)) return;
        const auto piece = common_token_to_piece(vocab, tok, false);
        std::vector<uint8_t> bytes(piece.begin(), piece.end());
        emit({{"type", "token"}, {"id", id}, {"token_id", tok}, {"piece_bytes", bytes}});
    }
};
static production_ipc production;
