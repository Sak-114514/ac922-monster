#pragma once

#if defined(__linux__)
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

namespace ggml_ple_io {
struct source_info {
    const char * base = nullptr;
    size_t bytes = 0;
    uint64_t offset = 0;
    dev_t device = 0;
    ino_t inode = 0;
    off_t file_size = 0;
    timespec modified{};
    std::string path;
    bool operator==(const source_info & rhs) const {
        return base == rhs.base && bytes == rhs.bytes && offset == rhs.offset && device == rhs.device &&
            inode == rhs.inode && file_size == rhs.file_size && modified.tv_sec == rhs.modified.tv_sec &&
            modified.tv_nsec == rhs.modified.tv_nsec && path == rhs.path;
    }
};

inline bool mapped_source(const ggml_tensor * tensor, source_info & out) {
    if (!tensor || !tensor->data) return false;
    const uintptr_t address = reinterpret_cast<uintptr_t>(tensor->data);
    const size_t bytes = ggml_nbytes(tensor);
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long begin = 0, end = 0;
        unsigned long long offset = 0, inode = 0;
        unsigned major_id = 0, minor_id = 0;
        char perms[5]{};
        int used = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx %4s %llx %x:%x %llu %n", &begin, &end, perms,
                &offset, &major_id, &minor_id, &inode, &used) != 7) continue;
        if (address < begin || address >= end || bytes > end - address) continue;
        std::string path = line.substr(used);
        const size_t first = path.find_first_not_of(' ');
        if (first == std::string::npos) return false;
        path.erase(0, first);
        if (path.empty() || path[0] != '/' || path.find(" (deleted)") != std::string::npos || path.find('\\') != std::string::npos) return false;
        struct stat st{};
        if (stat(path.c_str(), &st) || !S_ISREG(st.st_mode) || st.st_ino != inode ||
                major(st.st_dev) != major_id || minor(st.st_dev) != minor_id) return false;
        const uint64_t file_offset = offset + address - begin;
        if (file_offset > uint64_t(st.st_size) || bytes > uint64_t(st.st_size) - file_offset) return false;
        out = {reinterpret_cast<const char *>(address), bytes, file_offset, st.st_dev, st.st_ino,
            st.st_size, st.st_mtim, std::move(path)};
        return true;
    }
    return false;
}

struct aligned_scratch {
    void * data = nullptr;
    aligned_scratch() { if (posix_memalign(&data, 65536, 8192)) data = nullptr; }
    ~aligned_scratch() { std::free(data); }
};

struct aligned_span {
    uint64_t offset = 0;
    size_t bytes = 0;
    size_t skip = 0;
};

struct warm_stats {
    size_t rows_warmed = 0;
    size_t rows_cached = 0;
    bool budget_exhausted = false;
    uint64_t direct_bytes = 0;
    uint64_t direct_reads = 0;
};

inline aligned_span make_aligned_span(uint64_t offset, size_t bytes, size_t alignment = 4096) {
    const uint64_t start = offset & ~uint64_t(alignment - 1);
    const size_t skip = size_t(offset - start);
    const size_t length = ((skip + bytes + alignment - 1) / alignment) * alignment;
    return {start, length, skip};
}

class row_cache {
    static constexpr size_t n_shards = 64, ways = 4;
    const size_t n_sets;
    static size_t configured_sets() {
        const char * env = getenv("GGML_CPU_PLE_ROW_CACHE_ENTRIES");
        if (!env || !*env) return 1024;
        char * end = nullptr;
        const unsigned long long entries = std::strtoull(env, &end, 10);
        if (end == env || *end != '\0' || (entries != 262144 && entries != 524288)) return 1024;
        return static_cast<size_t>(entries) / (n_shards * ways);
    }
    struct entry {
        uint64_t row = UINT64_MAX, age = 0;
        std::array<uint8_t, 128> data{};
    };
    struct shard {
        std::mutex mutex;
        std::vector<entry> entries;
        uint64_t clock = 0;
    };
    std::array<shard, n_shards> shards;
    int fd = -1;
    size_t row_bytes = 0, page_size = 0;
    std::atomic<bool> direct_ok{false};
    std::atomic<uint64_t> hits{0}, resident{0}, direct{0}, fallback{0}, io_bytes{0};

    bool find(uint64_t row, uint8_t * result, bool count_hit) {
        const uint64_t hash = row * UINT64_C(11400714819323198485);
        auto & part = shards[hash & (n_shards - 1)];
        const size_t start = ((hash >> 6) & (n_sets - 1)) * ways;
        std::lock_guard<std::mutex> lock(part.mutex);
        if (part.entries.empty()) part.entries.resize(n_sets * ways);
        for (size_t i = start; i < start + ways; ++i) if (part.entries[i].row == row) {
            if (result) std::memcpy(result, part.entries[i].data.data(), row_bytes);
            part.entries[i].age = ++part.clock;
            if (count_hit) hits.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    void insert(uint64_t row, const uint8_t * data) {
        const uint64_t hash = row * UINT64_C(11400714819323198485);
        auto & part = shards[hash & (n_shards - 1)];
        const size_t start = ((hash >> 6) & (n_sets - 1)) * ways;
        std::lock_guard<std::mutex> lock(part.mutex);
        if (part.entries.empty()) part.entries.resize(n_sets * ways);
        size_t victim = start;
        for (size_t i = start; i < start + ways; ++i) {
            if (part.entries[i].row == row) { victim = i; break; }
            if (part.entries[i].age < part.entries[victim].age) victim = i;
        }
        auto & slot = part.entries[victim];
        slot.row = row;
        slot.age = ++part.clock;
        std::memcpy(slot.data.data(), data, row_bytes);
    }

    bool read_mapped(uint64_t row, uint8_t * result) {
        const char * source = info.base + row * row_bytes;
        std::memcpy(result, source, row_bytes);
        fallback.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

public:
    const source_info info;
    row_cache(source_info source, size_t width) : n_sets(configured_sets()), row_bytes(width), page_size(sysconf(_SC_PAGESIZE)), info(std::move(source)) {
        fd = open(info.path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd >= 0) {
            struct stat st{};
            if (fstat(fd, &st) || st.st_ino != info.inode || st.st_dev != info.device ||
                    st.st_size != info.file_size || st.st_mtim.tv_sec != info.modified.tv_sec || st.st_mtim.tv_nsec != info.modified.tv_nsec) {
                close(fd); fd = -1;
            }
        }
        direct_ok.store(fd >= 0);
    }
    ~row_cache() {
        if (fd >= 0) close(fd);
    }

    size_t row_width() const { return row_bytes; }
    size_t row_capacity() const { return n_sets * n_shards * ways; }
    uint64_t n_rows() const { return row_bytes ? info.bytes / row_bytes : 0; }
    bool direct_io_enabled() const { return direct_ok.load(std::memory_order_relaxed); }
    uint64_t direct_read_count() const { return direct.load(std::memory_order_relaxed); }
    uint64_t resident_read_count() const { return resident.load(std::memory_order_relaxed); }
    uint64_t fallback_read_count() const { return fallback.load(std::memory_order_relaxed); }
    uint64_t hit_count() const { return hits.load(std::memory_order_relaxed); }

    uint64_t read(uint64_t row, uint8_t * result) {
        if (!result || row >= n_rows() || row_bytes == 0 || row_bytes > 128) return 0;
        if (find(row, result, true)) return 0;

        const char * source = info.base + row * row_bytes;
        const uintptr_t address = reinterpret_cast<uintptr_t>(source);
        const uintptr_t first = address & ~(page_size - 1);
        const uintptr_t last = (address + row_bytes + page_size - 1) & ~(page_size - 1);
        unsigned char residency[2]{};
        const bool ready = mincore(reinterpret_cast<void *>(first), last - first, residency) == 0 &&
            (residency[0] & 1) && (last - first == page_size || (residency[1] & 1));
        bool read_ok = false;
        uint64_t direct_bytes = 0;
        if (ready) {
            std::memcpy(result, source, row_bytes);
            resident.fetch_add(1, std::memory_order_relaxed);
            read_ok = true;
        } else if (direct_ok.load(std::memory_order_relaxed)) {
            thread_local aligned_scratch scratch;
            const uint64_t offset = info.offset + row * row_bytes;
            const aligned_span span = make_aligned_span(offset, row_bytes);
            ssize_t size = -1;
            if (scratch.data && span.bytes <= 8192) {
                do { size = pread(fd, scratch.data, span.bytes, off_t(span.offset)); } while (size < 0 && errno == EINTR);
            }
            if (size >= ssize_t(span.skip + row_bytes)) {
                std::memcpy(result, static_cast<const char *>(scratch.data) + span.skip, row_bytes);
                direct.fetch_add(1, std::memory_order_relaxed);
                io_bytes.fetch_add(size, std::memory_order_relaxed);
                direct_bytes = uint64_t(size);
                read_ok = true;
            } else if (size < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) {
                direct_ok.store(false, std::memory_order_relaxed);
            }
        }
        if (!read_ok) read_mapped(row, result);
        insert(row, result);
        return direct_bytes;
    }

    // Warms the exact cache instance used by ordinary GGML GET_ROWS. The
    // conservative byte budget charges the aligned direct-I/O span per row,
    // even when a cache hit would avoid I/O.
    warm_stats warm_rows(const std::vector<uint64_t> & rows, size_t max_bytes, const std::atomic<bool> * cancelled) {
        warm_stats result;
        if (row_bytes == 0 || row_bytes > 128) return result;
        std::array<uint8_t, 128> packed{};
        size_t charged = 0;
        for (uint64_t row : rows) {
            if ((cancelled && cancelled->load(std::memory_order_acquire)) || row >= n_rows()) break;
            if (find(row, nullptr, false)) {
                ++result.rows_cached;
                continue;
            }
            const aligned_span span = make_aligned_span(info.offset + row * row_bytes, row_bytes);
            if (span.bytes > max_bytes - std::min(charged, max_bytes)) {
                result.budget_exhausted = true;
                break;
            }
            charged += span.bytes;
            const uint64_t direct_bytes = read(row, packed.data());
            ++result.rows_warmed;
            if (direct_bytes) {
                ++result.direct_reads;
                result.direct_bytes += direct_bytes;
            }
        }
        return result;
    }

    uint64_t io_byte_count() const { return io_bytes.load(std::memory_order_relaxed); }
};

// Out-of-line registry entry point: both llama and the CPU GET_ROWS backend
// must resolve to one process-wide instance even when built into separate DSOs.
GGML_API std::shared_ptr<row_cache> acquire(const ggml_tensor * table);
} // namespace ggml_ple_io
#endif
