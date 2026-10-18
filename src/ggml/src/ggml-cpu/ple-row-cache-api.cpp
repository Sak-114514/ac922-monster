#include "ple-row-cache-api.h"

#if defined(__linux__)
namespace ggml_ple_io {
std::shared_ptr<row_cache> acquire(const ggml_tensor * table) {
    source_info source;
    if (!mapped_source(table, source)) return {};
    static std::mutex mutex;
    static std::shared_ptr<row_cache> cache;
    std::lock_guard<std::mutex> lock(mutex);
    if (!cache || !(cache->info == source)) {
        cache = std::make_shared<row_cache>(std::move(source), table->nb[1]);
    }
    return cache;
}
} // namespace ggml_ple_io
#endif
