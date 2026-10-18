#pragma once

#if defined(__linux__)
#include "ple-row-cache-api.h"

static bool ggml_cpu_ple_cached_rows(const ggml_compute_params * params, ggml_tensor * dst, ggml_to_float_t dequantize) {
#if defined(__linux__)
    static const bool enabled = [] { const char * value = getenv("GGML_CPU_PLE_ROW_CACHE"); return value && atoi(value) != 0; }();
    const auto * table = dst->src[0];
    const auto * indices = dst->src[1];
    if (!enabled || table->type != GGML_TYPE_IQ4_NL || table->ne[0] != 160 || table->ne[2] != 1 || table->ne[3] != 1 ||
            strcmp(table->name, "per_layer_token_embd.weight") || ggml_nbytes(table) <= 64*1024*1024 ||
            !ggml_is_contiguous(table) || !ggml_is_contiguous(indices) || !ggml_is_contiguous(dst)) return false;
    using shared_cache = std::shared_ptr<ggml_ple_io::row_cache>;
    static_assert(sizeof(shared_cache) <= 64, "CPU graph workspace must fit shared cache ownership");
    GGML_ASSERT(params->wdata && params->wsize >= sizeof(shared_cache));
    auto * shared = static_cast<shared_cache *>(params->wdata);
    if (params->ith == 0) new (shared) shared_cache(ggml_ple_io::acquire(table));
    ggml_barrier(params->threadpool);
    auto cache = *shared;
    if (cache) {
        const int64_t count = ggml_nelements(indices);
        const int64_t per_thread = (count + params->nth - 1) / params->nth;
        const int64_t end = std::min(count, per_thread * (params->ith + 1));
        alignas(16) uint8_t packed[128];
        for (int64_t i = per_thread * params->ith; i < end; ++i) {
            const int32_t row = static_cast<const int32_t *>(indices->data)[i];
            GGML_ASSERT(row >= 0 && row < table->ne[1]);
            cache->read(row, packed);
            dequantize(packed, reinterpret_cast<float *>(static_cast<char *>(dst->data) + i * dst->nb[1]), 160);
        }
    }
    ggml_barrier(params->threadpool);
    if (params->ith == 0) shared->~shared_cache();
    ggml_barrier(params->threadpool);
    return bool(cache);
#else
    GGML_UNUSED(params); GGML_UNUSED(dst); GGML_UNUSED(dequantize);
    return false;
#endif
}

#endif // __linux__
