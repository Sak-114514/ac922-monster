#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

inline int32_t server_prefill_batch_limit(int32_t capacity, int32_t decode_tokens, int32_t budget) {
    capacity = std::max<int32_t>(0, capacity);
    decode_tokens = std::max<int32_t>(0, std::min(capacity, decode_tokens));
    if (decode_tokens == 0 || budget <= 0) {
        return capacity;
    }
    return decode_tokens + std::min(budget, capacity - decode_tokens);
}

inline bool server_cache_budget_fits(size_t used, size_t added, size_t limit) {
    return limit == 0 || (used <= limit && added <= limit - used);
}

// Leave at least one block to replay for an exact repeat, without moving older
// prefix block boundaries when a suffix is appended.
inline size_t server_prompt_checkpoint_boundary(size_t tokens, size_t block_size, size_t min_replay = 1) {
    min_replay = std::max<size_t>(1, min_replay);
    return tokens < min_replay || block_size == 0 ? 0 : ((tokens - min_replay) / block_size) * block_size;
}
