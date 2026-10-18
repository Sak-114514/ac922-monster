#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

// Slot bookkeeping uses full server-batch rows; a decode view exposes local rows.
// A speculative window must fit in one view. Preserve that existing constraint.
inline std::vector<int32_t> server_spec_local_output_rows(
        const std::vector<int32_t> & global, int32_t offset, int32_t count) {
    if (offset < 0 || count < 0) throw std::invalid_argument("invalid speculative decode view");
    const int64_t end = (int64_t) offset + count;
    std::vector<int32_t> local;
    local.reserve(global.size());
    for (int32_t row : global) {
        if (row < offset || (int64_t) row >= end) {
            throw std::out_of_range("speculative output row is outside its decode view");
        }
        local.push_back(row-offset);
    }
    return local;
}
