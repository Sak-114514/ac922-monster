#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// row_ids maps each execution row to its original input row.
inline bool llama_mtp_restore_row_order(
        const std::vector<int32_t> & row_ids, size_t n_rows, size_t width,
        float * data, std::vector<float> & scratch) {
    if (row_ids.size() != n_rows || width == 0 || (n_rows > 0 && data == nullptr)) {
        throw std::invalid_argument("invalid MTP row permutation dimensions");
    }
    std::vector<bool> seen(n_rows, false);
    bool identity = true;
    for (size_t row = 0; row < n_rows; ++row) {
        const int32_t original = row_ids[row];
        if (original < 0 || (size_t) original >= n_rows || seen[original]) {
            throw std::invalid_argument("invalid MTP row permutation");
        }
        seen[original] = true;
        identity = identity && (size_t) original == row;
    }
    if (identity) {
        return false;
    }
    scratch.resize(n_rows * width);
    std::memcpy(scratch.data(), data, n_rows * width * sizeof(float));
    for (size_t row = 0; row < n_rows; ++row) {
        std::memcpy(data + (size_t) row_ids[row] * width,
                scratch.data() + row * width, width * sizeof(float));
    }
    return true;
}
