#pragma once

#include <cstddef>

namespace v35_verify_row_budget {

inline constexpr std::size_t capacity(std::size_t lanes, std::size_t draft_depth) {
    return lanes * (draft_depth + 1);
}

inline constexpr bool accepts(std::size_t rows, std::size_t lanes, std::size_t draft_depth) {
    return rows <= capacity(lanes, draft_depth);
}

} // namespace v35_verify_row_budget
