#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace production_shape {
constexpr uint32_t warm_cells = 17152;
constexpr uint32_t idle_cells = 1024;
using capacities = std::array<uint32_t, 4>;
inline uint64_t total(const capacities & values) {
    uint64_t result=0;
    for (auto value: values) result+=value;
    return result;
}
inline capacities choose(const capacities & needed, uint32_t pool) {
    capacities exact{}, preferred{};
    for (size_t i=0;i<needed.size();++i) {
        exact[i]=std::max(idle_cells,needed[i]);
        preferred[i]=std::max(warm_cells,exact[i]);
    }
    if (total(preferred)>pool) {
        for (size_t i=0;i<needed.size();++i) if (!needed[i]) preferred[i]=idle_cells;
    }
    if (total(preferred)>pool) preferred=exact;
    if (total(preferred)>pool) throw std::runtime_error("shared context pool exceeded");
    return preferred;
}
inline bool rebuild(const capacities & existing, const capacities & desired, uint32_t pool) {
    for (size_t i=0;i<existing.size();++i) if (existing[i]<desired[i]) return true;
    return total(existing)>pool || total(existing)>2*total(desired);
}
}
