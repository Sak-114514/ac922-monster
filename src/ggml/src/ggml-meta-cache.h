#pragma once
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

// Lossless storage only; the dependency walker still uses the full public descriptor.
struct ggml_meta_cached_split {
    ggml_backend_meta_split_axis axis = GGML_BACKEND_SPLIT_AXIS_NONE;
    uint32_t n_segments = 0;
    std::array<uint32_t, 16> nr{};
    std::vector<int64_t> ne;

    void assign(const ggml_backend_meta_split_state & state, size_t members) {
        axis = state.axis;
        n_segments = state.n_segments;
        std::copy(std::begin(state.nr), std::end(state.nr), nr.begin());
        size_t count = sizeof(state.ne) / sizeof(state.ne[0]);
        static const bool packed = [] { const char * s = getenv("GGML_META_PACKED_CACHE"); return s && atoi(s) != 0; }();
        if (packed) {
            const size_t active = size_t(n_segments) * members;
            GGML_ASSERT(active <= count);
            while (count > active && state.ne[count - 1] == 0) --count;
        }
        ne.assign(state.ne, state.ne + count);
    }

    operator ggml_backend_meta_split_state() const {
        ggml_backend_meta_split_state state{};
        state.axis = axis;
        state.n_segments = n_segments;
        std::copy(nr.begin(), nr.end(), state.nr);
        std::copy(ne.begin(), ne.end(), state.ne);
        return state;
    }
};
