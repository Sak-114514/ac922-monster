#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

struct common_speculative_mtp_layout {
    std::vector<std::vector<int32_t>> rows;
    std::vector<int32_t> previous;
};

struct common_speculative_mtp_deferred {
    std::vector<int32_t> tokens;
    std::vector<int32_t> positions;
    std::vector<float> hidden;
    int32_t keep = -1;
    bool draft_pending = false;

    void clear() {
        tokens.clear();
        positions.clear();
        hidden.clear();
        keep = -1;
        draft_pending = false;
    }

    int32_t kept_rows() const {
        return keep < 0 ? static_cast<int32_t>(tokens.size())
                : std::min<int32_t>(keep, static_cast<int32_t>(tokens.size()));
    }
};

// Bounded opaque state for the generic per-sequence MTP path. The wire format is
// deliberately independent of the in-memory struct layout and is versioned so a
// stale/incompatible checkpoint is rejected before the caller mutates live state.
struct common_speculative_mtp_snapshot {
    int32_t seq_id = -1;
    int32_t n_seq = 0;
    int32_t n_embd = 0;
    int32_t n_max = 0;
    int32_t n_mtp_layers = 0;
    bool fold_multi = false;

    std::vector<float> pending_h;
    common_speculative_mtp_deferred deferred;
};

namespace common_speculative_mtp_snapshot_wire {

static constexpr uint32_t max_sequences = 4096;
static constexpr uint32_t max_embedding_width = 262144;
static constexpr uint32_t max_deferred_rows = 16384;
static constexpr size_t   max_bytes = 64 * 1024 * 1024;

// Stable byte offsets used by the format and by corruption fixtures.
static constexpr size_t magic_offset = 0;
static constexpr size_t version_offset = 4;
static constexpr size_t flags_offset = 6;
static constexpr size_t seq_id_offset = 8;
static constexpr size_t n_seq_offset = 12;
static constexpr size_t n_embd_offset = 16;
static constexpr size_t n_max_offset = 20;
static constexpr size_t n_mtp_layers_offset = 24;
static constexpr size_t n_rows_offset = 28;
static constexpr size_t keep_offset = 32;
static constexpr size_t draft_pending_offset = 36;
static constexpr size_t header_size = 40;
static constexpr size_t checksum_size = 8;

inline void append_u16(std::vector<uint8_t> & data, uint16_t value) {
    data.push_back((uint8_t) (value & 0xff));
    data.push_back((uint8_t) ((value >> 8) & 0xff));
}

inline void append_u32(std::vector<uint8_t> & data, uint32_t value) {
    for (unsigned int i = 0; i < 4; ++i) {
        data.push_back((uint8_t) ((value >> (8 * i)) & 0xff));
    }
}

inline void append_u64(std::vector<uint8_t> & data, uint64_t value) {
    for (unsigned int i = 0; i < 8; ++i) {
        data.push_back((uint8_t) ((value >> (8 * i)) & 0xff));
    }
}

inline uint16_t read_u16(const std::vector<uint8_t> & data, size_t offset) {
    if (offset > data.size() || data.size() - offset < 2) {
        throw std::invalid_argument("truncated MTP snapshot header");
    }
    return (uint16_t) data[offset] | ((uint16_t) data[offset + 1] << 8);
}

inline uint32_t read_u32(const std::vector<uint8_t> & data, size_t offset) {
    if (offset > data.size() || data.size() - offset < 4) {
        throw std::invalid_argument("truncated MTP snapshot header");
    }
    return (uint32_t) data[offset] |
            ((uint32_t) data[offset + 1] << 8) |
            ((uint32_t) data[offset + 2] << 16) |
            ((uint32_t) data[offset + 3] << 24);
}

inline uint64_t read_u64(const std::vector<uint8_t> & data, size_t offset) {
    if (offset > data.size() || data.size() - offset < 8) {
        throw std::invalid_argument("truncated MTP snapshot checksum");
    }
    uint64_t result = 0;
    for (unsigned int i = 0; i < 8; ++i) {
        result |= (uint64_t) data[offset + i] << (8 * i);
    }
    return result;
}

inline uint64_t checksum(const std::vector<uint8_t> & data, size_t size) {
    uint64_t result = 14695981039346656037ull;
    for (size_t i = 0; i < size; ++i) {
        result ^= data[i];
        result *= 1099511628211ull;
    }
    return result;
}

inline void validate_metadata(const common_speculative_mtp_snapshot & state) {
    static_assert(sizeof(float) == 4, "MTP snapshot requires 32-bit float rows");
    if (state.seq_id < 0 || state.n_seq <= 1 || (uint32_t) state.n_seq > max_sequences ||
            state.seq_id >= state.n_seq || state.n_embd <= 0 ||
            (uint32_t) state.n_embd > max_embedding_width || state.n_max < 0 ||
            (uint32_t) state.n_max >= max_deferred_rows || state.n_mtp_layers != 1) {
        throw std::invalid_argument("unsupported MTP snapshot dimensions");
    }
}

inline void validate_shape(const common_speculative_mtp_snapshot & state) {
    validate_metadata(state);

    const size_t rows = state.deferred.tokens.size();
    if (rows > max_deferred_rows || rows > (size_t) state.n_max + 1 ||
            state.deferred.positions.size() != rows ||
            state.deferred.hidden.size() != rows * (size_t) state.n_embd ||
            state.pending_h.size() != (size_t) state.n_embd ||
            state.deferred.keep < -1 || state.deferred.keep > (int32_t) rows) {
        throw std::invalid_argument("invalid MTP snapshot row dimensions");
    }

    const size_t bytes = header_size +
            (size_t) state.n_embd * sizeof(float) +
            rows * (2 * sizeof(int32_t) + (size_t) state.n_embd * sizeof(float)) +
            checksum_size;
    if (bytes > max_bytes) {
        throw std::invalid_argument("MTP snapshot exceeds byte limit");
    }
}

inline std::vector<uint8_t> encode(const common_speculative_mtp_snapshot & state) {
    validate_shape(state);

    std::vector<uint8_t> data;
    const size_t rows = state.deferred.tokens.size();
    data.reserve(header_size + (size_t) state.n_embd * sizeof(float) +
            rows * (2 * sizeof(int32_t) + (size_t) state.n_embd * sizeof(float)) + checksum_size);
    data.insert(data.end(), {'M', 'T', 'P', 'S'});
    append_u16(data, 1); // format version
    append_u16(data, state.fold_multi ? 1 : 0);
    append_u32(data, (uint32_t) state.seq_id);
    append_u32(data, (uint32_t) state.n_seq);
    append_u32(data, (uint32_t) state.n_embd);
    append_u32(data, (uint32_t) state.n_max);
    append_u32(data, (uint32_t) state.n_mtp_layers);
    append_u32(data, (uint32_t) rows);
    append_u32(data, (uint32_t) state.deferred.keep);
    append_u32(data, state.deferred.draft_pending ? 1 : 0);

    const size_t pending_bytes = (size_t) state.n_embd * sizeof(float);
    const size_t vector_bytes = rows * sizeof(int32_t);
    const size_t hidden_bytes = rows * (size_t) state.n_embd * sizeof(float);
    const size_t old_size = data.size();
    data.resize(old_size + pending_bytes + 2 * vector_bytes + hidden_bytes);
    std::memcpy(data.data() + old_size, state.pending_h.data(), pending_bytes);
    size_t offset = old_size + pending_bytes;
    if (vector_bytes != 0) {
        std::memcpy(data.data() + offset, state.deferred.tokens.data(), vector_bytes);
        offset += vector_bytes;
        std::memcpy(data.data() + offset, state.deferred.positions.data(), vector_bytes);
        offset += vector_bytes;
        std::memcpy(data.data() + offset, state.deferred.hidden.data(), hidden_bytes);
    }
    append_u64(data, checksum(data, data.size()));
    return data;
}

inline void reset_state(
        std::vector<float> & pending_h,
        common_speculative_mtp_deferred & deferred) {
    std::fill(pending_h.begin(), pending_h.end(), 0.0f);
    deferred.clear();
}

inline void decode(
        const std::vector<uint8_t> & data,
        const common_speculative_mtp_snapshot & expected,
        common_speculative_mtp_snapshot & destination) {
    validate_metadata(expected);
    if (data.size() < header_size || data.size() > max_bytes) {
        throw std::invalid_argument("invalid MTP snapshot byte length");
    }
    if (data[0] != 'M' || data[1] != 'T' || data[2] != 'P' || data[3] != 'S') {
        throw std::invalid_argument("invalid MTP snapshot magic");
    }
    if (read_u16(data, version_offset) != 1) {
        throw std::invalid_argument("unsupported MTP snapshot version");
    }
    const uint16_t flags = read_u16(data, flags_offset);
    const uint32_t seq_id = read_u32(data, seq_id_offset);
    const uint32_t n_seq = read_u32(data, n_seq_offset);
    const uint32_t n_embd = read_u32(data, n_embd_offset);
    const uint32_t n_max = read_u32(data, n_max_offset);
    const uint32_t n_mtp_layers = read_u32(data, n_mtp_layers_offset);
    const uint32_t n_rows = read_u32(data, n_rows_offset);
    const int32_t keep = (int32_t) read_u32(data, keep_offset);
    const uint32_t draft_pending = read_u32(data, draft_pending_offset);

    common_speculative_mtp_snapshot parsed;
    parsed.seq_id = (int32_t) seq_id;
    parsed.n_seq = (int32_t) n_seq;
    parsed.n_embd = (int32_t) n_embd;
    parsed.n_max = (int32_t) n_max;
    parsed.n_mtp_layers = (int32_t) n_mtp_layers;
    parsed.fold_multi = (flags & 1u) != 0;
    validate_metadata(parsed);

    if ((flags & ~1u) != 0 || draft_pending > 1 ||
            n_seq != (uint32_t) expected.n_seq ||
            n_embd != (uint32_t) expected.n_embd || n_max != (uint32_t) expected.n_max ||
            n_mtp_layers != (uint32_t) expected.n_mtp_layers ||
            ((flags & 1u) != 0) != expected.fold_multi) {
        throw std::invalid_argument("incompatible MTP snapshot identity");
    }

    parsed.deferred.keep = keep;
    parsed.deferred.draft_pending = draft_pending != 0;
    if (n_rows > max_deferred_rows || n_rows > n_max + 1 || keep < -1 || keep > (int32_t) n_rows) {
        throw std::invalid_argument("invalid MTP snapshot deferred row count");
    }

    const size_t expected_size = header_size +
            (size_t) n_embd * sizeof(float) +
            (size_t) n_rows * (2 * sizeof(int32_t) + (size_t) n_embd * sizeof(float)) +
            checksum_size;
    if (expected_size > max_bytes || data.size() != expected_size) {
        throw std::invalid_argument("MTP snapshot payload length does not match dimensions");
    }
    if (read_u64(data, expected_size - checksum_size) != checksum(data, expected_size - checksum_size)) {
        throw std::invalid_argument("MTP snapshot checksum mismatch");
    }

    parsed.deferred.tokens.resize(n_rows);
    parsed.deferred.positions.resize(n_rows);
    parsed.deferred.hidden.resize((size_t) n_rows * n_embd);
    parsed.pending_h.resize(n_embd);

    size_t offset = header_size;
    const size_t pending_bytes = (size_t) n_embd * sizeof(float);
    const size_t vector_bytes = (size_t) n_rows * sizeof(int32_t);
    const size_t hidden_bytes = (size_t) n_rows * n_embd * sizeof(float);
    std::memcpy(parsed.pending_h.data(), data.data() + offset, pending_bytes);
    offset += pending_bytes;
    if (vector_bytes != 0) {
        std::memcpy(parsed.deferred.tokens.data(), data.data() + offset, vector_bytes);
        offset += vector_bytes;
        std::memcpy(parsed.deferred.positions.data(), data.data() + offset, vector_bytes);
        offset += vector_bytes;
        std::memcpy(parsed.deferred.hidden.data(), data.data() + offset, hidden_bytes);
    }

    validate_shape(parsed);
    // The physical source slot is diagnostic metadata. Server context restore
    // remaps sequence IDs, so bind CPU-only MTP state to the requested lane.
    parsed.seq_id = expected.seq_id;
    destination = std::move(parsed);
}

} // namespace common_speculative_mtp_snapshot_wire

template <typename SeqIdAt>
common_speculative_mtp_layout common_speculative_mtp_make_layout(
        int32_t n_seq, int32_t n_tokens, SeqIdAt seq_id_at) {
    if (n_seq <= 0 || n_tokens < 0) {
        throw std::invalid_argument("invalid MTP batch dimensions");
    }

    common_speculative_mtp_layout result;
    result.rows.resize(n_seq);
    result.previous.assign(n_tokens, -1);
    for (int32_t row = 0; row < n_tokens; ++row) {
        const int32_t seq = seq_id_at(row);
        if (seq < 0 || seq >= n_seq) {
            throw std::invalid_argument("MTP batch sequence out of range");
        }
        auto & rows = result.rows[seq];
        if (!rows.empty()) {
            result.previous[row] = rows.back();
        }
        rows.push_back(row);
    }
    return result;
}

inline void common_speculative_mtp_shift_rows(
        const common_speculative_mtp_layout & layout, int32_t n_embd,
        const float * hidden, const std::vector<std::vector<float>> & pending,
        float * shifted) {
    if (n_embd <= 0 || pending.size() != layout.rows.size()) {
        throw std::invalid_argument("invalid MTP hidden dimensions");
    }
    for (const auto & row : pending) {
        if (row.size() != static_cast<size_t>(n_embd)) {
            throw std::invalid_argument("invalid MTP pending row width");
        }
    }
    if (layout.previous.empty()) {
        return;
    }
    if (hidden == nullptr || shifted == nullptr || hidden == shifted) {
        throw std::invalid_argument("missing MTP hidden storage");
    }

    const size_t row_bytes = static_cast<size_t>(n_embd) * sizeof(float);
    for (size_t seq = 0; seq < layout.rows.size(); ++seq) {
        for (int32_t row : layout.rows[seq]) {
            const int32_t previous = layout.previous[row];
            const float * source = previous < 0 ? pending[seq].data()
                    : hidden + static_cast<size_t>(previous) * n_embd;
            std::memcpy(shifted + static_cast<size_t>(row) * n_embd, source, row_bytes);
        }
    }
}
