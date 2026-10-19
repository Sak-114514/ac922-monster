#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace prefill_logit_dump_v16 {

constexpr uint32_t kFormatVersion = 1;
constexpr char kMagic[16] = {'A','C','9','2','2','P','F','L','O','G','I','T','S','V','1','6'};

struct dump_info {
    std::string path;
    std::string flow_id;
    std::string payload_crc32;
    uint64_t logits_count;
    uint64_t payload_bytes;
};

inline std::string json_escape(const std::string & value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() + 8);
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (ch < 0x20) {
                    result += "\\u00";
                    result.push_back(hex[(ch >> 4) & 0xf]);
                    result.push_back(hex[ch & 0xf]);
                } else {
                    result.push_back(static_cast<char>(ch));
                }
        }
    }
    return result;
}

inline std::string safe_component(const std::string & value) {
    std::string result;
    result.reserve(value.size());
    for (const unsigned char ch : value) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                (ch >= '0' && ch <= '9') || ch == '-' || ch == '_') {
            result.push_back(static_cast<char>(ch));
        } else {
            result.push_back('_');
        }
    }
    if (result.empty()) result = "unnamed";
    return result;
}

inline uint32_t crc32_update(uint32_t crc, const unsigned char * bytes, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (unsigned bit = 0; bit < 8; ++bit) {
            const uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return crc;
}

inline uint32_t payload_crc32(const float * logits, size_t count) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(logits[i]), "F32 logits must be 32 bits");
        std::memcpy(&bits, &logits[i], sizeof(bits));
        const unsigned char le[4] = {
            static_cast<unsigned char>(bits),
            static_cast<unsigned char>(bits >> 8),
            static_cast<unsigned char>(bits >> 16),
            static_cast<unsigned char>(bits >> 24),
        };
        crc = crc32_update(crc, le, sizeof(le));
    }
    return ~crc;
}

inline std::string crc32_hex(uint32_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(8, '0');
    for (int i = 7; i >= 0; --i) {
        result[static_cast<size_t>(i)] = digits[value & 0xfu];
        value >>= 4;
    }
    return result;
}

inline void write_u32_le(std::ostream & output, uint32_t value) {
    const char bytes[4] = {
        static_cast<char>(value), static_cast<char>(value >> 8),
        static_cast<char>(value >> 16), static_cast<char>(value >> 24),
    };
    output.write(bytes, sizeof(bytes));
}

template<typename Token>
inline dump_info write_full_f32(const std::filesystem::path & directory,
        const std::string & scenario, const std::string & flow_id,
        const std::string & prompt_source, const std::string & prompt_sha256,
        const std::vector<Token> & prompt, const float * logits, size_t count) {
    static_assert(std::numeric_limits<float>::is_iec559, "logit dumps require IEEE-754 F32");
    if (prompt.empty() || logits == nullptr || count == 0 || count > UINT32_MAX) {
        throw std::runtime_error("invalid prompt/logit buffer for prefill dump");
    }
    std::filesystem::create_directories(directory);
    const std::string basename = safe_component(scenario) + "__" + safe_component(flow_id) + ".ac922pflogits";
    const std::filesystem::path final_path = std::filesystem::absolute(directory / basename);
    const std::filesystem::path temporary_path = final_path.string() + ".tmp";
    const uint32_t crc = payload_crc32(logits, count);

    std::ostringstream header;
    header << "{\"format\":\"ac922-prefill-logits\",\"format_version\":" << kFormatVersion
           << ",\"scenario\":\"" << json_escape(scenario)
           << "\",\"flow_id\":\"" << json_escape(flow_id)
           << "\",\"prompt_source\":\"" << json_escape(prompt_source)
           << "\",\"prompt_sha256\":\"" << json_escape(prompt_sha256)
           << "\",\"dtype\":\"float32\",\"byte_order\":\"little\",\"logits_count\":" << count
           << ",\"logits_row_position\":" << prompt.size() - 1
           << ",\"logits_row_input_token_id\":" << static_cast<int64_t>(prompt.back())
           << ",\"prompt_token_positions\":[";
    for (size_t position = 0; position < prompt.size(); ++position) {
        if (position) header << ',';
        header << "{\"position\":" << position << ",\"token_id\":"
               << static_cast<int64_t>(prompt[position]) << '}';
    }
    header << "],\"payload_bytes\":" << count * sizeof(float)
           << ",\"payload_crc32\":\"" << crc32_hex(crc) << "\"}";

    const std::string encoded_header = header.str();
    if (encoded_header.size() > UINT32_MAX) throw std::runtime_error("prefill dump header is too large");
    std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create prefill logit dump: " + temporary_path.string());
    output.write(kMagic, sizeof(kMagic));
    write_u32_le(output, kFormatVersion);
    write_u32_le(output, static_cast<uint32_t>(encoded_header.size()));
    output.write(encoded_header.data(), static_cast<std::streamsize>(encoded_header.size()));
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &logits[i], sizeof(bits));
        write_u32_le(output, bits);
    }
    output.close();
    if (!output) throw std::runtime_error("failed writing prefill logit dump: " + temporary_path.string());
    if (std::rename(temporary_path.c_str(), final_path.c_str()) != 0) {
        throw std::runtime_error("cannot atomically publish prefill logit dump: " + final_path.string());
    }

    return {final_path.string(), flow_id, crc32_hex(crc), count, count * sizeof(float)};
}

} // namespace prefill_logit_dump_v16
