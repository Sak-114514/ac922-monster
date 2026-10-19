#pragma once

#include <string>

namespace cupti_window_v14 {

enum class phase { off, prefill1, decode4, invalid };

inline phase parse_phase(const char * value) {
    if (value == nullptr || *value == '\0' || std::string(value) == "off") return phase::off;
    if (std::string(value) == "prefill1") return phase::prefill1;
    if (std::string(value) == "decode4") return phase::decode4;
    return phase::invalid;
}

inline const char * phase_name(phase value) {
    switch (value) {
        case phase::off: return "off";
        case phase::prefill1: return "prefill1";
        case phase::decode4: return "decode4";
        case phase::invalid: return "invalid";
    }
    return "invalid";
}

inline bool selected_for(phase value, const std::string & label) {
    return (value == phase::prefill1 && label == "single8k-pp-probe-32") ||
           (value == phase::decode4 && label == "real8k-p4-2x2");
}

} // namespace cupti_window_v14
