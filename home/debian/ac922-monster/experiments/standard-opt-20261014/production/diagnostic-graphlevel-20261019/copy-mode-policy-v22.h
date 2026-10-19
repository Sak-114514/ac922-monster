#pragma once

#include <string>
#include <vector>

namespace copy_mode_policy_v22 {

inline bool parse_variants(const char * text, std::vector<int> & modes) {
    modes.clear();
    const std::string value = text && *text ? text : "1";
    if (value == "1") {
        modes = {1};
        return true;
    }
    if (value == "1,0") {
        modes = {1, 0};
        return true;
    }
    return false;
}

} // namespace copy_mode_policy_v22
