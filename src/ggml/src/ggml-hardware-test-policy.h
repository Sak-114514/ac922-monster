#pragma once

#include <cstdio>
#include <cstdlib>

// Private idle-only experiment control: 0=legacy, 1=pipeline, 2=fenced serial,
// 3=pipeline+structural graphs, 4=fenced serial+structural graphs.
inline int ggml_hardware_test_policy() {
    const char * path = std::getenv("GGML_HARDWARE_TEST_POLICY_FILE");
    if (!path || !*path) return -1;
    std::FILE * file = std::fopen(path, "r");
    if (!file) return 0;
    const int value = std::fgetc(file);
    std::fclose(file);
    return value >= '0' && value <= '4' ? value - '0' : 0;
}
