#pragma once
#include <cstdint>
namespace bank_prefill_profile {
struct submission { uint32_t logical_batch; uint32_t device_copies; uint32_t host_slots; };
inline submission choose(uint32_t ubatch, bool bulk256, bool compact512, uint32_t capacity) noexcept {
    if (ubatch == 512 && compact512 && capacity > 0 && capacity <= 17152) return {32768,4,4};
    if (ubatch >= 512) return {32768,8,8};
    return {bulk256 && ubatch == 256 ? 32768u : 4096u,4,4};
}
}
