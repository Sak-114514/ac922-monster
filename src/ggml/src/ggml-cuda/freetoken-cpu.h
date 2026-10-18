#pragma once

#include <cstdint>
#include <memory>
#include <vector>

void freetoken_unpack_nvfp4(const void * gguf_data, int64_t rows, int64_t columns,
                           uint8_t * packed, uint8_t * scales);

// FT row-major NVFP4: adjacent low/high nibbles, E4M3 per 16, FP16 per output row.
// Gate/up rows are [expert, 2*intermediate, hidden]; down rows are [expert, hidden, intermediate].
struct freetoken_cpu_bank {
    const uint8_t * gate_up;
    const uint8_t * down;
    const uint8_t * gate_up_scale;
    const uint8_t * down_scale;
    const uint16_t * gate_up_global;
    const uint16_t * down_global;
};

class freetoken_cpu_executor {
public:
    freetoken_cpu_executor(int hidden, int intermediate, int experts, int top_k, int max_tokens,
                          int threads, std::vector<freetoken_cpu_bank> banks, std::vector<int> cores = {});
    ~freetoken_cpu_executor();
    freetoken_cpu_executor(const freetoken_cpu_executor &) = delete;
    freetoken_cpu_executor & operator=(const freetoken_cpu_executor &) = delete;

    // One invocation at a time. Banks and buffers remain owned by the caller.
    void run(int layer, int tokens, const uint16_t * x, const int32_t * ids,
             const float * weights, uint16_t * y, const int8_t * xq = nullptr, const float * xs = nullptr);
    const char * isa() const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};
