#pragma once

#include "freetoken-cpu.h"
#include <cuda_runtime.h>

// Single-stream, single-request bridge. Source banks remain owned by the caller.
// Every submit must be followed by finish before submitting another task.
// Synchronize the stream and destroy graph executables before destroying this object.
class freetoken_cpu_dispatch {
public:
    freetoken_cpu_dispatch(int hidden, int intermediate, int experts, int top_k, int max_tokens,
                          int threads, std::vector<freetoken_cpu_bank> banks, std::vector<int> cores = {});
    ~freetoken_cpu_dispatch();
    freetoken_cpu_dispatch(const freetoken_cpu_dispatch &) = delete;
    freetoken_cpu_dispatch & operator=(const freetoken_cpu_dispatch &) = delete;

    // x: contiguous F32 [tokens, hidden]; routes/weights: [tokens, top_k].
    void submit(int layer, int tokens, const float * x, const int32_t * cpu_ids,
                const float * weights, cudaStream_t stream);
    // Adds the BF16 CPU result to optional F32 GPU output. out may alias gpu.
    void finish(int tokens, float * out, const float * gpu, cudaStream_t stream);
    // Reduce BF16 GPU routes, add the CPU result, then round once to BF16.
    void finish_routes(int tokens, float * out, const uint16_t * down, cudaStream_t stream);
    void check() const;

private:
    void finish_impl(int tokens, float * out, const float * gpu, const uint16_t * down, cudaStream_t stream);
    struct impl;
    std::unique_ptr<impl> p;
};
