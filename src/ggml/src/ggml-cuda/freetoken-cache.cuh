#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Banks are gate/up, down, block scales for both, and FP16 row scales for both.
// Each source pointer addresses a contiguous [expert, bytes_per_expert] CUDA/UVA bank.
struct freetoken_cache_layer {
    const void * banks[6];
};

struct freetoken_cache_view {
    int route_capacity;
    uint8_t * banks[6];
    size_t bytes_per_expert[6];
    int32_t * slot_for_id;
    int32_t * id_of_slot;
    uint64_t * usage;
    uint64_t * recency;
    uint64_t * step;
    int32_t * gpu_ids;
    int32_t * cpu_ids;
    int32_t * hit_ids;
    int32_t * fetched_ids;
    int32_t * fetch_experts;
    int32_t * fetch_slots;
    int32_t * counts; // unique active, unique misses, fetched
};

class freetoken_gpu_cache {
public:
    freetoken_gpu_cache(int experts, int slots, int max_routes, const size_t (&bytes)[6],
                        const std::vector<freetoken_cache_layer> & layers);
    ~freetoken_gpu_cache();
    freetoken_gpu_cache(const freetoken_gpu_cache &) = delete;
    freetoken_gpu_cache & operator=(const freetoken_gpu_cache &) = delete;

    // Calls and all consumers must be serialized on one stream. No host synchronization.
    // Synchronize and destroy graph executables before destroying this object.
    // Source banks remain alive and immutable until all work completes. IDs are not modified.
    // Negative/out-of-range IDs are dropped. A full live cache sends extra misses to CPU.
    void enqueue(int layer, int routes, const int32_t * ids, int max_fetch,
                 cudaStream_t stream, int fetch_fraction_q16 = 0);
    // Split routing/copy permits CPU submission between the two GPU launches.
    // Fetch must use the same layer/routes/cap/fraction as the preceding route call.
    void route(int layer, int routes, const int32_t * ids, int max_fetch,
               cudaStream_t stream, int fetch_fraction_q16 = 0);
    void fetch(int layer, int routes, int max_fetch, cudaStream_t stream, int fetch_fraction_q16 = 0);
    void reset(cudaStream_t stream);
    const freetoken_cache_view & view() const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};
