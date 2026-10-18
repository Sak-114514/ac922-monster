// SPDX-License-Identifier: Apache-2.0
// Native adaptation of FreeToken moe/offload_kernels.py; see licenses/LICENSE-freetoken.
// Source SHA256: 49aca4fa7efddcd13600e031ef7dd48981e7be3465219a192ddba7611ee5b6d4
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "freetoken-cache.cuh"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
constexpr int ft_route_limit = 256;
constexpr int ft_threads = 256;

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("FreeToken cache: ") + cudaGetErrorString(error));
    }
}

__global__ void route_experts(freetoken_cache_view v, int experts, int slots, int layer,
                              int routes, const int32_t * ids, int cap, int fraction, uint64_t reuse_age = 0) {
    __shared__ int missing[ft_route_limit];
    __shared__ int route_ids[ft_route_limit], unique[ft_route_limit], absent[ft_route_limit];
    __shared__ uint64_t missing_age[ft_route_limit];
    __shared__ int n_missing, n_fetch, n_active, selected;
    __shared__ uint64_t tick;
    __shared__ uint64_t best_age[ft_threads / 32];
    __shared__ int best_slot[ft_threads / 32];
    const int t = threadIdx.x;
    const int base = layer * experts;
    if (t == 0) { tick = ++*v.step; }
    if (t < routes) { route_ids[t] = ids[t]; }
    __syncthreads();
    if (t < routes) {
        const int e = route_ids[t];
        bool first = e >= 0 && e < experts;
        for (int j = 0; j < t && first; ++j) { first = route_ids[j] != e; }
        unique[t] = first;
        absent[t] = 0;
        if (first) {
            const int slot = v.slot_for_id[base + e];
            // Only the first route writes; protect all live hits before victim selection.
            if (slot >= 0) { v.usage[slot] = tick; }
            else { absent[t] = 1; }
        }
    }
    __syncthreads();
    if (t == 0) {
        n_active = n_missing = n_fetch = 0;
        for (int r = 0; r < routes; ++r) {
            n_active += unique[r];
            if (absent[r]) { missing[n_missing++] = route_ids[r]; }
        }
        if (fraction > 0) {
            const int lo = (n_missing * fraction) >> 16;
            const int a = max(lo * (65536 - fraction), (n_missing - lo) * fraction);
            const int b = max((lo + 1) * (65536 - fraction), (n_missing - lo - 1) * fraction);
            cap = a <= b ? lo : lo + 1;
        }
        n_fetch = min(cap, n_missing);
    }
    __syncthreads();
    if (t < n_missing) { missing_age[t] = v.recency[base + missing[t]]; }
    __syncthreads();
    if (t == 0) {
        // Recurring misses first; expert ID breaks recency ties.
        for (int i = 0; i < n_fetch; ++i) {
            int best = i;
            for (int j = i + 1; j < n_missing; ++j) {
                const uint64_t a = missing_age[j];
                const uint64_t b = missing_age[best];
                if (a > b || (a == b && missing[j] < missing[best])) { best = j; }
            }
            if (reuse_age && (!missing_age[best] || tick - missing_age[best] > reuse_age)) {
                n_fetch = i;
                break;
            }
            const int e = missing[i]; missing[i] = missing[best]; missing[best] = e;
            const uint64_t age = missing_age[i]; missing_age[i] = missing_age[best]; missing_age[best] = age;
        }
    }
    __syncthreads();
    for (int i = 0; i < n_fetch; ++i) {
        uint64_t age = UINT64_MAX;
        int slot = -1;
        for (int s = t; s < slots; s += blockDim.x) {
            const uint64_t a = v.usage[s];
            if (a < tick && (a < age || (a == age && (slot < 0 || s < slot)))) {
                age = a; slot = s;
            }
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            const uint64_t a = __shfl_down_sync(0xffffffff, (unsigned long long) age, offset);
            const int s = __shfl_down_sync(0xffffffff, slot, offset);
            if (s >= 0 && (a < age || (a == age && (slot < 0 || s < slot)))) { age = a; slot = s; }
        }
        if ((t & 31) == 0) { best_age[t / 32] = age; best_slot[t / 32] = slot; }
        __syncthreads();
        if (t < 32) {
            age = t < ft_threads / 32 ? best_age[t] : UINT64_MAX;
            slot = t < ft_threads / 32 ? best_slot[t] : -1;
            for (int offset = 16; offset > 0; offset /= 2) {
                const uint64_t a = __shfl_down_sync(0xffffffff, (unsigned long long) age, offset);
                const int s = __shfl_down_sync(0xffffffff, slot, offset);
                if (s >= 0 && (a < age || (a == age && (slot < 0 || s < slot)))) { age = a; slot = s; }
            }
        }
        if (t == 0) {
            selected = slot;
            if (selected < 0) { n_fetch = i; }
            else {
                const int old = v.id_of_slot[selected];
                if (old >= 0) { v.slot_for_id[old] = -1; }
                const int e = missing[i];
                v.id_of_slot[selected] = base + e;
                v.slot_for_id[base + e] = selected;
                v.usage[selected] = tick;
                v.fetch_experts[i] = e;
                v.fetch_slots[i] = selected;
            }
        }
        __syncthreads();
        if (selected < 0) { break; }
    }
    if (t < routes) {
        const int e = route_ids[t];
        const bool valid = e >= 0 && e < experts;
        const int s = valid ? v.slot_for_id[base + e] : -1;
        bool fetched = false;
        for (int i = 0; i < n_fetch; ++i) { fetched |= missing[i] == e; }
        v.gpu_ids[t] = s;
        v.cpu_ids[t] = valid && s < 0 ? e : -1;
        v.hit_ids[t] = s >= 0 && !fetched ? s : -1;
        v.fetched_ids[t] = fetched ? s : -1;
    }
    if (t < routes && unique[t]) { v.recency[base + route_ids[t]] = tick; }
    if (t == 0) {
        v.counts[0] = n_active; v.counts[1] = n_missing; v.counts[2] = n_fetch;
    }
}

__global__ void copy_experts(freetoken_cache_view v, const freetoken_cache_layer * layers, int layer) {
    const int i = blockIdx.y;
    if (i >= v.counts[2]) { return; }
    const int bank = blockIdx.z;
    const size_t bytes = v.bytes_per_expert[bank];
    const uint8_t * src = static_cast<const uint8_t *>(layers[layer].banks[bank]) +
                         size_t(v.fetch_experts[i]) * bytes;
    uint8_t * dst = v.banks[bank] + size_t(v.fetch_slots[i]) * bytes;
    // Bank rows and source pointers have 16-byte alignment.
    const size_t words = bytes / sizeof(uint4);
    for (size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         j < words; j += size_t(gridDim.x) * blockDim.x) {
        reinterpret_cast<uint4 *>(dst)[j] = reinterpret_cast<const uint4 *>(src)[j];
    }
}
} // namespace

struct freetoken_gpu_cache::impl {
    freetoken_cache_view v = {};
    freetoken_cache_layer * layers = nullptr;
    int experts, slots, max_routes, n_layers, device;
    uint64_t reuse_age = 0;
    std::vector<void *> allocations;

    template<typename T> void allocate(T * & ptr, size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::invalid_argument("FreeToken cache allocation overflow");
        }
        check(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)));
        allocations.push_back(ptr);
    }
    void release() noexcept {
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        for (void * ptr : allocations) { cudaFree(ptr); }
        cudaSetDevice(previous);
    }
    void check_device() const {
        int current;
        check(cudaGetDevice(&current));
        if (current != device) { throw std::runtime_error("FreeToken cache used on another device"); }
    }
    impl(int e, int s, int r, const size_t (&bytes)[6], const std::vector<freetoken_cache_layer> & host) :
        experts(e), slots(s), max_routes(r), n_layers(int(host.size())) {
        if (e < 1 || e > 65536 || s < 1 || r < 1 || r > ft_route_limit || host.empty() || host.size() > 128) {
            throw std::invalid_argument("Invalid FreeToken cache dimensions");
        }
        if (const char * value = std::getenv("GGML_FT_FETCH_REUSE_AGE")) {
            char * end = nullptr;
            const long age = std::strtol(value, &end, 10);
            if (end == value || *end || age < 0 || age > 1000000000) {
                throw std::invalid_argument("GGML_FT_FETCH_REUSE_AGE must be an integer from 0 to 1000000000");
            }
            reuse_age = uint64_t(age);
        }
        for (int b = 0; b < 6; ++b) {
            if (bytes[b] == 0 || bytes[b] % 16 || bytes[b] > SIZE_MAX / size_t(std::max(e, s))) {
                throw std::invalid_argument("FreeToken cache bank rows must be nonempty and 16-byte aligned");
            }
            for (const auto & layer : host) {
                if (!layer.banks[b] || reinterpret_cast<uintptr_t>(layer.banks[b]) % 16) {
                    throw std::invalid_argument("Invalid FreeToken source bank");
                }
            }
        }
        check(cudaGetDevice(&device));
        v.route_capacity = r;
        allocations.reserve(24);
        try {
            for (int b = 0; b < 6; ++b) {
                v.bytes_per_expert[b] = bytes[b];
                allocate(v.banks[b], size_t(s) * bytes[b]);
            }
            allocate(layers, host.size());
            check(cudaMemcpy(layers, host.data(), host.size() * sizeof(host[0]), cudaMemcpyHostToDevice));
            allocate(v.slot_for_id, size_t(e) * n_layers);
            allocate(v.id_of_slot, s); allocate(v.usage, s);
            allocate(v.recency, size_t(e) * n_layers); allocate(v.step, 1);
            allocate(v.gpu_ids, r); allocate(v.cpu_ids, r);
            allocate(v.hit_ids, r); allocate(v.fetched_ids, r);
            allocate(v.fetch_experts, r); allocate(v.fetch_slots, r); allocate(v.counts, 3);
            reset(nullptr);
            check(cudaStreamSynchronize(nullptr));
        } catch (...) { release(); throw; }
    }
    ~impl() { release(); }
    void reset(cudaStream_t stream) {
        check_device();
        check(cudaMemsetAsync(v.slot_for_id, 0xff, size_t(experts) * n_layers * sizeof(int32_t), stream));
        check(cudaMemsetAsync(v.id_of_slot, 0xff, size_t(slots) * sizeof(int32_t), stream));
        check(cudaMemsetAsync(v.usage, 0, size_t(slots) * sizeof(uint64_t), stream));
        check(cudaMemsetAsync(v.recency, 0, size_t(experts) * n_layers * sizeof(uint64_t), stream));
        check(cudaMemsetAsync(v.step, 0, sizeof(uint64_t), stream));
        check(cudaMemsetAsync(v.counts, 0, 3 * sizeof(int32_t), stream));
    }
};

freetoken_gpu_cache::freetoken_gpu_cache(int e, int s, int r, const size_t (&bytes)[6],
        const std::vector<freetoken_cache_layer> & layers) : p(new impl(e, s, r, bytes, layers)) {}
freetoken_gpu_cache::~freetoken_gpu_cache() = default;
const freetoken_cache_view & freetoken_gpu_cache::view() const { return p->v; }
void freetoken_gpu_cache::reset(cudaStream_t stream) { p->reset(stream); }
void freetoken_gpu_cache::route(int layer, int routes, const int32_t * ids, int max_fetch,
                                 cudaStream_t stream, int fraction) {
    p->check_device();
    if (layer < 0 || layer >= p->n_layers || routes < 1 || routes > p->max_routes ||
        !ids || max_fetch < 0 || fraction < 0 || fraction > 65536) {
        throw std::invalid_argument("Invalid FreeToken cache routing request");
    }
    route_experts<<<1, ft_threads, 0, stream>>>(p->v, p->experts, p->slots, layer, routes, ids, max_fetch, fraction, p->reuse_age);
    check(cudaGetLastError());
}
void freetoken_gpu_cache::fetch(int layer, int routes, int max_fetch, cudaStream_t stream, int fraction) {
    p->check_device();
    if (layer < 0 || layer >= p->n_layers || routes < 1 || routes > p->max_routes ||
        max_fetch < 0 || fraction < 0 || fraction > 65536) {
        throw std::invalid_argument("Invalid FreeToken cache fetch request");
    }
    const int copies = fraction ? routes : std::min(routes, max_fetch);
    if (copies == 0) { return; }
    copy_experts<<<dim3(32, copies, 6), ft_threads, 0, stream>>>(p->v, p->layers, layer);
    check(cudaGetLastError());
}

void freetoken_gpu_cache::enqueue(int layer, int routes, const int32_t * ids, int max_fetch,
                                 cudaStream_t stream, int fraction) {
    route(layer, routes, ids, max_fetch, stream, fraction);
    fetch(layer, routes, max_fetch, stream, fraction);
}

#endif // CUDA-only FreeToken path
