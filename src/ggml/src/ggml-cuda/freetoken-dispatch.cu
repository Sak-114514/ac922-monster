// SPDX-License-Identifier: Apache-2.0
// Native adaptation of FreeToken kernel/csrc/cpu_moe/cpu_moe_ext.cpp; see licenses/LICENSE-freetoken.
// Source SHA256: f25698c3a9a9741f4ed3e7460a2af7558cb42111531f04de1472dd8835cef8d1
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "freetoken-dispatch.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace {
static int coordinator_cpu() {
    const char * value = std::getenv("GGML_FT_COORD_CPU");
    if (!value) { return -1; }
    char * end = nullptr;
    const long cpu = std::strtol(value, &end, 10);
    if (end == value || *end || cpu < -1) {
        throw std::invalid_argument("GGML_FT_COORD_CPU must be -1 or an allowed CPU index");
    }
    if (cpu == -1) { return -1; }
#if defined(__linux__)
    cpu_set_t allowed;
    if (cpu >= CPU_SETSIZE || sched_getaffinity(0, sizeof(allowed), &allowed) != 0 || !CPU_ISSET(cpu, &allowed)) {
        throw std::invalid_argument("GGML_FT_COORD_CPU is outside the allowed CPU affinity");
    }
    return int(cpu);
#else
    throw std::invalid_argument("GGML_FT_COORD_CPU requires Linux");
#endif
}
static void ft_check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("FreeToken dispatch: ") + cudaGetErrorString(error));
    }
}
static void ft_driver_check(CUresult error) {
    if (error != CUDA_SUCCESS) {
        throw std::runtime_error("FreeToken stream memory operation failed: " + std::to_string(int(error)));
    }
}
using memop_fn = CUresult (CUDAAPI *)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
static memop_fn memop(const char * name) {
#if CUDART_VERSION >= 11030
    void * ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    ft_check(cudaGetDriverEntryPoint(name, &ptr, cudaEnableDefault, &status));
    if (status != cudaDriverEntryPointSuccess || !ptr) {
        throw std::runtime_error("FreeToken requires CUDA stream memory operations");
    }
    return reinterpret_cast<memop_fn>(ptr);
#else
    (void) name;
    throw std::runtime_error("FreeToken requires CUDA 11.3 or later");
#endif
}
struct alignas(64) flags {
    std::atomic<uint64_t> ready{0};
    alignas(64) std::atomic<uint64_t> done{0};
};
struct task {
    int layer, tokens;
};
struct buffers {
    flags * signal;
    task * job;
    uint16_t * x;
    uint16_t * y;
    int8_t * xq;
    float * xs;
    int32_t * ids;
    float * weights;
};
__global__ void prepare(buffers b, int h, int k, int layer, int tokens,
                        const float * x, const int32_t * ids, const float * weights) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Quantize groups of 16 with the same BF16 and rounding rules as the CPU executor.
    if (tid < tokens * (h / 16)) {
        const int base = tid * 16;
        float values[16];
        float amax = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const __nv_bfloat16 value = __float2bfloat16(x[base + j]);
            b.x[base + j] = __bfloat16_as_ushort(value);
            values[j] = __bfloat162float(value);
            amax = fmaxf(amax, fabsf(values[j]));
        }
        const float scale = amax > 0.0f ? __fdiv_rn(amax, 127.0f) : 1.0f;
        const float inv = __fdiv_rn(1.0f, scale);
        b.xs[tid] = scale;
        for (int j = 0; j < 16; ++j) {
            const float q = roundf(values[j] * inv);
            b.xq[base + j / 2 + (j % 2) * 8] = int8_t(fminf(127.0f, fmaxf(-127.0f, q)));
        }
    }
    if (tid < tokens * k) { b.ids[tid] = ids[tid]; b.weights[tid] = weights[tid]; }
    if (tid == 0) { b.job->layer = layer; b.job->tokens = tokens; }
}
struct wait_sample { uint64_t calls, ns, maximum, span_ns, buckets[5]; };
struct wait_stats { uint64_t start, submit_start; int row; wait_sample samples[8]; };
static __device__ __forceinline__ uint64_t gpu_time() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
__global__ void begin_submit(wait_stats * stats) { stats->submit_start = gpu_time(); }
__global__ void begin_wait(wait_stats * stats, const int32_t * ids, int routes, int tokens) {
    bool active = false;
    for (int r = 0; r < routes; ++r) { active |= ids[r] >= 0; }
    stats->row = 2*(tokens-1) + int(active);
    stats->start = gpu_time();
}
template<bool TIMING = false, bool ROUTES = false>
__global__ void finish_cpu(const uint16_t * cpu, float * out, const float * gpu, int n, wait_stats * stats = nullptr,
                           const uint16_t * down = nullptr, int hidden = 0, int top_k = 0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (TIMING) {
        if (i == 0) {
            const uint64_t end = gpu_time();
            const uint64_t ns = end - stats->start;
            auto & sample = stats->samples[stats->row];
            ++sample.calls; sample.ns += ns;
            sample.span_ns += end - stats->submit_start;
            sample.maximum = sample.maximum > ns ? sample.maximum : ns;
            ++sample.buckets[(ns >= 5000) + (ns >= 25000) + (ns >= 100000) + (ns >= 500000)];
        }
    }
    if (i < n) {
        if constexpr (ROUTES) {
            const int token = i/hidden, col = i%hidden;
            float sum = 0;
            for (int k = 0; k < top_k; ++k) {
                sum = __fadd_rn(sum,__bfloat162float(__ushort_as_bfloat16(down[(size_t(token)*top_k+k)*hidden+col])));
            }
            const float result = __fadd_rn(__bfloat162float(__ushort_as_bfloat16(cpu[i])),sum);
            out[i] = __bfloat162float(__float2bfloat16(result));
        } else { out[i] = __bfloat162float(__ushort_as_bfloat16(cpu[i])) + (gpu ? gpu[i] : 0.0f); }
    }
}
} // namespace

struct freetoken_cpu_dispatch::impl {
    int h, k, max_tokens, layers, device;
    const int coord_cpu = coordinator_cpu();
    freetoken_cpu_executor executor;
    buffers host{}, gpu{}, staging{};
    uint8_t * host_input = nullptr, * device_input = nullptr;
    size_t input_bytes = 0;
    memop_fn write, wait;
    wait_stats * gpu_wait = nullptr;
    std::vector<void *> allocations;
    std::atomic<bool> stop{false}, failed{false};
    std::thread coordinator;
    bool bound = false;
    cudaStream_t bound_stream = nullptr;
    bool pending = false;
    cudaStream_t pending_stream = nullptr;
    int pending_tokens = 0;
    const bool timing = std::getenv("GGML_FT_CPU_TIMING") && std::strcmp(std::getenv("GGML_FT_CPU_TIMING"),"1") == 0;
    const bool wait_timing = std::getenv("GGML_FT_WAIT_TIMING") && std::strcmp(std::getenv("GGML_FT_WAIT_TIMING"),"1") == 0;
    const bool route_stats = std::getenv("GGML_FT_CPU_ROUTE_STATS") && std::strcmp(std::getenv("GGML_FT_CPU_ROUTE_STATS"),"1") == 0;
    struct sample { uint64_t calls = 0, routes = 0, empty = 0, ns = 0; } samples[4];
    struct route_sample { uint64_t calls = 0, routes = 0, unique = 0, multiplicity[257]{}, tokens[5]{}; } route_samples[4];

    void count_routes() {
        int ids[256], counts[256]{}, masks[256]{};
        int groups = 0;
        auto & sample = route_samples[host.job->tokens-1];
        ++sample.calls;
        for (int r = 0; r < host.job->tokens*k; ++r) {
            const int id = host.ids[r];
            if (id < 0) { continue; }
            int g = 0;
            while (g < groups && ids[g] != id) { ++g; }
            if (g == groups) { ids[groups++] = id; }
            ++counts[g]; masks[g] |= 1 << (r/k); ++sample.routes;
        }
        sample.unique += groups;
        for (int g = 0; g < groups; ++g) {
            ++sample.multiplicity[counts[g]];
            int tokens = 0;
            for (int t = 0; t < host.job->tokens; ++t) { tokens += (masks[g] >> t) & 1; }
            ++sample.tokens[tokens];
        }
    }

    template<typename T> void allocate(T * & a, T * & b, size_t count) {
        ft_check(cudaHostAlloc(reinterpret_cast<void **>(&a), count * sizeof(T), cudaHostAllocMapped));
        allocations.push_back(a);
        ft_check(cudaHostGetDevicePointer(reinterpret_cast<void **>(&b), a, 0));
    }
    CUdeviceptr ready() const { return reinterpret_cast<CUdeviceptr>(gpu.signal) + offsetof(flags, ready); }
    CUdeviceptr done() const { return reinterpret_cast<CUdeviceptr>(gpu.signal) + offsetof(flags, done); }
    void release() noexcept {
        cudaFree(gpu_wait);
        cudaFree(device_input);
        for (void * a : allocations) { cudaFreeHost(a); }
    }
    template<typename T> void input_field(T * & a, T * & b, size_t count, size_t & offset) {
        offset = (offset + 63) & ~size_t(63);
        a = reinterpret_cast<T *>(host_input + offset);
        b = reinterpret_cast<T *>(device_input + offset);
        offset += count * sizeof(T);
    }
    impl(int H, int I, int E, int K, int T, int threads, std::vector<freetoken_cpu_bank> banks,
         std::vector<int> cores) : h(H), k(K), max_tokens(T), layers(int(banks.size())),
         executor(H, I, E, K, T, threads, std::move(banks), std::move(cores)) {
        static_assert(std::atomic<uint64_t>::is_always_lock_free, "Mapped flags require lock-free atomics");
        ft_check(cudaGetDevice(&device));
        write = memop("cuStreamWriteValue64"); wait = memop("cuStreamWaitValue64");
        allocations.reserve(8);
        try {
            if (wait_timing) {
                ft_check(cudaMalloc(&gpu_wait, sizeof(wait_stats)));
                ft_check(cudaMemset(gpu_wait, 0, sizeof(wait_stats)));
            }
            allocate(host.signal, gpu.signal, 1); new (host.signal) flags;
            allocate(host.y, gpu.y, size_t(T)*H);
            input_bytes = 6*64 + sizeof(task) + size_t(T)*H*3 + size_t(T)*(H/16)*sizeof(float) + size_t(T)*K*8;
            uint8_t * mapped_input;
            allocate(host_input, mapped_input, input_bytes);
            ft_check(cudaMalloc(&device_input, input_bytes));
            ft_check(cudaMemset(device_input, 0, input_bytes));
            size_t offset = 0;
            input_field(host.job, staging.job, 1, offset);
            input_field(host.x, staging.x, size_t(T)*H, offset);
            input_field(host.xq, staging.xq, size_t(T)*H, offset);
            input_field(host.xs, staging.xs, size_t(T)*(H/16), offset);
            input_field(host.ids, staging.ids, size_t(T)*K, offset);
            input_field(host.weights, staging.weights, size_t(T)*K, offset);
            input_bytes = offset;
            // Probe eager operations before starting any task or wait on CPU completion.
            ft_driver_check(write(nullptr, done(), 7, CU_STREAM_WRITE_VALUE_DEFAULT));
            ft_driver_check(wait(nullptr, done(), 7, CU_STREAM_WAIT_VALUE_GEQ));
            ft_check(cudaStreamSynchronize(nullptr));
            host.signal->done.store(0);
            coordinator = std::thread([this] { loop(); });
#if defined(__linux__)
            pthread_setname_np(coordinator.native_handle(), "ft-moe-coord");
            if (coord_cpu >= 0) {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(coord_cpu, &set);
                const int error = pthread_setaffinity_np(coordinator.native_handle(), sizeof(set), &set);
                if (error) { throw std::runtime_error("FreeToken coordinator affinity: " + std::string(std::strerror(error))); }
                fprintf(stderr, "FreeToken: CPU coordinator pinned to logical CPU %d\n", coord_cpu);
            }
#endif
        } catch (...) {
            stop.store(true, std::memory_order_release);
            if (coordinator.joinable()) { coordinator.join(); }
            release();
            throw;
        }
    }
    ~impl() {
        stop.store(true, std::memory_order_release);
        if (coordinator.joinable()) { coordinator.join(); }
        if (gpu_wait) {
            wait_stats stats{};
            const auto error = cudaMemcpy(&stats, gpu_wait, sizeof(stats), cudaMemcpyDeviceToHost);
            if (error == cudaSuccess) for (int row = 0; row < 8; ++row) {
                const auto & s = stats.samples[row];
                if (s.calls) { fprintf(stderr,"FreeToken GPU wait: tokens=%d active=%d calls=%llu total_ms=%.3f mean_us=%.3f max_us=%.3f buckets=%llu,%llu,%llu,%llu,%llu\n",
                    row/2+1,row%2,(unsigned long long)s.calls,s.ns/1e6,double(s.ns)/s.calls/1e3,s.maximum/1e3,
                    (unsigned long long)s.buckets[0],(unsigned long long)s.buckets[1],(unsigned long long)s.buckets[2],
                    (unsigned long long)s.buckets[3],(unsigned long long)s.buckets[4]); }
                if (s.calls) { fprintf(stderr,"FreeToken hybrid span: tokens=%d active=%d calls=%llu total_ms=%.3f mean_us=%.3f\n",
                    row/2+1,row%2,(unsigned long long)s.calls,s.span_ns/1e6,double(s.span_ns)/s.calls/1e3); }
            }
            else { fprintf(stderr,"FreeToken GPU wait report failed: %s\n",cudaGetErrorString(error)); }
        }
        if (timing) for (int t = 0; t < 4; ++t) {
            const auto & s = samples[t];
            if (s.calls) { fprintf(stderr,"FreeToken CPU timing: tokens=%d calls=%llu routes=%llu empty=%llu compute_ms=%.3f mean_us=%.3f\n",
                t+1,(unsigned long long)s.calls,(unsigned long long)s.routes,(unsigned long long)s.empty,s.ns/1e6,double(s.ns)/s.calls/1e3); }
        }
        if (route_stats) for (int t = 0; t < 4; ++t) {
            const auto & s = route_samples[t];
            if (!s.calls) { continue; }
            fprintf(stderr,"FreeToken CPU routes: tokens=%d calls=%llu routes=%llu unique=%llu\n",
                t+1,(unsigned long long)s.calls,(unsigned long long)s.routes,(unsigned long long)s.unique);
            for (int n = 1; n <= 256; ++n) if (s.multiplicity[n]) {
                fprintf(stderr,"FreeToken CPU route multiplicity: tokens=%d routes_per_expert=%d groups=%llu\n",t+1,n,(unsigned long long)s.multiplicity[n]);
            }
            for (int n = 1; n <= 4; ++n) if (s.tokens[n]) {
                fprintf(stderr,"FreeToken CPU route sharing: tokens=%d tokens_per_expert=%d groups=%llu\n",t+1,n,(unsigned long long)s.tokens[n]);
            }
        }
        release();
    }
    void loop() noexcept {
        auto active = std::chrono::steady_clock::now();
        while (!stop.load(std::memory_order_acquire)) {
            if (!host.signal->ready.load(std::memory_order_acquire)) {
                if (std::chrono::steady_clock::now() - active > std::chrono::milliseconds(50)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                } else {
#if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
#else
                    std::this_thread::yield();
#endif
                }
                continue;
            }
            host.signal->ready.store(0, std::memory_order_release);
            try {
                if (failed.load()) { throw std::runtime_error("CPU executor failed"); }
                const auto begin = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                executor.run(host.job->layer, host.job->tokens, host.x, host.ids, host.weights,
                             host.y, host.xq, host.xs);
                if (timing) {
                    const auto end = std::chrono::steady_clock::now();
                    auto & sample = samples[host.job->tokens-1];
                    ++sample.calls;
                    int active = 0;
                    for (int r = 0; r < host.job->tokens*k; ++r) { active += host.ids[r] >= 0; }
                    sample.routes += active; sample.empty += active == 0;
                    sample.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
                }
                if (route_stats) { count_routes(); }
            } catch (...) {
                failed.store(true, std::memory_order_release);
                // Complete the handshake on failure; never leave the GPU front-end waiting.
                for (int i = 0; i < max_tokens * h; ++i) { host.y[i] = 0x7fc0; }
            }
            host.signal->done.store(1, std::memory_order_release);
            active = std::chrono::steady_clock::now();
        }
    }
    void check() const {
        if (failed.load(std::memory_order_acquire)) { throw std::runtime_error("FreeToken CPU executor failed"); }
        int current; ft_check(cudaGetDevice(&current));
        if (current != device) { throw std::runtime_error("FreeToken dispatch used on another device"); }
    }
};

freetoken_cpu_dispatch::freetoken_cpu_dispatch(int h, int i, int e, int k, int t, int threads,
        std::vector<freetoken_cpu_bank> banks, std::vector<int> cores) :
    p(new impl(h, i, e, k, t, threads, std::move(banks), std::move(cores))) {}
freetoken_cpu_dispatch::~freetoken_cpu_dispatch() = default;
void freetoken_cpu_dispatch::check() const { p->check(); }
void freetoken_cpu_dispatch::submit(int layer, int tokens, const float * x, const int32_t * ids,
                                   const float * weights, cudaStream_t stream) {
    p->check();
    if ((p->bound && stream != p->bound_stream) || p->pending || layer < 0 || layer >= p->layers || tokens < 1 || tokens > p->max_tokens || !x || !ids || !weights) {
        throw std::invalid_argument("Invalid FreeToken CPU submission");
    }
    p->bound = true; p->bound_stream = stream;
    if (p->gpu_wait) { begin_submit<<<1,1,0,stream>>>(p->gpu_wait); ft_check(cudaGetLastError()); }
    const int work = tokens * std::max(p->h / 16, p->k);
    prepare<<<(work + 127) / 128, 128, 0, stream>>>(p->staging, p->h, p->k, layer, tokens, x, ids, weights);
    ft_check(cudaGetLastError());
    // One DMA replaces scattered PCIe stores from the quantization kernel.
    ft_check(cudaMemcpyAsync(p->host_input, p->device_input, p->input_bytes, cudaMemcpyDeviceToHost, stream));
    ft_driver_check(p->write(reinterpret_cast<CUstream>(stream), p->done(), 0, CU_STREAM_WRITE_VALUE_DEFAULT));
    ft_driver_check(p->write(reinterpret_cast<CUstream>(stream), p->ready(), 1, CU_STREAM_WRITE_VALUE_DEFAULT));
    p->pending = true; p->pending_stream = stream; p->pending_tokens = tokens;
}
void freetoken_cpu_dispatch::finish(int tokens, float * out, const float * gpu, cudaStream_t stream) {
    finish_impl(tokens,out,gpu,nullptr,stream);
}
void freetoken_cpu_dispatch::finish_routes(int tokens, float * out, const uint16_t * down, cudaStream_t stream) {
    if (!down) { throw std::invalid_argument("Missing FreeToken GPU route outputs"); }
    finish_impl(tokens,out,nullptr,down,stream);
}
void freetoken_cpu_dispatch::finish_impl(int tokens, float * out, const float * gpu, const uint16_t * down, cudaStream_t stream) {
    p->check();
    if (!p->pending || tokens != p->pending_tokens || stream != p->pending_stream || !out) {
        throw std::invalid_argument("Invalid FreeToken CPU completion");
    }
    if (p->gpu_wait) {
        begin_wait<<<1,1,0,stream>>>(p->gpu_wait,p->staging.ids,tokens*p->k,tokens);
        ft_check(cudaGetLastError());
    }
    ft_driver_check(p->wait(reinterpret_cast<CUstream>(stream), p->done(), 1, CU_STREAM_WAIT_VALUE_GEQ));
    const int n = tokens * p->h;
    if (down) {
        if (p->gpu_wait) { finish_cpu<true,true><<<(n + 255) / 256, 256, 0, stream>>>(p->gpu.y,out,nullptr,n,p->gpu_wait,down,p->h,p->k); }
        else { finish_cpu<false,true><<<(n + 255) / 256, 256, 0, stream>>>(p->gpu.y,out,nullptr,n,nullptr,down,p->h,p->k); }
    } else {
        if (p->gpu_wait) { finish_cpu<true><<<(n + 255) / 256, 256, 0, stream>>>(p->gpu.y, out, gpu, n, p->gpu_wait); }
        else { finish_cpu<<<(n + 255) / 256, 256, 0, stream>>>(p->gpu.y, out, gpu, n); }
    }
    ft_check(cudaGetLastError());
    p->pending = false;
}

#endif // CUDA-only FreeToken path
