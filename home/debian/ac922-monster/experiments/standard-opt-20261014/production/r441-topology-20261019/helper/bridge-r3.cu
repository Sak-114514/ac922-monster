#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif
#include "bridge-api.h"
#include "context-gate-v441.h"
#include "update-policy-v441.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {
constexpr size_t k_max_entries = 32;

// Host-only launch of the unchanged captured graph. No device controller is created.
struct entry {
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t generation = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t child = nullptr;
    cudaEvent_t done = nullptr;
    bool done_recorded = false;
    // A successful child enqueue without its completion record is not safe to
    // rebind or launch again until the stream has been drained.
    bool launch_pending_unfenced = false;
    bool attempted = false;
    bool captured_only = false;
    int outcome = AC922_TAIL_INSTANTIATION_DECLINED;
};

struct shared_exec_slot {
    explicit shared_exec_slot(uint64_t owner_generation, cudaStream_t stream_value,
            uint64_t bucket_value, size_t nodes_value)
        : generation(owner_generation), stream(stream_value),
          topology_bucket(bucket_value), topology_nodes(nodes_value) {}

    std::mutex mutex;
    const uint64_t generation;
    const cudaStream_t stream;
    const uint64_t topology_bucket;
    const size_t topology_nodes;
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t done = nullptr;
    bool done_recorded = false;
    bool launch_pending_unfenced = false;
    const void * committed_key = nullptr;
    const void * committed_graph = nullptr;
    uint64_t commit_serial = 0;
    size_t instantiations = 0;
    size_t update_successes = 0;
    size_t update_failures = 0;
    size_t fallback_launches = 0;
    bool first_update_failure_logged = false;
};

struct shared_key_entry {
    std::mutex mutex;
    entry binding;
    entry fallback;
    uint64_t topology_bucket = 0;
};

struct slot_key {
    cudaStream_t stream = nullptr;
    uint64_t topology = 0;
    bool operator==(const slot_key & other) const noexcept {
        return stream == other.stream && topology == other.topology;
    }
};

struct slot_key_hash {
    size_t operator()(const slot_key & key) const noexcept {
        const size_t a = std::hash<cudaStream_t>{}(key.stream);
        const size_t b = std::hash<uint64_t>{}(key.topology);
        return a ^ (b + size_t(0x9e3779b9) + (a << 6) + (a >> 2));
    }
};

std::atomic<uint64_t> next_context_generation{1};

bool host_shared_exec_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("AC922_HOST_GRAPH_SHARED_EXEC");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

bool topology_buckets_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("AC922_HOST_GRAPH_TOPOLOGY_BUCKETS");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

// A bucket is only a candidate selection hint. CUDA validates every update;
// neither this fingerprint nor a collision can authorize executable replay.
cudaError_t topology_hint(cudaGraph_t graph, uint64_t * hint, size_t * nodes) {
    *hint = 0;
    *nodes = 0;
    if (!topology_buckets_enabled()) return cudaSuccess;
    size_t count = 0;
    cudaError_t status = cudaGraphGetNodes(graph, nullptr, &count);
    if (status != cudaSuccess) return status;
    if (count == 0 || count > 4096) return cudaSuccess;
    std::vector<cudaGraphNode_t> handles(count);
    size_t actual = count;
    status = cudaGraphGetNodes(graph, handles.data(), &actual);
    if (status != cudaSuccess) return status;
    if (actual != count) return cudaErrorInvalidValue;
    uint64_t hash = 1469598103934665603ull;
    auto append = [&](uint64_t value) {
        hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
        hash *= 1099511628211ull;
    };
    append(count);
    std::unordered_map<cudaGraphNode_t, uint32_t> indices;
    indices.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (!indices.emplace(handles[i], uint32_t(i)).second) return cudaErrorInvalidValue;
        cudaGraphNodeType type;
        status = cudaGraphNodeGetType(handles[i], &type);
        if (status != cudaSuccess) return status;
        append(uint64_t(type));
        if (type != cudaGraphNodeTypeKernel && type != cudaGraphNodeTypeMemcpy &&
                type != cudaGraphNodeTypeMemset) {
            // Retain R2's ordinary slot/fallback for unclassified node kinds.
            return cudaSuccess;
        }
    }
    size_t edge_count = 0;
    status = cudaGraphGetEdges(graph, nullptr, nullptr, &edge_count);
    if (status != cudaSuccess) return status;
    if (edge_count > 16384) return cudaSuccess;
    std::vector<cudaGraphNode_t> from(edge_count), to(edge_count);
    size_t actual_edges = edge_count;
    status = cudaGraphGetEdges(graph, from.data(), to.data(), &actual_edges);
    if (status != cudaSuccess) return status;
    if (actual_edges != edge_count) return cudaErrorInvalidValue;
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    edges.reserve(edge_count);
    for (size_t i = 0; i < edge_count; ++i) {
        const auto source = indices.find(from[i]), destination = indices.find(to[i]);
        if (source == indices.end() || destination == indices.end()) return cudaErrorInvalidValue;
        edges.emplace_back(source->second, destination->second);
    }
    std::sort(edges.begin(), edges.end());
    append(edge_count);
    for (const auto & edge : edges) { append(edge.first); append(edge.second); }
    *hint = hash ? hash : 1;
    *nodes = count;
    return cudaSuccess;
}

struct context {
    explicit context(int ordinal)
        : device(ordinal), generation(next_context_generation.fetch_add(1,
                    std::memory_order_relaxed)), lifecycle(generation) {}
    int device;
    const uint64_t generation;
    ac922_v441::context_gate lifecycle;
    std::mutex mutex;
    std::unordered_map<const void *, entry> entries;
    std::unordered_map<const void *, std::shared_ptr<shared_key_entry>> shared_entries;
    std::unordered_map<slot_key, std::shared_ptr<shared_exec_slot>, slot_key_hash> shared_slots;
};

struct owner_device_key {
    const void * owner = nullptr;
    int device = -1;
    bool operator==(const owner_device_key & other) const noexcept {
        return owner == other.owner && device == other.device;
    }
};

struct owner_device_hash {
    size_t operator()(const owner_device_key & key) const noexcept {
        const size_t owner_hash = std::hash<const void *>{}(key.owner);
        const size_t device_hash = std::hash<int>{}(key.device);
        return owner_hash ^ (device_hash + size_t(0x9e3779b9) +
                (owner_hash << 6) + (owner_hash >> 2));
    }
};

struct registry {
    std::mutex mutex;
    // Keep the R439 owner-only namespace for the default path. Shared mode
    // adds device to the identity so one opaque owner can safely span devices.
    std::unordered_map<const void *, std::shared_ptr<context>> contexts;
    std::unordered_map<owner_device_key, std::shared_ptr<context>, owner_device_hash>
            shared_contexts;
};

registry & states() {
    static registry value;
    return value;
}

cudaError_t launch_shared_impl(const void * owner, const void * key,
        cudaGraph_t captured, int invalidate, cudaStream_t stream, int device, int * outcome);
cudaError_t can_prepare_shared_impl(const void * owner, const void * key,
        int device, int * can_prepare);
cudaError_t prepare_shared_impl(const void * owner, const void * key,
        cudaGraph_t captured, cudaStream_t stream, int device, int * prepared);
cudaError_t ready_shared_impl(const void * owner, const void * key,
        cudaGraph_t captured, int device, int * is_ready);
cudaError_t drop_context_shared(const void * owner);

cudaError_t retire(entry & value) {
    cudaError_t first = cudaSuccess;
    auto remember = [&](cudaError_t status) {
        if (first == cudaSuccess && status != cudaSuccess) first = status;
    };
    // The captured graph may still be queued. Retain the done fence before destroying it.
    if (value.launch_pending_unfenced && value.stream)
        remember(cudaStreamSynchronize(value.stream));
    if (value.done_recorded) remember(cudaEventSynchronize(value.done));
    if (value.child) remember(cudaGraphExecDestroy(value.child));
    if (value.done) remember(cudaEventDestroy(value.done));
    value = {};
    return first;
}

bool may_decline(cudaError_t error) {
    return error == cudaErrorNotSupported || error == cudaErrorInvalidValue ||
            error == cudaErrorMemoryAllocation || error == cudaErrorInvalidConfiguration;
}

bool stage_timing_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("AC922_TAIL_STAGE_TIMING");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

bool read_thread_cpu_ns(int64_t * value) {
#if defined(__linux__) && defined(CLOCK_THREAD_CPUTIME_ID)
    const int saved_errno = errno;
    timespec sample{};
    const int status = clock_gettime(CLOCK_THREAD_CPUTIME_ID, &sample);
    errno = saved_errno;
    if (status != 0) return false;
    *value = int64_t(sample.tv_sec) * 1000000000LL + int64_t(sample.tv_nsec);
    return true;
#else
    (void) value;
    return false;
#endif
}

cudaError_t instantiate_host_child(cudaGraphExec_t * child, cudaGraph_t captured,
        int device) {
    if (!stage_timing_enabled()) {
        return cudaGraphInstantiateWithFlags(child, captured, 0);
    }

    const auto wall_begin = std::chrono::steady_clock::now();
    int64_t cpu_begin = 0;
    const bool cpu_begin_valid = read_thread_cpu_ns(&cpu_begin);
    const cudaError_t status = cudaGraphInstantiateWithFlags(child, captured, 0);
    int64_t cpu_after = 0;
    const bool cpu_after_valid = read_thread_cpu_ns(&cpu_after);
    const auto wall_end = std::chrono::steady_clock::now();
    const int64_t wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
            wall_end - wall_begin).count();
    const bool valid = cpu_begin_valid && cpu_after_valid && cpu_after >= cpu_begin;
    const int64_t thread_cpu_us = valid ? (cpu_after - cpu_begin) / 1000 : -1;
    std::fprintf(stderr,
            "ac922-tail-stage-v439 phase=child_graph_instantiate device=%d status=%d "
            "wall_us=%lld thread_cpu_us=%lld diagnostic_valid=%d controller_allocated=0\n",
            device, int(status), (long long) wall_us, (long long) thread_cpu_us,
            int(valid));
    return status;
}

cudaError_t prepare(entry & value, int device) {
    size_t count = 0;
    cudaError_t status = cudaGraphGetNodes(value.captured, nullptr, &count);
    if (status != cudaSuccess) return status;
    std::vector<cudaGraphNode_t> nodes(count);
    status = cudaGraphGetNodes(value.captured, nodes.data(), &count);
    if (status != cudaSuccess) return status;
    for (const auto node : nodes) {
        cudaGraphNodeType type;
        status = cudaGraphNodeGetType(node, &type);
        if (status != cudaSuccess) return status;
        if (type != cudaGraphNodeTypeKernel && type != cudaGraphNodeTypeMemcpy &&
                type != cudaGraphNodeTypeMemset && type != cudaGraphNodeTypeGraph) {
            std::fprintf(stderr, "ac922-tail-bridge node_declined type=%d total_nodes=%zu scope=unchanged_graph\n", int(type), count);
            value.outcome = AC922_TAIL_UNSUPPORTED_NODE;
            return cudaSuccess;
        }
    }
    // Upload is metadata preparation, not execution of the captured graph.
    // The compute stream already contains input/cross-device dependencies.
    // Draining it here serializes the pipeline while admitting a fresh key.
    // A non-blocking private stream avoids both those waits and legacy-stream
    // coupling. Uploads finish before the captured graph is enqueued.
    cudaStream_t upload_stream=nullptr;
    status=cudaStreamCreateWithFlags(&upload_stream,cudaStreamNonBlocking);
    if(status!=cudaSuccess)return status;
    auto prepare_uploads=[&]() -> cudaError_t {
        cudaError_t next=instantiate_host_child(&value.child,value.captured,device);
        if(next!=cudaSuccess)return next;
        next=cudaGraphUpload(value.child,upload_stream);
        if(next!=cudaSuccess)return next;
        return cudaEventCreateWithFlags(&value.done,cudaEventDisableTiming);
    };
    const cudaError_t prepared=prepare_uploads();
    const auto wait_begin=std::chrono::steady_clock::now();
    // Drain even on failure before outer retire() can destroy an uploaded exec.
    const cudaError_t drained=cudaStreamSynchronize(upload_stream);
    const auto wait_end=std::chrono::steady_clock::now();
    const cudaError_t destroyed=cudaStreamDestroy(upload_stream);
    if(drained!=cudaSuccess)return drained;
    if(prepared!=cudaSuccess)return prepared;
    if(destroyed!=cudaSuccess)return destroyed;
    std::fprintf(stderr,"ac922-tail-bridge upload_isolated=1 upload_wait_us=%lld compute_stream_not_drained=1\n",
        (long long)std::chrono::duration_cast<std::chrono::microseconds>(wait_end-wait_begin).count());
    value.outcome = AC922_TAIL_LAUNCHED;
    return cudaSuccess;
}

cudaError_t launch_impl(const void * owner, const void * key, cudaGraph_t captured,
        cudaGraphExec_t generation, bool captured_only, int invalidate,
        cudaStream_t stream, int device, int * outcome) {
    if (!owner || !key || !captured || (!captured_only && !generation) || !outcome) return cudaErrorInvalidValue;
    if (captured_only && host_shared_exec_enabled())
        return launch_shared_impl(owner, key, captured, invalidate, stream, device, outcome);
    *outcome = AC922_TAIL_INSTANTIATION_DECLINED;
    int current = -1;
    cudaError_t status = cudaGetDevice(&current);
    if (status != cudaSuccess) return status;
    if (current != device) return cudaErrorInvalidDevice;
    // Do not turn an earlier asynchronous failure into an admission fallback.
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) return status;
    std::shared_ptr<context> state;
    {
        auto & all = states();
        std::lock_guard<std::mutex> lock(all.mutex);
        auto & slot = all.contexts[owner];
        if (!slot) slot = std::make_shared<context>(device);
        state = slot;
    }
    ac922_v441::context_guard lifecycle_guard(state->lifecycle,
            host_shared_exec_enabled());
    if (!lifecycle_guard) return cudaErrorNotReady;
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->device != device) return cudaErrorInvalidDevice;
    auto found = state->entries.find(key);
    if (found == state->entries.end()) {
        if (state->entries.size() >= k_max_entries) {
            *outcome = AC922_TAIL_CACHE_CAPACITY;
            return cudaSuccess;
        }
        found = state->entries.emplace(key, entry{}).first;
    }
    auto & value = found->second;
    if (value.attempted && (invalidate || value.captured != captured ||
            value.generation != generation || value.captured_only != captured_only)) {
        status = retire(value);
        if (status != cudaSuccess) return status;
    }
    if (!value.attempted) {
        value.attempted = true;
        value.captured = captured;
        value.generation = generation;
        value.captured_only = captured_only;
        value.stream = stream;
        status = prepare(value, device);
        if (status != cudaSuccess) {
            const cudaError_t original_error = status;
            // R385 prepare() drains its temporary upload stream on every exit.
            // The captured-only path has queued no work on the parent compute stream.
            // Preserve the conservative V1 drain behavior unchanged.
            const cudaError_t drained = captured_only ? cudaSuccess : cudaStreamSynchronize(stream);
            const cudaError_t cleaned = retire(value);
            if (drained != cudaSuccess) return drained;
            if (cleaned != cudaSuccess) return cleaned;
            if (!may_decline(original_error)) return original_error;
            const cudaError_t pending = cudaPeekAtLastError();
            if (pending != cudaSuccess && pending != original_error) return pending;
            if (pending == original_error) (void) cudaGetLastError();
            value.attempted = true;
            value.captured = captured;
            value.generation = generation;
            value.captured_only = captured_only;
            value.stream = stream;
            value.outcome = AC922_TAIL_INSTANTIATION_DECLINED;
            std::fprintf(stderr, "ac922-tail-bridge admission requested=1 selected=0 device=%d error=%d text=%s\n",
                    device, int(original_error), cudaGetErrorString(original_error));
        } else {
            std::fprintf(stderr, "ac922-tail-bridge admission requested=1 selected=%d device=%d outcome=%d entries=%zu owner=%p key=%p\n",
                    value.outcome == AC922_TAIL_LAUNCHED, device, value.outcome, state->entries.size(), owner, key);
        }
    }
    if (value.outcome != AC922_TAIL_LAUNCHED) {
        *outcome = value.outcome;
        return cudaSuccess;
    }
    if (value.stream != stream) {
        status = retire(value);
        if (status != cudaSuccess) return status;
        value.attempted = true;
        value.captured = captured;
        value.generation = generation;
        value.captured_only = captured_only;
        value.stream = stream;
        value.outcome = AC922_TAIL_STREAM_CHANGED;
        *outcome = value.outcome;
        return cudaSuccess;
    }
    if (value.launch_pending_unfenced) return cudaErrorNotReady;
    // Same already-instantiated captured math, with an ordinary host-launch graph exec.
    // This helper never launches that child from the device, so modes cannot overlap.
    static std::once_flag launch_mode_log;
    std::call_once(launch_mode_log, [] {
        std::fprintf(stderr, "ac922-tail-bridge launch_mode=host_child_single_exec controller_allocated=0 captured_math_unchanged=1\n");
    });
    status = cudaGraphLaunch(value.child, stream);
    if (status != cudaSuccess) return status;
    value.launch_pending_unfenced = true;
    status = cudaEventRecord(value.done, stream);
    if (status != cudaSuccess) return status;
    value.done_recorded = true;
    value.launch_pending_unfenced = false;
    *outcome = AC922_TAIL_LAUNCHED;
    return cudaSuccess;
}

void log_v3_prepare(int prepared, const char * reason, int device,
        const void * owner, const void * key, cudaError_t status = cudaSuccess) {
    // Shared mode reports slot totals at owner teardown. Keep successful
    // per-ubatch prepare hits out of stderr; retain declines and errors.
    if (host_shared_exec_enabled() && prepared) return;
    std::fprintf(stderr,
            "ac922-tail-bridge v3_prepare requested=1 prepared=%d reason=%s device=%d owner=%p key=%p status=%d\n",
            prepared, reason, device, const_cast<void *>(owner),
            const_cast<void *>(key), int(status));
}

cudaError_t validate_v3_request(const void * owner, const void * key, int device) {
    if (!owner || !key) return cudaErrorInvalidValue;
    int current = -1;
    cudaError_t status = cudaGetDevice(&current);
    if (status != cudaSuccess) return status;
    if (current != device) return cudaErrorInvalidDevice;
    // Preserve prior asynchronous errors; never turn them into a decline.
    return cudaPeekAtLastError();
}

cudaError_t entry_is_quiescent(entry & value, bool * quiescent) {
    *quiescent = false;
    if (value.launch_pending_unfenced) return cudaSuccess;
    if (value.done_recorded) {
        const cudaError_t status = cudaEventQuery(value.done);
        if (status == cudaErrorNotReady) return cudaSuccess;
        if (status != cudaSuccess) return status;
    }
    *quiescent = true;
    return cudaSuccess;
}

cudaError_t shared_context_for(const void * owner, int device, bool create,
        std::shared_ptr<context> * result) {
    if (!owner || !result) return cudaErrorInvalidValue;
    result->reset();
    auto & all = states();
    std::lock_guard<std::mutex> lock(all.mutex);
    const owner_device_key context_key{owner, device};
    const auto found = all.shared_contexts.find(context_key);
    if (found != all.shared_contexts.end()) {
        *result = found->second;
    } else if (create) {
        auto state = std::make_shared<context>(device);
        all.shared_contexts.emplace(context_key, state);
        *result = std::move(state);
    }
    return cudaSuccess;
}

cudaError_t ensure_entry_event(entry & value) {
    if (value.done) return cudaSuccess;
    return cudaEventCreateWithFlags(&value.done, cudaEventDisableTiming);
}

void set_shared_binding(shared_key_entry & key_entry, cudaGraph_t captured,
        cudaStream_t stream) {
    auto & binding = key_entry.binding;
    binding.attempted = true;
    binding.captured = captured;
    binding.generation = nullptr;
    binding.captured_only = true;
    binding.stream = stream;
    binding.outcome = AC922_TAIL_LAUNCHED;
}

struct shared_handles {
    std::shared_ptr<context> state;
    std::shared_ptr<shared_key_entry> key_entry;
    std::shared_ptr<shared_exec_slot> slot;
    std::unique_lock<std::mutex> state_lock;
    std::unique_lock<std::mutex> key_lock;
    std::unique_lock<std::mutex> slot_lock;
    bool key_created = false;
    bool slot_created = false;
};

// Lock order for shared mode is context -> per-key -> per-stream slot. The
// context lock is released only after both stable shared_ptr targets are locked.
cudaError_t admit_shared_handles(const std::shared_ptr<context> & state,
        const void * key, cudaStream_t stream, cudaGraph_t captured,
        int device, bool check_quiescence,
        shared_handles * handles, bool * declined, int * decline_outcome) {
    if (!state || !key || !handles || !declined || !decline_outcome)
        return cudaErrorInvalidValue;
    *declined = false;
    *decline_outcome = AC922_TAIL_INSTANTIATION_DECLINED;

    handles->state = state;
    handles->state_lock = std::unique_lock<std::mutex>(state->mutex);
    if (state->device != device) return cudaErrorInvalidDevice;
    auto key_it = state->shared_entries.find(key);
    if (key_it == state->shared_entries.end()) {
        if (state->entries.size() + state->shared_entries.size() >= k_max_entries) {
            *declined = true;
            *decline_outcome = AC922_TAIL_CACHE_CAPACITY;
            return cudaSuccess;
        }
        auto inserted = state->shared_entries.emplace(key,
                std::make_shared<shared_key_entry>());
        key_it = inserted.first;
        handles->key_created = true;
    }
    handles->key_entry = key_it->second;
    handles->key_lock = std::unique_lock<std::mutex>(handles->key_entry->mutex);

    auto check_binding = [&](entry & binding, bool stream_sensitive) -> cudaError_t {
        if (!binding.attempted) return cudaSuccess;
        if (stream_sensitive && binding.stream != stream) {
            bool quiescent = false;
            cudaError_t status = entry_is_quiescent(binding, &quiescent);
            if (status != cudaSuccess) return status;
            if (!quiescent) {
                *declined = true;
                *decline_outcome = AC922_TAIL_STREAM_CHANGED;
                return cudaSuccess;
            }
            status = retire(binding);
            if (status != cudaSuccess) return status;
            return cudaSuccess;
        }
        if (check_quiescence) {
            bool quiescent = false;
            cudaError_t status = entry_is_quiescent(binding, &quiescent);
            if (status != cudaSuccess) return status;
            if (!quiescent) {
                *declined = true;
                *decline_outcome = AC922_TAIL_INSTANTIATION_DECLINED;
            }
        } else if (binding.launch_pending_unfenced) {
            return cudaErrorNotReady;
        }
        return cudaSuccess;
    };

    cudaError_t status = check_binding(handles->key_entry->binding, true);
    if (status != cudaSuccess || *declined) return status;
    status = check_binding(handles->key_entry->fallback, true);
    if (status != cudaSuccess || *declined) return status;

    uint64_t bucket = 0;
    size_t graph_nodes = 0;
    status = topology_hint(captured, &bucket, &graph_nodes);
    if (status != cudaSuccess) return status;
    const slot_key selected_slot{stream, bucket};
    auto slot_it = state->shared_slots.find(selected_slot);
    if (slot_it == state->shared_slots.end()) {
        if (state->shared_slots.size() >= k_max_entries) {
            if (handles->key_created) state->shared_entries.erase(key_it);
            handles->key_lock.unlock();
            handles->key_entry.reset();
            *declined = true;
            *decline_outcome = AC922_TAIL_CACHE_CAPACITY;
            return cudaSuccess;
        }
        auto inserted = state->shared_slots.emplace(selected_slot,
                std::make_shared<shared_exec_slot>(state->generation, stream, bucket, graph_nodes));
        slot_it = inserted.first;
        handles->slot_created = true;
        if (topology_buckets_enabled()) {
            std::fprintf(stderr, "ac922-tail-bridge topology_bucket requested=1 selected=%d"
                    " device=%d owner=%p stream=%p bucket=%llu nodes=%zu cap=32"
                    " replay_authorized=0 update_validation_required=1 reason=%s\n",
                    int(bucket != 0), device, (void *) state.get(), (void *) stream,
                    (unsigned long long) bucket, graph_nodes,
                    bucket ? "flat_node_types_and_edges" : "unclassified_or_bounded_legacy_slot");
        }
    }
    handles->slot = slot_it->second;
    handles->slot_lock = std::unique_lock<std::mutex>(handles->slot->mutex);
    if (handles->slot->generation != state->generation)
        return cudaErrorInvalidResourceHandle;
    if (handles->slot->launch_pending_unfenced) return cudaErrorNotReady;

    handles->state_lock.unlock();
    return cudaSuccess;
}

void commit_slot_binding(shared_exec_slot & slot, const void * key,
        cudaGraph_t captured) {
    slot.committed_key = key;
    slot.committed_graph = captured;
    ++slot.commit_serial;
}

cudaError_t clear_known_update_failure() {
    const cudaError_t pending = cudaPeekAtLastError();
    if (pending == cudaSuccess) return cudaSuccess;
    if (pending != cudaErrorGraphExecUpdateFailure) return pending;
    // Clear only the exact expected update-failure status. If an unrelated
    // error is observed, return it instead of consuming or masking it.
    const cudaError_t cleared = cudaGetLastError();
    if (cleared == cudaSuccess || cleared == cudaErrorGraphExecUpdateFailure)
        return cudaSuccess;
    return cleared;
}

// On first use this prepares an ordinary flags-zero host child and moves it
// into the shared slot. On later uses it always updates from this call's graph.
// Only the documented exact update-failure code selects a per-key fallback.
cudaError_t bind_shared_slot(shared_exec_slot & slot, const void * key,
        cudaGraph_t captured, int device, bool * update_failure,
        int * decline_outcome) {
    if (!update_failure || !decline_outcome) return cudaErrorInvalidValue;
    *update_failure = false;
    *decline_outcome = AC922_TAIL_INSTANTIATION_DECLINED;
    if (slot.launch_pending_unfenced) return cudaErrorNotReady;

    if (!slot.exec) {
        entry initial;
        initial.attempted = true;
        initial.captured = captured;
        initial.captured_only = true;
        initial.stream = slot.stream;
        cudaError_t status = prepare(initial, device);
        if (status != cudaSuccess) {
            const cudaError_t cleaned = retire(initial);
            if (cleaned != cudaSuccess) return cleaned;
            if (!may_decline(status)) return status;
            *decline_outcome = AC922_TAIL_INSTANTIATION_DECLINED;
            return cudaSuccess;
        }
        if (initial.outcome != AC922_TAIL_LAUNCHED || !initial.child || !initial.done) {
            *decline_outcome = initial.outcome;
            const cudaError_t cleaned = retire(initial);
            return cleaned;
        }
        slot.exec = initial.child;
        slot.done = initial.done;
        initial.child = nullptr;
        initial.done = nullptr;
        slot.instantiations = 1;
        commit_slot_binding(slot, key, captured);
        return cudaSuccess;
    }

    cudaGraphExecUpdateResultInfo update_info{};
    const cudaError_t status = cudaGraphExecUpdate(slot.exec, captured, &update_info);
    const auto decision = ac922_v441::classify_update_status(int(status),
            int(cudaSuccess), int(cudaErrorGraphExecUpdateFailure));
    if (decision == ac922_v441::update_decision::commit_shared) {
        ++slot.update_successes;
        commit_slot_binding(slot, key, captured);
        return cudaSuccess;
    }
    if (decision == ac922_v441::update_decision::use_per_key_fallback) {
        ++slot.update_failures;
        const cudaError_t error_state = clear_known_update_failure();
        if (error_state != cudaSuccess) return error_state;
        *update_failure = true;
        if (!slot.first_update_failure_logged) {
            std::fprintf(stderr,
                    "ac922-tail-bridge v441_update_failure slot=%p failed_key=%p failed_graph=%p "
                    "committed_key=%p committed_graph=%p commit_serial=%llu result=%d\n",
                    (void *) &slot, const_cast<void *>(key), (void *) captured,
                    const_cast<void *>(slot.committed_key), (void *) slot.committed_graph,
                    (unsigned long long) slot.commit_serial, int(update_info.result));
            slot.first_update_failure_logged = true;
        }
        return cudaSuccess;
    }
    // cudaGraphExecUpdate can also surface prior asynchronous errors. Do not
    // reinterpret those errors as a topology mismatch or fallback decision.
    return status;
}

cudaError_t ensure_shared_key_event(shared_key_entry & key_entry) {
    return ensure_entry_event(key_entry.binding);
}

cudaError_t ensure_per_key_fallback(shared_key_entry & key_entry,
        cudaGraph_t captured, cudaStream_t stream, int device,
        bool * available, int * decline_outcome) {
    if (!available || !decline_outcome) return cudaErrorInvalidValue;
    *available = false;
    *decline_outcome = AC922_TAIL_INSTANTIATION_DECLINED;
    auto & fallback = key_entry.fallback;
    const bool exact = fallback.attempted && fallback.captured == captured &&
            fallback.generation == nullptr && fallback.captured_only &&
            fallback.stream == stream;
    if (exact && fallback.outcome == AC922_TAIL_LAUNCHED &&
            fallback.child && fallback.done) {
        if (fallback.launch_pending_unfenced) return cudaErrorNotReady;
        *available = true;
        return cudaSuccess;
    }
    if (exact && (fallback.outcome == AC922_TAIL_UNSUPPORTED_NODE ||
            fallback.outcome == AC922_TAIL_INSTANTIATION_DECLINED)) {
        *decline_outcome = fallback.outcome;
        return cudaSuccess;
    }
    if (fallback.attempted) {
        cudaError_t status = retire(fallback);
        if (status != cudaSuccess) return status;
    }
    fallback.attempted = true;
    fallback.captured = captured;
    fallback.generation = nullptr;
    fallback.captured_only = true;
    fallback.stream = stream;
    cudaError_t status = prepare(fallback, device);
    if (status != cudaSuccess) {
        const cudaError_t original = status;
        const cudaError_t cleaned = retire(fallback);
        if (cleaned != cudaSuccess) return cleaned;
        if (!may_decline(original)) return original;
        fallback.attempted = true;
        fallback.captured = captured;
        fallback.generation = nullptr;
        fallback.captured_only = true;
        fallback.stream = stream;
        fallback.outcome = AC922_TAIL_INSTANTIATION_DECLINED;
        *decline_outcome = fallback.outcome;
        return cudaSuccess;
    }
    if (fallback.outcome == AC922_TAIL_LAUNCHED && fallback.child && fallback.done) {
        *available = true;
        return cudaSuccess;
    }
    *decline_outcome = fallback.outcome;
    return cudaSuccess;
}

void set_shared_key_binding(shared_key_entry & key_entry, cudaGraph_t captured,
        cudaStream_t stream, uint64_t topology_bucket) {
    set_shared_binding(key_entry, captured, stream);
    key_entry.topology_bucket = topology_bucket;
}

cudaError_t record_shared_launch(shared_exec_slot & slot, cudaStream_t stream) {
    slot.launch_pending_unfenced = true;
    const cudaError_t status = cudaEventRecord(slot.done, stream);
    if (status != cudaSuccess) return status;
    slot.done_recorded = true;
    slot.launch_pending_unfenced = false;
    return cudaSuccess;
}

cudaError_t record_binding_launch(entry & binding, cudaStream_t stream) {
    binding.launch_pending_unfenced = true;
    const cudaError_t status = cudaEventRecord(binding.done, stream);
    if (status != cudaSuccess) return status;
    binding.done_recorded = true;
    binding.launch_pending_unfenced = false;
    return cudaSuccess;
}

cudaError_t record_fallback_launch(entry & fallback, cudaStream_t stream) {
    fallback.launch_pending_unfenced = true;
    const cudaError_t status = cudaEventRecord(fallback.done, stream);
    if (status != cudaSuccess) return status;
    fallback.done_recorded = true;
    fallback.launch_pending_unfenced = false;
    return cudaSuccess;
}

cudaError_t can_prepare_shared_impl(const void * owner, const void * key,
        int device, int * can_prepare) {
    if (!can_prepare) return cudaErrorInvalidValue;
    *can_prepare = 0;
    const cudaError_t validation = validate_v3_request(owner, key, device);
    if (validation != cudaSuccess) return validation;

    std::shared_ptr<context> state;
    cudaError_t status = shared_context_for(owner, device, false, &state);
    if (status != cudaSuccess) return status;
    if (!state) {
        *can_prepare = 1;
        return cudaSuccess;
    }
    ac922_v441::context_guard lifecycle_guard(state->lifecycle);
    if (!lifecycle_guard) return cudaErrorNotReady;

    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (state->device != device) return cudaErrorInvalidDevice;
    const auto found = state->shared_entries.find(key);
    if (found == state->shared_entries.end()) {
        *can_prepare = state->entries.size() + state->shared_entries.size() < k_max_entries;
        return cudaSuccess;
    }
    std::lock_guard<std::mutex> key_lock(found->second->mutex);
    bool binding_quiet = true;
    status = entry_is_quiescent(found->second->binding, &binding_quiet);
    if (status != cudaSuccess) return status;
    bool fallback_quiet = true;
    status = entry_is_quiescent(found->second->fallback, &fallback_quiet);
    if (status != cudaSuccess) return status;
    *can_prepare = binding_quiet && fallback_quiet;
    return cudaSuccess;
}

cudaError_t prepare_shared_impl(const void * owner, const void * key,
        cudaGraph_t captured, cudaStream_t stream, int device, int * prepared) {
    if (!prepared) return cudaErrorInvalidValue;
    *prepared = 0;
    if (!captured) return cudaErrorInvalidValue;
    cudaError_t status = validate_v3_request(owner, key, device);
    if (status != cudaSuccess) return status;

    std::shared_ptr<context> state;
    status = shared_context_for(owner, device, true, &state);
    if (status != cudaSuccess) return status;
    ac922_v441::context_guard lifecycle_guard(state->lifecycle);
    if (!lifecycle_guard) return cudaErrorNotReady;

    shared_handles handles;
    bool declined = false;
    int decline_outcome = AC922_TAIL_INSTANTIATION_DECLINED;
    status = admit_shared_handles(state, key, stream, captured, device, true, &handles,
            &declined, &decline_outcome);
    if (status != cudaSuccess) {
        log_v3_prepare(0, "cuda-error", device, owner, key, status);
        return status;
    }
    if (declined) {
        log_v3_prepare(0, decline_outcome == AC922_TAIL_CACHE_CAPACITY ?
                "capacity" : "in-flight", device, owner, key);
        return cudaSuccess;
    }

    bool update_failure = false;
    status = bind_shared_slot(*handles.slot, key, captured, device,
            &update_failure, &decline_outcome);
    if (status != cudaSuccess) {
        log_v3_prepare(0, "cuda-error", device, owner, key, status);
        return status;
    }
    if (!handles.slot->exec) {
        log_v3_prepare(0, "unsupported-or-declined", device, owner, key);
        return cudaSuccess;
    }

    if (update_failure) {
        bool fallback_available = false;
        status = ensure_per_key_fallback(*handles.key_entry, captured, stream,
                device, &fallback_available, &decline_outcome);
        if (status != cudaSuccess) {
            log_v3_prepare(0, "cuda-error", device, owner, key, status);
            return status;
        }
        if (!fallback_available) {
            log_v3_prepare(0, "topology-fallback-declined", device, owner, key);
            return cudaSuccess;
        }
    }
    set_shared_key_binding(*handles.key_entry, captured, stream, handles.slot->topology_bucket);
    *prepared = 1;
    log_v3_prepare(1, update_failure ? "prepared-per-key-fallback" :
            "prepared-shared-host-child-flags0", device, owner, key);
    return cudaSuccess;
}

cudaError_t launch_shared_impl(const void * owner, const void * key,
        cudaGraph_t captured, int invalidate, cudaStream_t stream, int device,
        int * outcome) {
    if (!outcome) return cudaErrorInvalidValue;
    *outcome = AC922_TAIL_INSTANTIATION_DECLINED;
    if (!owner || !key || !captured) return cudaErrorInvalidValue;
    int current = -1;
    cudaError_t status = cudaGetDevice(&current);
    if (status != cudaSuccess) return status;
    if (current != device) return cudaErrorInvalidDevice;
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) return status;

    std::shared_ptr<context> state;
    status = shared_context_for(owner, device, true, &state);
    if (status != cudaSuccess) return status;
    ac922_v441::context_guard lifecycle_guard(state->lifecycle);
    if (!lifecycle_guard) return cudaErrorNotReady;

    shared_handles handles;
    bool declined = false;
    int decline_outcome = AC922_TAIL_INSTANTIATION_DECLINED;
    status = admit_shared_handles(state, key, stream, captured, device, false, &handles,
            &declined, &decline_outcome);
    if (status != cudaSuccess) return status;
    if (declined) {
        *outcome = decline_outcome;
        return cudaSuccess;
    }

    if (invalidate && handles.key_entry->fallback.attempted) {
        status = retire(handles.key_entry->fallback);
        if (status != cudaSuccess) return status;
    }

    bool update_failure = false;
    status = bind_shared_slot(*handles.slot, key, captured, device,
            &update_failure, &decline_outcome);
    if (status != cudaSuccess) return status;
    if (!handles.slot->exec) {
        *outcome = decline_outcome;
        return cudaSuccess;
    }

    if (update_failure) {
        bool fallback_available = false;
        status = ensure_per_key_fallback(*handles.key_entry, captured, stream,
                device, &fallback_available, &decline_outcome);
        if (status != cudaSuccess) return status;
        if (!fallback_available) {
            *outcome = decline_outcome;
            return cudaSuccess;
        }
        status = ensure_shared_key_event(*handles.key_entry);
        if (status != cudaSuccess) return status;
        status = cudaGraphLaunch(handles.key_entry->fallback.child, stream);
        if (status != cudaSuccess) return status;
        ++handles.slot->fallback_launches;
        set_shared_key_binding(*handles.key_entry, captured, stream, handles.slot->topology_bucket);
        const cudaError_t binding_record = record_binding_launch(
                handles.key_entry->binding, stream);
        status = record_fallback_launch(handles.key_entry->fallback, stream);
        if (binding_record != cudaSuccess) return binding_record;
        if (status != cudaSuccess) return status;
    } else {
        status = ensure_shared_key_event(*handles.key_entry);
        if (status != cudaSuccess) return status;
        set_shared_key_binding(*handles.key_entry, captured, stream, handles.slot->topology_bucket);
        status = cudaGraphLaunch(handles.slot->exec, stream);
        if (status != cudaSuccess) return status;
        const cudaError_t binding_record = record_binding_launch(
                handles.key_entry->binding, stream);
        status = record_shared_launch(*handles.slot, stream);
        if (binding_record != cudaSuccess) return binding_record;
        if (status != cudaSuccess) return status;
    }
    *outcome = AC922_TAIL_LAUNCHED;
    return cudaSuccess;
}

cudaError_t ready_shared_impl(const void * owner, const void * key,
        cudaGraph_t captured, int device, int * is_ready) {
    if (!is_ready) return cudaErrorInvalidValue;
    *is_ready = 0;
    if (!owner || !key || !captured) return cudaErrorInvalidValue;

    std::shared_ptr<context> state;
    const cudaError_t status = shared_context_for(owner, device, false, &state);
    if (status != cudaSuccess || !state) return status;
    ac922_v441::context_guard lifecycle_guard(state->lifecycle);
    if (!lifecycle_guard) return cudaErrorNotReady;

    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (state->device != device) return cudaErrorInvalidDevice;
    const auto key_it = state->shared_entries.find(key);
    if (key_it != state->shared_entries.end()) {
        std::lock_guard<std::mutex> key_lock(key_it->second->mutex);
        const entry & binding = key_it->second->binding;
        const entry & fallback = key_it->second->fallback;
        if (fallback.attempted && fallback.captured_only &&
                fallback.captured == captured &&
                fallback.outcome == AC922_TAIL_LAUNCHED && fallback.child &&
                fallback.done && !fallback.launch_pending_unfenced) {
            *is_ready = 1;
            return cudaSuccess;
        }
        if (binding.attempted && binding.captured_only &&
                binding.captured == captured &&
                binding.outcome == AC922_TAIL_LAUNCHED &&
                !binding.launch_pending_unfenced) {
            const auto slot_it = state->shared_slots.find(slot_key{binding.stream, key_it->second->topology_bucket});
            if (slot_it != state->shared_slots.end()) {
                auto & slot = *slot_it->second;
                std::lock_guard<std::mutex> slot_lock(slot.mutex);
                if (slot.generation == state->generation && slot.exec &&
                        !slot.launch_pending_unfenced) {
                    // Another key may currently occupy this shared executable.
                    // Every actual V2 launch rebinds from its supplied graph.
                    *is_ready = 1;
                }
            }
        }
    }
    return cudaSuccess;
}

cudaError_t retire_shared_slot(shared_exec_slot & slot) {
    cudaError_t first = cudaSuccess;
    auto remember = [&](cudaError_t status) {
        if (first == cudaSuccess && status != cudaSuccess) first = status;
    };
    if (slot.launch_pending_unfenced && slot.stream)
        remember(cudaStreamSynchronize(slot.stream));
    if (slot.done_recorded && slot.done) remember(cudaEventSynchronize(slot.done));
    if (slot.exec) remember(cudaGraphExecDestroy(slot.exec));
    if (slot.done) remember(cudaEventDestroy(slot.done));
    slot.exec = nullptr;
    slot.done = nullptr;
    slot.done_recorded = false;
    slot.launch_pending_unfenced = false;
    return first;
}

cudaError_t drop_context_shared(const void * owner) {
    if (!owner) return cudaErrorInvalidValue;
    auto & all = states();
    std::vector<std::shared_ptr<context>> victims;
    {
        std::lock_guard<std::mutex> registry_lock(all.mutex);
        const auto legacy = all.contexts.find(owner);
        if (legacy != all.contexts.end()) victims.push_back(legacy->second);
        for (const auto & item : all.shared_contexts) {
            if (item.first.owner == owner) victims.push_back(item.second);
        }
        if (victims.empty()) return cudaSuccess;
        // Keep each closing generation in the registry until its resources
        // are drained, so late lookups cannot create a parallel context.
        for (const auto & state : victims) state->lifecycle.close();
    }
    for (const auto & state : victims) state->lifecycle.wait_until_idle();

    int previous = -1;
    cudaError_t status = cudaGetDevice(&previous);
    if (status != cudaSuccess) return status;

    cudaError_t first = cudaSuccess;
    auto remember = [&](cudaError_t next) {
        if (first == cudaSuccess && next != cudaSuccess) first = next;
    };
    std::vector<std::shared_ptr<context>> retired;
    for (const auto & state : victims) {
        status = cudaSetDevice(state->device);
        if (status != cudaSuccess) {
            remember(status);
            continue;
        }
        {
            std::lock_guard<std::mutex> state_lock(state->mutex);
            for (auto & item : state->entries) remember(retire(item.second));
            state->entries.clear();
            for (auto & item : state->shared_entries) {
                auto & key_entry = *item.second;
                std::lock_guard<std::mutex> key_lock(key_entry.mutex);
                remember(retire(key_entry.binding));
                remember(retire(key_entry.fallback));
            }
            state->shared_entries.clear();
            for (auto & item : state->shared_slots) {
                auto & slot = *item.second;
                std::lock_guard<std::mutex> slot_lock(slot.mutex);
                const char * summary_format = topology_buckets_enabled() ?
                        "ac922-tail-bridge v441_slot_summary owner=%p generation=%llu device=%d "
                        "stream=%p committed_key=%p committed_graph=%p commit_serial=%llu "
                        "instantiations=%zu updates=%zu update_failures=%zu fallback_launches=%zu "
                        "topology_bucket=%llu slot_nodes=%zu\n" :
                        "ac922-tail-bridge v441_slot_summary owner=%p generation=%llu device=%d "
                        "stream=%p committed_key=%p committed_graph=%p commit_serial=%llu "
                        "instantiations=%zu updates=%zu update_failures=%zu fallback_launches=%zu\n";
                std::fprintf(stderr, summary_format,
                        const_cast<void *>(owner), (unsigned long long) state->generation,
                        state->device, (void *) slot.stream,
                        const_cast<void *>(slot.committed_key), (void *) slot.committed_graph,
                        (unsigned long long) slot.commit_serial, slot.instantiations,
                        slot.update_successes, slot.update_failures, slot.fallback_launches,
                        (unsigned long long) slot.topology_bucket, slot.topology_nodes);
                remember(retire_shared_slot(slot));
            }
            state->shared_slots.clear();
        }
        retired.push_back(state);
    }
    const cudaError_t restored = cudaSetDevice(previous);
    {
        std::lock_guard<std::mutex> registry_lock(all.mutex);
        const auto legacy = all.contexts.find(owner);
        if (legacy != all.contexts.end() &&
                std::find(retired.begin(), retired.end(), legacy->second) != retired.end())
            all.contexts.erase(legacy);
        for (auto it = all.shared_contexts.begin(); it != all.shared_contexts.end();) {
            if (it->first.owner == owner &&
                    std::find(retired.begin(), retired.end(), it->second) != retired.end()) {
                it = all.shared_contexts.erase(it);
            } else {
                ++it;
            }
        }
    }
    remember(restored);
    return first;
}

cudaError_t can_prepare_impl(const void * owner, const void * key,
        int device, int * can_prepare) {
    if (host_shared_exec_enabled())
        return can_prepare_shared_impl(owner, key, device, can_prepare);
    if (!can_prepare) return cudaErrorInvalidValue;
    *can_prepare = 0;
    cudaError_t status = validate_v3_request(owner, key, device);
    if (status != cudaSuccess) return status;

    auto & all = states();
    std::shared_ptr<context> state;
    {
        std::lock_guard<std::mutex> lock(all.mutex);
        const auto found = all.contexts.find(owner);
        if (found == all.contexts.end()) {
            *can_prepare = 1;
            return cudaSuccess;
        }
        state = found->second;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->device != device) return cudaErrorInvalidDevice;
    const auto found = state->entries.find(key);
    if (found == state->entries.end()) {
        *can_prepare = state->entries.size() < k_max_entries;
        return cudaSuccess;
    }
    bool quiescent = false;
    status = entry_is_quiescent(found->second, &quiescent);
    if (status != cudaSuccess) return status;
    *can_prepare = quiescent;
    return cudaSuccess;
}

cudaError_t prepare_captured_impl(const void * owner, const void * key,
        cudaGraph_t captured, cudaStream_t stream, int device, int * prepared) {
    if (host_shared_exec_enabled())
        return prepare_shared_impl(owner, key, captured, stream, device, prepared);
    if (!prepared) return cudaErrorInvalidValue;
    *prepared = 0;
    if (!captured) return cudaErrorInvalidValue;
    cudaError_t status = validate_v3_request(owner, key, device);
    if (status != cudaSuccess) return status;

    auto & all = states();
    std::shared_ptr<context> state;
    {
        std::lock_guard<std::mutex> lock(all.mutex);
        auto & slot = all.contexts[owner];
        if (!slot) slot = std::make_shared<context>(device);
        state = slot;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->device != device) return cudaErrorInvalidDevice;

    auto found = state->entries.find(key);
    if (found == state->entries.end() && state->entries.size() >= k_max_entries) {
        log_v3_prepare(0, "capacity", device, owner, key);
        return cudaSuccess;
    }
    if (found == state->entries.end()) {
        found = state->entries.emplace(key, entry{}).first;
    }

    auto & value = found->second;
    if (value.attempted) {
        bool quiescent = false;
        status = entry_is_quiescent(value, &quiescent);
        if (status != cudaSuccess) {
            log_v3_prepare(0, "cuda-error", device, owner, key, status);
            return status;
        }
        if (!quiescent) {
            log_v3_prepare(0, "in-flight", device, owner, key);
            return cudaSuccess;
        }
    }

    const bool exact_binding = value.attempted && value.captured == captured &&
            value.generation == nullptr && value.captured_only && value.stream == stream;
    if (exact_binding && value.outcome == AC922_TAIL_LAUNCHED &&
            value.child && value.done) {
        *prepared = 1;
        log_v3_prepare(1, "exact-cache-hit-host-child-flags0", device, owner, key);
        return cudaSuccess;
    }
    if (exact_binding && (value.outcome == AC922_TAIL_UNSUPPORTED_NODE ||
            value.outcome == AC922_TAIL_INSTANTIATION_DECLINED)) {
        log_v3_prepare(0, "cached-decline", device, owner, key);
        return cudaSuccess;
    }

    if (value.attempted) {
        status = retire(value);
        if (status != cudaSuccess) {
            log_v3_prepare(0, "cuda-error", device, owner, key, status);
            return status;
        }
    }

    value.attempted = true;
    value.captured = captured;
    value.generation = nullptr;
    value.captured_only = true;
    value.stream = stream;
    try {
        // R398's existing prepare() instantiates with flags=0 and uploads on
        // its private stream. It creates the child/done-event pair only; it
        // does not launch the model graph or record the completion event.
        status = prepare(value, device);
    } catch (...) {
        (void) retire(value);
        state->entries.erase(found);
        throw;
    }
    if (status != cudaSuccess) {
        const cudaError_t cleaned = retire(value);
        state->entries.erase(found);
        if (cleaned != cudaSuccess) {
            log_v3_prepare(0, "cleanup-error", device, owner, key, cleaned);
            return cleaned;
        }
        log_v3_prepare(0, "cuda-error", device, owner, key, status);
        return status;
    }
    if (value.outcome == AC922_TAIL_LAUNCHED && value.child && value.done) {
        *prepared = 1;
        log_v3_prepare(1, "prepared-host-child-flags0", device, owner, key);
        return cudaSuccess;
    }
    log_v3_prepare(0, "unsupported-node", device, owner, key);
    return cudaSuccess;
}

cudaError_t ready_captured_only(const void * owner, const void * key, cudaGraph_t captured,
        int device, int * is_ready) {
    if (host_shared_exec_enabled())
        return ready_shared_impl(owner, key, captured, device, is_ready);
    if (!is_ready) return cudaErrorInvalidValue;
    *is_ready = 0;
    if (!owner || !key || !captured) return cudaErrorInvalidValue;
    auto & all = states();
    std::shared_ptr<context> state;
    {
        std::lock_guard<std::mutex> lock(all.mutex);
        const auto found = all.contexts.find(owner);
        if (found == all.contexts.end()) return cudaSuccess;
        state = found->second;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->device != device) return cudaErrorInvalidDevice;
    const auto found = state->entries.find(key);
    if (found == state->entries.end()) return cudaSuccess;
    const auto & value = found->second;
    *is_ready = value.attempted && value.captured_only && value.captured == captured &&
            value.outcome == AC922_TAIL_LAUNCHED && value.child != nullptr &&
            value.done != nullptr && !value.launch_pending_unfenced;
    return cudaSuccess;
}
} // namespace

extern "C" cudaError_t ac922_tail_bridge_launch(const void * owner, const void * key,
        cudaGraph_t captured, cudaGraphExec_t generation, int invalidate,
        cudaStream_t stream, int device, int * outcome) {
    try {
        return launch_impl(owner, key, captured, generation, false, invalidate, stream, device, outcome);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t ac922_tail_bridge_v2_ready(const void * owner, const void * key,
        cudaGraph_t captured, int device, int * is_ready) {
    try {
        return ready_captured_only(owner, key, captured, device, is_ready);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t ac922_tail_bridge_v2_launch_captured(const void * owner, const void * key,
        cudaGraph_t captured, int invalidate, cudaStream_t stream, int device, int * outcome) {
    try {
        return launch_impl(owner, key, captured, nullptr, true, invalidate, stream, device, outcome);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t ac922_tail_bridge_v3_prepare_captured(const void * owner,
        const void * key, cudaGraph_t captured, cudaStream_t stream,
        int device, int * prepared) {
    try {
        return prepare_captured_impl(owner, key, captured, stream, device, prepared);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t ac922_tail_bridge_v3_can_prepare(const void * owner,
        const void * key, int device, int * can_prepare) {
    try {
        return can_prepare_impl(owner, key, device, can_prepare);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t ac922_tail_bridge_drop_context(const void * owner) {
    try {
        if (host_shared_exec_enabled()) return drop_context_shared(owner);
        std::shared_ptr<context> state;
        {
            auto & all = states();
            std::lock_guard<std::mutex> lock(all.mutex);
            const auto it = all.contexts.find(owner);
            if (it == all.contexts.end()) return cudaSuccess;
            state = it->second;
            all.contexts.erase(it);
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        int previous = -1;
        cudaError_t status = cudaGetDevice(&previous);
        if (status != cudaSuccess) return status;
        status = cudaSetDevice(state->device);
        if (status != cudaSuccess) return status;
        cudaError_t first = cudaSuccess;
        for (auto & item : state->entries) {
            const cudaError_t next = retire(item.second);
            if (first == cudaSuccess && next != cudaSuccess) first = next;
        }
        state->entries.clear();
        const cudaError_t restored = cudaSetDevice(previous);
        return first == cudaSuccess ? restored : first;
    } catch (...) {
        return cudaErrorUnknown;
    }
}
