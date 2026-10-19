#include "bridge-api.h"
#include "captured-observer-private.hpp"
#include <cstdlib>
#include <cuda_runtime.h>

#include <cstdio>
#include <chrono>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {
constexpr size_t k_max_entries = 32;

// One child per graph environment; following work on the parent stream waits
// for this tail child. Computation is entirely the original captured graph.
__global__ void launch_unchanged_child(cudaGraphExec_t child) {
    if (blockIdx.x || threadIdx.x) return;
    const cudaError_t status = cudaGraphLaunch(child, cudaStreamGraphTailLaunch);
    if (status != cudaSuccess) {
        printf("ac922-tail-bridge device launch failed status=%d\n", int(status));
        asm volatile("trap;");
    }
}

struct entry {
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t generation = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t child = nullptr;
    cudaGraph_t controller_graph = nullptr;
    cudaGraphExec_t controller = nullptr;
    cudaEvent_t done = nullptr;
    bool done_recorded = false;
    bool attempted = false;
    bool captured_only = false;
    int outcome = AC922_TAIL_INSTANTIATION_DECLINED;
};

struct context {
    explicit context(int ordinal) : device(ordinal) {}
    int device;
    std::mutex mutex;
    std::unordered_map<const void *, entry> entries;
    ac922_captured_observer::history observer;
};

struct registry {
    std::mutex mutex;
    std::unordered_map<const void *, std::shared_ptr<context>> contexts;
};

registry & states() {
    static registry value;
    return value;
}

cudaError_t retire(entry & value) {
    cudaError_t first = cudaSuccess;
    auto remember = [&](cudaError_t status) {
        if (first == cudaSuccess && status != cudaSuccess) first = status;
    };
    // A controller may still be queued without having called the child yet.
    // Destroying that child's handle before this fence would be invalid.
    if (value.done_recorded) remember(cudaEventSynchronize(value.done));
    if (value.controller) remember(cudaGraphExecDestroy(value.controller));
    if (value.child) remember(cudaGraphExecDestroy(value.child));
    if (value.controller_graph) remember(cudaGraphDestroy(value.controller_graph));
    if (value.done) remember(cudaEventDestroy(value.done));
    value = {};
    return first;
}

bool may_decline(cudaError_t error) {
    return error == cudaErrorNotSupported || error == cudaErrorInvalidValue ||
            error == cudaErrorMemoryAllocation || error == cudaErrorInvalidConfiguration;
}

cudaError_t prepare(entry & value) {
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
    // coupling. Its uploads must finish before the controller may be enqueued.
    cudaStream_t upload_stream=nullptr;
    status=cudaStreamCreateWithFlags(&upload_stream,cudaStreamNonBlocking);
    if(status!=cudaSuccess)return status;
    auto prepare_uploads=[&]() -> cudaError_t {
        cudaError_t next=cudaGraphInstantiateWithFlags(&value.child,value.captured,
            cudaGraphInstantiateFlagDeviceLaunch);
        if(next!=cudaSuccess)return next;
        next=cudaGraphUpload(value.child,upload_stream);
        if(next!=cudaSuccess)return next;
        next=cudaGraphCreate(&value.controller_graph,0);
        if(next!=cudaSuccess)return next;
        cudaGraphNode_t node=nullptr;
        cudaKernelNodeParams parameters{};
        parameters.func=reinterpret_cast<void *>(launch_unchanged_child);
        parameters.gridDim=dim3(1);parameters.blockDim=dim3(1);
        void *args[]={&value.child};parameters.kernelParams=args;
        next=cudaGraphAddKernelNode(&node,value.controller_graph,nullptr,0,&parameters);
        if(next!=cudaSuccess)return next;
        next=cudaGraphInstantiateWithFlags(&value.controller,value.controller_graph,
            cudaGraphInstantiateFlagDeviceLaunch);
        if(next!=cudaSuccess)return next;
        next=cudaGraphUpload(value.controller,upload_stream);
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
        const char * observer_flag = std::getenv("AC922_CAPTURED_ARGUMENT_OBSERVER");
        const bool observer_requested = observer_flag && std::strcmp(observer_flag, "1") == 0;
        ac922_captured_observer::observation observation;
        if (observer_requested) {
            try { observation = state->observer.observe(captured, device); }
            catch (...) { observation.selected = false; observation.reason = "observer_exception_no_replay"; }
        }
        const auto prepare_begin = std::chrono::steady_clock::now();
        status = prepare(value);
        if (observer_requested) {
            const auto prepare_end = std::chrono::steady_clock::now();
            std::fprintf(stderr, "captured-argument-observer requested=1 selected=%d device=%d owner=%p key=%p"
                    " nodes=%zu edges=%zu bytes=%zu hash_low=%016llx hash_high=%016llx"
                    " seen_hash_before=%d observe_us=%llu original_prepare_us=%lld prepare_status=%d prepare_outcome=%d"
                    " prepare_begin_us=%lld prepare_end_us=%lld"
                    " replay_authorized=0 hash_is_potential_only=1 reason=%s\n",
                    int(observation.selected), device, owner, key, observation.nodes, observation.edges,
                    observation.serialized_bytes, (unsigned long long) observation.hash_low,
                    (unsigned long long) observation.hash_high, int(observation.seen_hash_before),
                    (unsigned long long) observation.elapsed_us,
                    (long long) std::chrono::duration_cast<std::chrono::microseconds>(prepare_end-prepare_begin).count(),
                    int(status), value.outcome,
                    (long long) std::chrono::duration_cast<std::chrono::microseconds>(prepare_begin.time_since_epoch()).count(),
                    (long long) std::chrono::duration_cast<std::chrono::microseconds>(prepare_end.time_since_epoch()).count(),
                    observation.reason.c_str());
        }
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
    status = cudaGraphLaunch(value.controller, stream);
    if (status != cudaSuccess) return status;
    status = cudaEventRecord(value.done, stream);
    if (status != cudaSuccess) return status;
    value.done_recorded = true;
    *outcome = AC922_TAIL_LAUNCHED;
    return cudaSuccess;
}

cudaError_t ready_captured_only(const void * owner, const void * key, cudaGraph_t captured,
        int device, int * is_ready) {
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
            value.controller != nullptr && value.done != nullptr;
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

extern "C" cudaError_t ac922_tail_bridge_drop_context(const void * owner) {
    try {
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
