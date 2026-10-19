#pragma once
#include <cuda_runtime_api.h>

// A successful call with outcome 1 enqueued the controller and its tail child.
// A successful call with a negative outcome enqueued no model computation;
// the caller must use its original host graph launch. Errors must propagate.
enum ac922_tail_bridge_outcome {
    AC922_TAIL_LAUNCHED = 1,
    AC922_TAIL_UNSUPPORTED_NODE = -1,
    AC922_TAIL_INSTANTIATION_DECLINED = -2,
    AC922_TAIL_CACHE_CAPACITY = -3,
    AC922_TAIL_STREAM_CHANGED = -4,
};

extern "C" cudaError_t ac922_tail_bridge_launch(
        const void * owner, const void * key, cudaGraph_t captured,
        cudaGraphExec_t generation, int invalidate, cudaStream_t stream,
        int device, int * outcome);

// Call after submission owners retire and before their streams/buffers die.
extern "C" cudaError_t ac922_tail_bridge_drop_context(const void * owner);

// Optional ABI v2: the main backend keeps the captured graph for identity but
// never creates/passes a standard cudaGraphExec_t. The helper owns child and
// controller executables as before.
extern "C" cudaError_t ac922_tail_bridge_v2_ready(
        const void * owner, const void * key, cudaGraph_t captured,
        int device, int * ready);
extern "C" cudaError_t ac922_tail_bridge_v2_launch_captured(
        const void * owner, const void * key, cudaGraph_t captured,
        int invalidate, cudaStream_t stream, int device, int * outcome);
