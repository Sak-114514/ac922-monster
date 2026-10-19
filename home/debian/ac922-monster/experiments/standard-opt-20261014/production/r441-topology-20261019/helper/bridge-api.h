#pragma once
#include <cuda_runtime_api.h>

// ABI v1/v2 declarations are retained from the R398 host-child helper.
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
extern "C" cudaError_t ac922_tail_bridge_drop_context(const void * owner);
extern "C" cudaError_t ac922_tail_bridge_v2_ready(
        const void * owner, const void * key, cudaGraph_t captured,
        int device, int * ready);
extern "C" cudaError_t ac922_tail_bridge_v2_launch_captured(
        const void * owner, const void * key, cudaGraph_t captured,
        int invalidate, cudaStream_t stream, int device, int * outcome);

// Optional ABI v3. Preparation may inspect, instantiate with flags=0, upload,
// and create the completion event, but must not launch the captured graph or
// record that event. A zero result declines safely; the caller keeps its normal
// ordered graph execution. Errors, including a prior CUDA error, propagate.
extern "C" cudaError_t ac922_tail_bridge_v3_prepare_captured(
        const void * owner, const void * key, cudaGraph_t captured,
        cudaStream_t stream, int device, int * prepared);
extern "C" cudaError_t ac922_tail_bridge_v3_can_prepare(
        const void * owner, const void * key, int device, int * can_prepare);
