#pragma once

#include "common.cuh"

// In-graph AllReduce for 2-8 GPUs with peer access between every pair (NVLink). The meta backend puts one
// GGML_OP_ALLREDUCE node per partial tensor into each GPU's graph, so a whole pass runs as one CUDA graph per GPU.
// Each call pushes the local partial into every peer's staging buffer, then waits on one flag per peer in device memory.
// Call counters live in device memory, so replays of a captured graph stay in step on all GPUs.

struct ggml_cuda_p2p_ar;

// nullptr if the devices can not use peer access
ggml_cuda_p2p_ar * ggml_cuda_p2p_ar_init(const int * devices, size_t n_devices);
void               ggml_cuda_p2p_ar_free(ggml_cuda_p2p_ar * ar);

// make the staging buffers hold nbytes per call; synchronizes the devices if they must grow
bool ggml_cuda_p2p_ar_reserve(ggml_cuda_p2p_ar * ar, size_t nbytes);

// fill the op params of an ALLREDUCE node that runs on device index rank
void ggml_cuda_p2p_ar_set_params(ggml_cuda_p2p_ar * ar, size_t rank, ggml_tensor * node);

void ggml_cuda_op_allreduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
