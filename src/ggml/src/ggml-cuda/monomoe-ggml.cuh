#pragma once
#include "common.cuh"
bool ggml_cuda_monomoe_supported(int device, const ggml_tensor * t);
size_t ggml_cuda_monomoe_alloc_size(const ggml_tensor * t);
void ggml_cuda_op_monomoe(ggml_backend_cuda_context & ctx, ggml_tensor * t);
