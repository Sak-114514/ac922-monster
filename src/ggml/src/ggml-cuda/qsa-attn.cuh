#pragma once

#include "common.cuh"

// Sparse QSA attention: src[5] carries block ids, or physical cell ids when op_params[7]
// has the explicit-cell tag (see qsa-attn.cu).
bool ggml_cuda_qsa_attn_is_sparse(const ggml_tensor * dst);
bool ggml_cuda_qsa_attn_supported(const ggml_tensor * dst);
void ggml_cuda_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
