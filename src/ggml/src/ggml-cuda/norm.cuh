#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_scale(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * scale_node);

bool ggml_cuda_rms_norm_gate_match(const ggml_tensor * rms, const ggml_tensor * mul, const ggml_tensor * sig, const ggml_tensor * out);
void ggml_cuda_op_rms_norm_gate(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * mul, ggml_tensor * sig,
        ggml_tensor * out, block_q8_1 * yq, int64_t ne10);
