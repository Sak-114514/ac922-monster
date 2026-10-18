#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);

// dst = sigmoid(g[token]) * sh + (weighted expert sum, as ggml_cuda_op_moe_weighted_reduction); g one value per token
void ggml_cuda_op_moe_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * experts, const ggml_tensor * expert_scale,
                           const ggml_tensor * weights, const ggml_tensor * g, const ggml_tensor * sh, ggml_tensor * dst);
