#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// ssm_conv (+ bias + silu) and the RMS_NORM + SCALE of up to two head windows of its output (n pairs) in one kernel;
// false (nothing launched) if the layout does not fit
bool ggml_cuda_op_ssm_conv_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node,
                                ggml_tensor * silu_dst, ggml_tensor * const * rms, ggml_tensor * const * scale, int n);
