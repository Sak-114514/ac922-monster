#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);

// RMS_NORM, MUL by a weight row, then a multi-section ROPE (possibly through a reshape) in one kernel;
// false (nothing launched) if the shapes or the rope mode are not supported
bool ggml_cuda_op_rms_norm_mul_rope_multi(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, const ggml_tensor * w,
        ggml_tensor * rope);
