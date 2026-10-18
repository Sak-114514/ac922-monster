#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define GGML_CUDA_CONCAT_MAX_COPIES 8
bool ggml_cuda_concat_copies_match(const ggml_tensor * concat, const ggml_tensor * const * copies, int n);
void ggml_cuda_op_concat_copies(ggml_backend_cuda_context & ctx, ggml_tensor * concat, ggml_tensor * const * copies, int n);
