#pragma once

#include "common.cuh"

// GGML_OP_HC_FUSED: hyper-connection mix (down, up) and combine for few tokens, see ggml_hc_mix_down
#define HC_FUSED_MAX_TOKENS 8

void ggml_cuda_op_hc_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_hc_fused_supported(const ggml_tensor * op);

// hc combine (mode 2) directly followed by the next hc mix-down (mode 0) that reads it: the combine is done by the
// mix-down's first kernel. false (nothing launched) if the pair does not fit
bool ggml_cuda_op_hc_combine_mix_down(ggml_backend_cuda_context & ctx, ggml_tensor * combine, ggml_tensor * down);
