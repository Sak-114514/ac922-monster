#pragma once

// Optional FlashAttention decode backend ported from the 1Cat-vLLM Flash-V100
// SM70 kernels (onecat-kernels/attention/sm70_grouped_long).
//
// Selected only when all of the following hold:
//   * NVIDIA Volta (compute capability 7.0),
//   * GGML_CUDA_FATTN_ONECAT=1 is set in the environment (default: off),
//   * head dim 256 for Q/K/V, single query token per stream (decode shape),
//   * GQA ratio divisible by 6, no ALiBi bias, no logit softcap, no sinks.
// Everything else (prefill, multi-token verify, other head dims) automatically
// stays on the stock fattn kernel family.
//
// Numerical contract: Q is rounded to FP16 (like the stock MMA kernel does
// in-kernel), scores are computed with Volta WMMA in FP32 and scaled there,
// softmax probabilities are kept in FP32 for the PV accumulation (FP16-KV
// path of the original kernel), split-K partial outputs are stored as FP16
// and merged in FP32. The stock fattn-mma-f16 path rounds P to FP16 instead,
// so element-wise agreement is expected only within that tolerance.

#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace ggml_onecat_v100 {

// Returns true when the env gate is enabled (GGML_CUDA_FATTN_ONECAT=1).
bool env_enabled();

// Shape/type admission check for the Onecat Flash-V100 decode path.
// Does not check the compute capability; the caller has already verified
// cc == GGML_CUDA_CC_VOLTA via the dispatch in fattn.cu.
bool supported(const ggml_tensor * dst);

// Total bytes needed behind dst->data (must match what launch() uses).
size_t get_alloc_size(const ggml_tensor * dst);

// Runs the Onecat partition + reduce kernels for an admitted tensor.
void launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

} // namespace ggml_onecat_v100

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
