#pragma once

#include "common.cuh"

// Fused qwen4exp PLE depthwise conv (1Cat qwen38_ple_spec_sm70 lineage, adapted).
//
// Replaces the generic ggml chain inside llama_model_qwen4exp::graph::build_ple:
//   concat(conv_state, x) -> 4x (cont(transpose(view)) * w) -> add chain -> silu
//     -> per-rollback-slot tail copies
// with one kernel that also emits the n_slots rolled conv-state windows in the
// dst row tail (the caller copies them into cache_ple_r_l exactly as before).
//
// Tensor contract (all contiguous, row-major ggml layout):
//   src[0] x      F32 [C, T, S]     conv input ("normalized"); C = hc_dim (10240), token-major rows
//   src[1] state  F32 [H, C, S]     gathered conv history; H = (kern-1)*dil (9); time contiguous
//   src[2] w      F16 [kern, C]     ple_conv1d, k fastest (same bytes as 1Cat weight [C,4])
//   dst          F32 [C, T + n_slots*H, S]
//       rows [0, T)          silu(conv) output
//       rows [T, T+n_slots*H) slot s state = window ending s tokens before the end
//   op_params: i32(0) = dilation (ple_ngram_size), i32(1) = n_slots (n_rs_seq + 1)
//
// Numerics (default): tap order k = 0..kern-1 with __fmul_rn/__fadd_rn and
// silu = v / (1.0f + expf(-v)) — bit-identical to the replaced CUDA mul/add/silu
// chain. LLAMA_PLE_ONECAT_F16=1 selects the 1Cat checkpoint-FP16 rounding
// (half-rounded operands and accumulator, fmaf) for A/B parity experiments.

void ggml_cuda_op_ple_conv_sm70(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
