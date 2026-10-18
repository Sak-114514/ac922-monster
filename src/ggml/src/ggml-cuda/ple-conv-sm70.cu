// Fused qwen4exp PLE dilated causal depthwise conv + SiLU + rolled state windows.
//
// Adapted from the 1Cat-vLLM SM70 kernel `qwen38_ple_spec_sm70.cu` (vLLM
// contributors, Apache-2.0) for the llama-v100-v3 fork:
//   - torch/ATen wrapper  -> a ggml CUDA op (one dst, three ggml srcs)
//   - fixed H10240/M5/M10 single-request contract -> arbitrary C, T, S
//   - the 13-column state + read-time rollback trick -> the fork's own
//     [H, C] per-slot cache rows (rollback is resolved by the recurrent cache
//     before the graph runs, so the kernel needs no `accepted` counts)
//   - FP16 activation tensors -> F32 in/out (fork activations); the F16 weight
//     is kept (ple_conv1d is F16 in the checkpoint); an optional template mode
//     reproduces the 1Cat half-rounded arithmetic for parity measurements
//
// Launch shape: one thread per channel, blocks of 128 channels; blockIdx.y
// tiles the output rows (conv rows first, then the n_slots state windows),
// blockIdx.z is the sequence. All launch parameters derive from tensor shapes
// and op_params only -> CUDA-graph safe.
//
// Row semantics (dst is [C, T + n_slots*H, S]):
//   rows [0, T)            silu(sum_k w[k] * padded[r + k*dil]),
//                         padded[i] = i < H ? state[i] : x[i - H]
//   rows [T, T+n_slots*H)  slot s window: padded[max(0, T - s) + j], j in [0, H)
//                         (the same tail build_conv_state_at writes per slot)

#include "ple-conv-sm70.cuh"

#include <cuda_fp16.h>

namespace {

constexpr int PLE_THREADS      = 128;
constexpr int PLE_ROWS_PER_BLOCK = 8;

template <int KERN, int DIL, bool REG_STATE, bool HALF_ROUND>
__global__ void ple_conv_sm70_kernel(
        const float * __restrict__ x,       // [C, T, S]
        const float * __restrict__ state,   // [H, C, S]
        const half  * __restrict__ w,       // [KERN, C]
        float       * __restrict__ dst,     // [C, T + n_slots*H, S]
        int C, int T, int n_slots) {
    constexpr int HIST = (KERN - 1) * DIL;

    ggml_cuda_pdl_lc();

    const int c = blockIdx.x * PLE_THREADS + threadIdx.x;
    if (c >= C) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int s  = blockIdx.z;
    const int r0 = blockIdx.y * PLE_ROWS_PER_BLOCK;
    const int R  = T + n_slots * HIST;

    // per-(seq, channel) bases
    const int64_t x_base     = (int64_t) s * T * C + c;      // x[t] at x_base + t*C
    const int64_t state_base = ((int64_t) s * C + c) * HIST; // state[j] contiguous in j
    const int64_t dst_base   = (int64_t) s * R * C + c;      // dst[r] at dst_base + r*C

    float wf[KERN];
#pragma unroll
    for (int k = 0; k < KERN; ++k) {
        wf[k] = __half2float(w[int64_t(c) * KERN + k]);
    }

    float hist[REG_STATE ? HIST : 1];
    if constexpr (REG_STATE) {
#pragma unroll
        for (int j = 0; j < HIST; ++j) {
            hist[j] = state[state_base + j];
        }
    }

    // padded[i] = i < HIST ? history[i] : x[i - HIST]
    auto win = [&](int i) -> float {
        if (i < HIST) {
            if constexpr (REG_STATE) {
                return hist[i];
            } else {
                return state[state_base + i];
            }
        }
        return x[x_base + int64_t(i - HIST) * C];
    };

#pragma unroll 1
    for (int r = r0; r < r0 + PLE_ROWS_PER_BLOCK; ++r) {
        if (r >= R) {
            break;
        }

        if (r < T) {
            float v = 0.0f;
#pragma unroll
            for (int k = 0; k < KERN; ++k) {
                const float a = win(r + k * DIL);
                if constexpr (HALF_ROUND) {
                    // 1Cat path: operands are half in the source kernel
                    const float a_h = __half2float(__float2half_rn(a));
                    v = __fmaf_rn(a_h, wf[k], v);
                } else {
                    // exact path: IEEE mul then add, k ascending — the same
                    // rounding sequence as the generic CUDA mul/add chain
                    v = __fadd_rn(__fmul_rn(a, wf[k]), v);
                }
            }
            if constexpr (HALF_ROUND) {
                v = __half2float(__float2half_rn(v));
            }
            // matches ggml_cuda_op_silu_single
            dst[dst_base + int64_t(r) * C] = v / (1.0f + expf(-v));
        } else {
            const int rr    = r - T;
            const int slot  = rr / HIST;
            const int j     = rr % HIST;
            const int start = T > slot ? T - slot : 0;
            dst[dst_base + int64_t(r) * C] = win(start + j);
        }
    }
}

} // namespace

void ggml_cuda_op_ple_conv_sm70(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * state = dst->src[1];
    const ggml_tensor * w     = dst->src[2];

    GGML_ASSERT(x->type     == GGML_TYPE_F32 && ggml_is_contiguous(x));
    GGML_ASSERT(state->type == GGML_TYPE_F32 && ggml_is_contiguous(state));
    GGML_ASSERT(w->type     == GGML_TYPE_F16 && ggml_is_contiguous(w));
    GGML_ASSERT(dst->type   == GGML_TYPE_F32 && ggml_is_contiguous(dst));

    const int dil     = ggml_get_op_params_i32(dst, 0);
    const int n_slots = ggml_get_op_params_i32(dst, 1);

    const int C = (int) x->ne[0];
    const int T = (int) x->ne[1];
    const int S = (int) x->ne[2];
    const int H = (int) state->ne[0];
    const int K = (int) w->ne[0];

    GGML_ASSERT((int) w->ne[1]     == C && "ple conv weight channel mismatch");
    GGML_ASSERT((int) state->ne[1] == C && "ple conv state channel mismatch");
    GGML_ASSERT((int) dst->ne[0]   == C && (int) dst->ne[2] == S);
    GGML_ASSERT((int) dst->ne[1]   == T + n_slots * H && "ple conv dst row count mismatch");
    GGML_ASSERT(H == (K - 1) * dil && "ple conv history must be (kern-1)*dil");
    GGML_ASSERT(S <= 65535 && "ple conv sequence count exceeds gridDim.z");
    GGML_ASSERT(n_slots >= 1);

    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO("%s: using opt-in Onecat PLE SM70 convolution (C=%d T=%d S=%d K=%d dil=%d slots=%d)\n",
                __func__, C, T, S, K, dil, n_slots);
        logged = true;
    }
    const int R  = T + n_slots * H;
    const dim3 block(PLE_THREADS, 1, 1);
    const dim3 grid((C + PLE_THREADS - 1) / PLE_THREADS,
                    (R + PLE_ROWS_PER_BLOCK - 1) / PLE_ROWS_PER_BLOCK,
                    S);

    const float * x_d     = (const float *) x->data;
    const float * state_d = (const float *) state->data;
    const half  * w_d     = (const half  *) w->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    // LLAMA_PLE_ONECAT_F16=1: 1Cat checkpoint-FP16 rounding (parity A/B only)
    static const bool half_round = [] {
        const char * e = getenv("LLAMA_PLE_ONECAT_F16");
        return e != nullptr && atoi(e) != 0;
    }();

    if (K == 4 && dil == 3) {
        // qwen3.8-flash-next: ple_conv_kernel=4, ngram=3, HIST=9 fits registers
        if (half_round) {
            ple_conv_sm70_kernel<4, 3, true, true><<<grid, block, 0, stream>>>(x_d, state_d, w_d, dst_d, C, T, n_slots);
        } else {
            ple_conv_sm70_kernel<4, 3, true, false><<<grid, block, 0, stream>>>(x_d, state_d, w_d, dst_d, C, T, n_slots);
        }
    } else if (K == 4 && dil == 1) {
        if (half_round) {
            ple_conv_sm70_kernel<4, 1, false, true><<<grid, block, 0, stream>>>(x_d, state_d, w_d, dst_d, C, T, n_slots);
        } else {
            ple_conv_sm70_kernel<4, 1, false, false><<<grid, block, 0, stream>>>(x_d, state_d, w_d, dst_d, C, T, n_slots);
        }
    } else if (K == 3 && dil == 3) {
        if (half_round) {
            ple_conv_sm70_kernel<3, 3, false, true><<<grid, block, 0, stream>>>(x_d, state_d, w_d, dst_d, C, T, n_slots);
        } else {
            ple_conv_sm70_kernel<3, 3, false, false><<<grid, block, 0, stream>>>(x_d, state_d, w_d, dst_d, C, T, n_slots);
        }
    } else {
        GGML_ABORT("%s: unsupported ple conv geometry kern=%d dil=%d (kernel covers the"
                   " qwen3.8 ple_conv_kernel=4/ngram=3 checkpoint; add template"
                   " instantiations in %s to support more)", __func__, K, dil, __FILE__);
    }
}
