#pragma once
// =============================================================================
// kernel_ik_fused_moe.cuh -- port of ik_llama.cpp's fused MoE GEMV structure
// (k_fused_mul_mat_vec_q / fused_mul_mat_vec_q, ik mmvq-templates.cuh @5f89bfc,
// lines 152-330) for the MoE gate/up projections on V100 / sm_70.
// Wired in at mul_mat_vec_q_moe_launch() behind GGML_IK_FUSED_MOE (default 0).
//
// Structure vs the stock mul_mat_vec_q_moe (warp-per-token, block=(32,nc)):
//   ik : grid=(ceil(nrows/2), slots); block=(32,1). ONE warp computes 2 rows
//        for ALL tokens (unrolled j loop), __launch_bounds__(32,1) gives the
//        warp the full 255-register budget for the larger accumulator set
//        (tmp/tmp_gate[nc][2]) -- "tell the compiler to use as many registers
//        as it wants" (ik comment).
//   fork: block=(32,nc); each warp staffs one token and streams K alone.
// Both use the fork's vecdotq decoders verbatim; the kbx/lane partition
// (kbx = tid/(qi/vdr), step vdr*32/qi) and warp_reduce_sum are identical, so
// for identical inputs/ids this kernel's outputs are bit-identical to the
// stock kernel's (same adds in the same order per lane, same reduce tree,
// same GLU epilogue helpers).
//
// REQUIRED ADAPTATION vs ik (why this is not a verbatim copy): ik's host only
// enables the kernel at nc==1, where one expert per channel (slot) serves the
// whole batch -- the kernel reads ids once per channel. This fork's
// MUL_MAT_ID routing is per (slot, token): ids[slot + token*ids_stride]
// (traffic memo in moe-rework/INTEGRATION.md: ~50 distinct experts per nc5
// launch). This port therefore reads the expert id per token inside the j
// loop.
//
// VERDICT (2026-10-04, ik-eval/INTEGRATION.md, keep this switch OFF):
// with per-(slot,token) ids the x offset is j-variant, so nvcc cannot CSE
// the x decode across the j loop -- and that CSE is the ENTIRE phase-1 gain.
// Measured head-to-head vs the stock kernel in one harness (bench_ikfused_*):
//   correct-semantics port : 0.59x-0.91x (slower) at every shape/mode tested
//   uniform-only ik form    : 2.33x (IQ3_S N640 nc5, 60.7us vs 141.2us) --
//     reproducible, but semantically wrong for nc>1 under this fork's routing
// e2e wiring check (GGML_IK_FUSED_MOE=2, island preset): 512-token greedy
// parity matched:true; decode 88.76 vs 91.98/91.83 tok/s baseline = -3.6%.
// Retained in-tree (default 0) as a verified, reversible artifact; see
// INTEGRATION.md before enabling for anything.
//
// Coverage: only the R17 MoE gate/up pool is instantiated (IQ2_XXS/XS/S,
// IQ3_XXS/S). Q2_0 (down, GGML_SM70_MOE_REWORK) and IQ4_NL (down) are
// deliberately NOT instantiated -- type dispatch keeps the paths separate.
// NVFP4 scale fusion is not handled here (guarded out at the dispatch site).
// =============================================================================

namespace ik_fused_moe {

using vec_dot_q_t = float (*)(const void * __restrict__, const block_q8_1 * __restrict__, const int &, const int &);

template <ggml_type type>
static constexpr bool supported() {
    switch (type) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return true;
        default:
            return false;
    }
}

template <ggml_type type>
static constexpr __device__ vec_dot_q_t get_vec_dot() {
    switch (type) {
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

// VDR_*_Q8_1_MMVQ values for the supported types (identical in ik and fork,
// ik-eval/REPORT.md section 1.1).
template <ggml_type type>
static constexpr __device__ int get_vdr() {
    switch (type) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return 2;
        default:
            return 1;
    }
}

// One warp computes rows_per_block=2 output rows for ALL tokens (unrolled j
// loop) of one expert slot (blockIdx.y). Grid: (ceil(nrows_x/2), nchannels_dst).
template <ggml_type type, int ncols_dst, bool has_fusion>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void ik_fused_mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr<type>();
    constexpr vec_dot_q_t vec_dot_q_cuda = get_vec_dot<type>();

    constexpr int rows_per_block  = 2;                  // ik: one warp owns two rows
    constexpr int blocks_per_iter = vdr * WARP_SIZE / qi; // nwarps == 1

    const int blocks_per_row_x = ncols_x / qk;
    const int row0             = rows_per_block * blockIdx.x;

    const uint32_t channel_dst = blockIdx.y;

    ggml_cuda_pdl_sync();

    // fork MUL_MAT_ID semantics: one expert per (slot, token); all lanes read
    // the same addresses (broadcast), one per token.
    uint32_t expert[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        expert[j] = (uint32_t) ids[channel_dst + j*ids_stride];
    }
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;
    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
    }

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y;

    // partial sums: [token][row]; the accumulation order per element matches
    // the stock kernel exactly (same lane -> kbx mapping, same adds).
    float tmp[ncols_dst][rows_per_block]      = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_block] = {{0.0f}};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const uint32_t kbx_offset = expert[j]*stride_channel_x + row0*stride_row_x;
#pragma unroll
            for (int i = 0; i < rows_per_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // single warp: warp-level reduction only, no shared memory
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
            tmp[j][i] = warp_reduce_sum<WARP_SIZE>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<WARP_SIZE>(tmp_gate[j][i]);
                }
            }
        }
    }

    // write back: lane i holds row i (as in the stock kernel)
    if (threadIdx.x < rows_per_block && uint32_t(row0 + threadIdx.x) < nrows_x) {
        const int i = threadIdx.x;
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            float result = tmp[j][i];
            if constexpr (has_fusion) {
                const uint32_t bias_idx = expert[j]*stride_channel_dst + row0 + i;
                if (x_bias) {
                    result += x_bias[bias_idx];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[j][i];
                    if (gate_bias) {
                        gate_value += gate_bias[bias_idx];
                    }
                    switch (active_glu) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI:
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        case GGML_GLU_OP_SWIGLU_CLAMP:
                            result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                            break;
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[channel_dst*stride_channel_dst + j*stride_col_dst + row0 + i] = result;
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit);
    }
}

template <ggml_type type, int ncols_dst>
static bool launch_ncols(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ids_stride, const uint32_t nchannels_dst, cudaStream_t stream) {
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    const dim3 block_nums((nrows_x + 1) / 2, nchannels_dst, 1);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    if (has_fusion) {
        ggml_cuda_kernel_launch((ik_fused_mul_mat_vec_q_moe<type, ncols_dst, true>), launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch((ik_fused_mul_mat_vec_q_moe<type, ncols_dst, false>), launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst, ids_stride);
    }
    return true;
}

template <ggml_type type>
static bool launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const uint32_t nchannels_dst, cudaStream_t stream) {
    switch (ncols_dst) {
        case 2: return launch_ncols<type, 2 >(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, ids_stride, nchannels_dst, stream);
        case 3: return launch_ncols<type, 3 >(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, ids_stride, nchannels_dst, stream);
        case 4: return launch_ncols<type, 4 >(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, ids_stride, nchannels_dst, stream);
        case 5: return launch_ncols<type, 5 >(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, ids_stride, nchannels_dst, stream);
        case 6: return launch_ncols<type, 6 >(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, ids_stride, nchannels_dst, stream);
        case 7: return launch_ncols<type, 7 >(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, ids_stride, nchannels_dst, stream);
        case 8: return launch_ncols<type, 8 >(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, ids_stride, nchannels_dst, stream);
        default: return false;
    }
}

} // namespace ik_fused_moe
