#include "common.cuh"
#include "mmq.cuh"
#include "mmq-moe-grouped.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "optimization-path-log.h"

#include <algorithm>
#include <climits>
#include <cstdlib>

bool ggml_cuda_moe_grouped_enabled() {
    static const int enabled = [] {
        const char * value = getenv("GGML_CUDA_MOE_GROUPED");
        return value ? atoi(value) : 0;
    }();
    return enabled != 0;
}

static bool ggml_cuda_mul_mat_q_moe_grouped_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static void ggml_cuda_mul_mat_q_moe_grouped_switch_type(
        ggml_backend_cuda_context & ctx, const mmq_args & args, const int32_t * tile_offsets,
        const int n_experts, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q2_0:
            mul_mat_q_moe_grouped_case<GGML_TYPE_Q2_0>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_moe_grouped_case<GGML_TYPE_Q4_0>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_moe_grouped_case<GGML_TYPE_Q6_K>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_moe_grouped_case<GGML_TYPE_IQ2_XXS>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_moe_grouped_case<GGML_TYPE_IQ2_XS>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_moe_grouped_case<GGML_TYPE_IQ2_S>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_moe_grouped_case<GGML_TYPE_IQ3_XXS>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_moe_grouped_case<GGML_TYPE_IQ3_S>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_moe_grouped_case<GGML_TYPE_IQ4_NL>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_moe_grouped_case<GGML_TYPE_IQ4_XS>(ctx, args, tile_offsets, n_experts, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

bool ggml_cuda_mul_mat_q_moe_grouped(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(        ids->type  == GGML_TYPE_I32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(        ids->nb[0] == ggml_type_size(ids->type));

    // Refuse unsupported cases up front (before any launch or pool allocation) so that the
    // caller can transparently fall back to the regular ids path.
    if (ne13 != 1) {
        ggml_cuda_optimization_path_log(ctx.device, "mmq_moe_grouped", true, false,
                "shape_ne13", ne12, ne01, ne02);
        return false;
    }
    if (blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4)) {
        ggml_cuda_optimization_path_log(ctx.device, "mmq_moe_grouped", true, false,
                "native_fp4_layout", ne12, ne01, ne02);
        return false; // Native FP4 MMQ layouts are not covered by this path.
    }
    if (!ggml_cuda_mul_mat_q_moe_grouped_supported(src0->type)) {
        ggml_cuda_optimization_path_log(ctx.device, "mmq_moe_grouped", true, false,
                "unsupported_quant_type", ne12, ne01, ne02);
        return false;
    }
    if (ne02 > MMQ_MOE_GROUPED_MAX_EXPERTS) {
        ggml_cuda_optimization_path_log(ctx.device, "mmq_moe_grouped", true, false,
                "expert_limit", ne12, ne01, ne02);
        return false; // The tile_offsets scan kernel is a single CUDA block.
    }
    {
        // There must be a usable (J, fallback) config (mirroring the probe in
        // mul_mat_q_moe_grouped_switch_J), and no candidate may need more grid.y I-tiles
        // than CUDA's 65535 limit.
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        const bool   fallback = ne01 % 128 != 0;
        bool usable = false;
        int  nty_max = 0;
        static constexpr int J_candidates[] = { 8, 16, 24, 32, 48, 64, 128 };
        for (const int J : J_candidates) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J, fallback, cc);
            if (config.type == GGML_TYPE_COUNT) {
                continue;
            }
            if (mmq_get_nbytes_shared(config, cc) > smpbo) {
                continue;
            }
            usable = true;
            nty_max = std::max(nty_max, (int) ((ne01 + config.I - 1) / config.I));
        }
        if (!usable) {
            ggml_cuda_optimization_path_log(ctx.device, "mmq_moe_grouped", true, false,
                    "no_shared_config", ne12, ne01, ne02);
            return false;
        }
        if (nty_max > 65535) {
            ggml_cuda_optimization_path_log(ctx.device, "mmq_moe_grouped", true, false,
                    "grid_y_limit", ne12, ne01, ne02);
            return false; // CUDA grid.y limit.
        }
    }

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding (as ggml_cuda_mul_mat_q does).
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const bool fallback = ne01 % 128 != 0;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
    ggml_cuda_pool_alloc<int32_t> tile_offsets(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once
    // and scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    // The GEMM tile width must be known before the tile_offsets scan kernel is launched.
    // This mirrors mul_mat_q_moe_grouped_switch_J, which re-derives the same J_best.
    const int J_grouped = [&] {
        const int    id    = ggml_cuda_get_device();
        const int    cc_   = ggml_cuda_info().devices[id].cc;
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

        int J_max = 128;
        static const int J_env_grouped = getenv("GGML_CUDA_MOE_GROUPED_J") ? atoi(getenv("GGML_CUDA_MOE_GROUPED_J")) : 0;
        static const int J_env_mmq     = getenv("GGML_CUDA_MMQ_MOE_J")     ? atoi(getenv("GGML_CUDA_MMQ_MOE_J"))     : 0;
        if (J_env_grouped > 0) {
            J_max = J_env_grouped;
        } else if (J_env_mmq > 0) {
            J_max = J_env_mmq;
        } else if (ggml_cuda_highest_compiled_arch(cc_) < GGML_CUDA_CC_TURING) {
            const int64_t avg = (ne_get_rows + ne02 - 1) / ne02;
            J_max = avg <= 16 ? 32 : avg <= 32 ? 64 : 128;
        }

        int J_best        = 0;
        int ntiles_J_best = INT_MAX;
        static constexpr int J_candidates[] = { 8, 16, 24, 32, 48, 64, 128 };
        for (const int J : J_candidates) {
            if (J > J_max) {
                break;
            }
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J, fallback, cc_);
            if (config.type == GGML_TYPE_COUNT) {
                continue;
            }
            if (mmq_get_nbytes_shared(config, cc_) > smpbo) {
                continue;
            }
            const int ntiles_x = ((int64_t) ne12 + config.J - 1) / config.J;
            if (ntiles_x < ntiles_J_best) {
                J_best = J;
                ntiles_J_best = ntiles_x;
            }
        }
        GGML_ASSERT(J_best != 0);
        return J_best;
    }();

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);

        // Live per-expert column-tile prefix sums for the compact grouped launch.
        launch_mmq_moe_tile_offsets(expert_bounds.get(), tile_offsets.get(), ne02, J_grouped, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    // Same quantized activation buffer as the regular path, including the J_max-sized tail
    // padding that covers the (discarded) over-read of an expert's last partial tile.
    const int J_max_alloc = ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11);
    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ +
        J_max_alloc * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12_flat = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13_flat = ne12*s12_flat;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the
    // expert index inside the kernel (via the tile_offsets binary search).
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        nullptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12_flat, s2,
        ne03, 1, s03, s13_flat, s3,
        ne12, ne12};

    ggml_cuda_mul_mat_q_moe_grouped_switch_type(ctx, args, tile_offsets.get(), ne02, stream);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_optimization_path_log(ctx.device, "mmq_moe_grouped", true, true,
            "compact_grouped_launch", ne12, ne01, ne02);
    return true;
}
