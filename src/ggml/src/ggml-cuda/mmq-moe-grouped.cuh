#pragma once

// MoE "grouped" MMQ: a compact single launch for GGML_OP_MUL_MAT_ID.
//
// Recon summary (see patches/moe-grouped-NOTES.md for details):
//   The regular mul_mat_q kernel already handles all experts of one MUL_MAT_ID in a single
//   launch by putting the expert index on blockIdx.z (nchannels_y == ne02) and by early-exiting
//   column tiles beyond an expert's token count (expert_bounds[zt]). But the launch grid is
//   sized from ncols_max == ne12 (the whole ubatch), so with 512 experts and ~10 tokens per
//   expert at J == 32 only ~1 of the 16 column tiles per expert is live: >90% of the launched
//   CUDA blocks exit immediately, which matches the observed 2%-of-peak prefill utilization.
//
//   This variant launches only the *live* tiles: a tiny scan kernel derives, from the existing
//   expert_bounds output of mm_ids_helper, the per-expert column-tile prefix sums
//   (tile_offsets[e] = sum_{e'<e} ceil(tokens_e' / J)). The GEMM kernel maps blockIdx.x to
//   (expert, column tile) by binary search over tile_offsets and then runs the exact same
//   mul_mat_q_process_tile body (same k loop, same write-back through ids_dst) as the regular
//   kernel's non-stream-k MoE path, so per-element arithmetic is identical.
//
// Gated by GGML_CUDA_MOE_GROUPED=1 (default off). GGML_CUDA_MOE_GROUPED_J overrides the tile
// width; GGML_CUDA_MMQ_MOE_J is honored the same way as in the regular path.

#include "mmq.cuh"

// Host-side upper bound of live column tiles: ceil(c_e/J) summed over experts is at most
// ncols_y/J + n_experts (each expert adds less than one extra tile). Blocks above the exact
// device-side count exit after a single read of tile_offsets[n_experts].
static inline int64_t mmq_moe_grouped_n_tiles_ub(const int64_t ncols_y, const int n_experts, const int J) {
    return ncols_y / J + n_experts;
}

// Exclusive prefix sum of per-expert MMQ column-tile counts:
//   tile_offsets[e]   = sum_{e' < e} ceil((expert_bounds[e'+1] - expert_bounds[e']) / J)
//   tile_offsets[ne2] = total number of live column tiles
// expert_bounds is produced by ggml_cuda_launch_mm_ids_helper earlier on the same stream.
// Single block, Hillis-Steele scan over shared memory; supports up to 1024 experts
// (MMQ_MOE_GROUPED_MAX_EXPERTS, bounded by the maximum blockDim.x).
#define MMQ_MOE_GROUPED_MAX_EXPERTS 1024

static __global__ void mmq_moe_tile_offsets(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_offsets,
        const int n_experts, const int J) {

    extern __shared__ int32_t scan_mm[]; // n_experts elements

    const int i = threadIdx.x;

    int32_t v = 0;
    if (i < n_experts) {
        const int32_t cnt = expert_bounds[i + 1] - expert_bounds[i];
        v = cnt > 0 ? (cnt + J - 1) / J : 0;
        scan_mm[i] = v;
    }
    __syncthreads();

    for (int off = 1; off < n_experts; off <<= 1) {
        int32_t add = 0;
        if (i >= off && i < n_experts) {
            add = scan_mm[i - off];
        }
        __syncthreads();
        v += add;
        if (i < n_experts) {
            scan_mm[i] = v;
        }
        __syncthreads();
    }

    if (i < n_experts) {
        tile_offsets[i] = i > 0 ? scan_mm[i - 1] : 0;
    }
    if (i == 0) {
        tile_offsets[n_experts] = scan_mm[n_experts - 1];
    }
}

static inline void launch_mmq_moe_tile_offsets(
        const int32_t * expert_bounds, int32_t * tile_offsets, const int n_experts, const int J, cudaStream_t stream) {

    GGML_ASSERT(n_experts > 0);
    GGML_ASSERT(n_experts <= MMQ_MOE_GROUPED_MAX_EXPERTS);

    // The scan assigns element i to thread i, so blockDim.x must cover n_experts.
    const int block_size = n_experts <= 32 ? 32 : n_experts <= 128 ? 128 : n_experts <= 512 ? 512 : 1024;
    const size_t nbytes_shared = n_experts * sizeof(int32_t);
    mmq_moe_tile_offsets<<<1, block_size, nbytes_shared, stream>>>(expert_bounds, tile_offsets, n_experts, J);
}

// ---------------------------------------------------------------------------------------------

// One CUDA block computes one *live* tile of J sorted token slots of one expert and I output
// rows, i.e. exactly the work the regular kernel does for the tiles that survive its
// early-exit check. Grid: x = live column tiles (upper bound, see above), y = I-tiles.
template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback), ggml_cuda_mmq_get_occupancy(type, J, fallback))
static __global__ void mul_mat_q_moe_grouped(
        const char * __restrict__ x, const int * __restrict__ y,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_offsets, float * __restrict__ dst,
        const int n_experts, const int n_k_blocks,
        const int nrows_x, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int stride_channel_x) {

    // Skip unused template specializations for faster compilation:
    if (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    const int t  = blockIdx.x; // Flat live column-tile index over all experts.
    const int it = blockIdx.y; // I-tile index within one expert's weight matrix.

    if (t >= tile_offsets[n_experts]) {
        return; // Padding block of the host-side upper bound.
    }

    // Find the expert owning tile t: the last e with tile_offsets[e] <= t. Experts with zero
    // tokens span an empty [tile_offsets[e], tile_offsets[e]) range and are skipped naturally.
    int zt_lo = 0;
    int zt_hi = n_experts - 1;
    while (zt_lo < zt_hi) {
        const int zt_mid = zt_lo + (zt_hi - zt_lo + 1) / 2;
        if (tile_offsets[zt_mid] <= t) {
            zt_lo = zt_mid;
        } else {
            zt_hi = zt_mid - 1;
        }
    }
    const int zt = zt_lo;
    const int jt = t - tile_offsets[zt]; // In [0, ceil(col_diff/J) - 1] by construction.

    const int col_low  = expert_bounds[zt + 0];
    const int col_high = expert_bounds[zt + 1];
    const int col_diff = col_high - col_low;

    // Initialize the ids for writing back data with the destination indices of this expert's
    // sorted token slots (same shared-memory slot layout as mul_mat_q).
    extern __shared__ int data_mul_mat_q[];
    int * ids_dst_shared = data_mul_mat_q;

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

        if (j0 + nwarps*warp_size > J && j >= J) {
            break;
        }

        const int jcol = jt*J + j;
        // jcol >= col_diff only for the padding lanes of an expert's last tile; their results
        // are dropped by the write-back's j_max check (ids value is never used).
        ids_dst_shared[j] = jcol < col_diff ? ids_dst[col_low + jcol] : j;
    }
    __syncthreads();

    const int offset_x = zt*stride_channel_x + it*I*stride_row_x;

    const int tile_x_max_i = nrows_x  - it*I - 1;
    const int tile_y_max_j = col_diff - jt*J - 1;

    constexpr bool fixup = false;
    mul_mat_q_process_tile<type, J, fallback, fixup>
        (x, offset_x, y + (col_low + jt*J)*(int)(sizeof(block_q8_1_mmq)/sizeof(int)), ids_dst_shared,
         dst + it*I, nullptr, nullptr,
         stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, 0, n_k_blocks);
}

template <ggml_type type, int J, bool fallback>
static void launch_mul_mat_q_moe_grouped(
        ggml_backend_cuda_context & ctx, const mmq_args & args, const int32_t * tile_offsets,
        const int n_experts, cudaStream_t stream) {

    const int id        = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[id].cc;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    GGML_UNUSED(ctx); // No scratch buffers: every block writes its tile directly.

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
    GGML_ASSERT(config.nthreads % warp_size == 0);
    const int nbytes_shared = mmq_get_nbytes_shared(config, cc);

    const dim3 block_dims(warp_size, config.nthreads / warp_size, 1);

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_moe_grouped<type, J, false>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_moe_grouped<type, J,  true>), nbytes_shared);

    const int nty = (args.nrows_x + config.I - 1) / config.I;
    GGML_ASSERT(nty <= 65535); // CUDA grid.y limit, checked by the caller before the launch.

    const int n_tiles_ub = (int) mmq_moe_grouped_n_tiles_ub(args.ncols_y, n_experts, J);
    const dim3 block_nums(n_tiles_ub, nty, 1);

    const int n_k_blocks = (int) (args.ncols_x / ggml_cuda_type_traits<type>::qk);

    mul_mat_q_moe_grouped<type, J, fallback><<<block_nums, block_dims, nbytes_shared, stream>>>
        (args.x, args.y, args.ids_dst, args.expert_bounds, tile_offsets, args.dst,
         n_experts, n_k_blocks,
         (int) args.nrows_x, (int) args.stride_row_x, (int) args.ncols_y, (int) args.nrows_dst,
         (int) args.stride_channel_x);
}

template <ggml_type type, bool fallback>
static void mul_mat_q_moe_grouped_switch_J(
        ggml_backend_cuda_context & ctx, const mmq_args & args, const int32_t * tile_offsets,
        const int n_experts, cudaStream_t stream) {
    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    // Tile width selection mirrors mul_mat_q_switch_J (mmq.cuh): on pre-Turing (V100 dp4a)
    // cap J by the average tokens per expert, GGML_CUDA_MOE_GROUPED_J overrides, and the
    // legacy GGML_CUDA_MMQ_MOE_J is honored too. The minimum-tile-count probe then uses the
    // same ncols_opt as the regular path, so with no env vars set the same J is picked as by
    // the regular kernel on V100 (J == 32 for ~10 tokens/expert), keeping A/B parity exact.
    int J_max = 128;
    {
        static const int J_env_grouped = getenv("GGML_CUDA_MOE_GROUPED_J") ? atoi(getenv("GGML_CUDA_MOE_GROUPED_J")) : 0;
        static const int J_env_mmq     = getenv("GGML_CUDA_MMQ_MOE_J")     ? atoi(getenv("GGML_CUDA_MMQ_MOE_J"))     : 0;
        if (J_env_grouped > 0) {
            J_max = J_env_grouped;
        } else if (J_env_mmq > 0) {
            J_max = J_env_mmq;
        } else if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_TURING) {
            const int64_t avg = (args.ncols_y + n_experts - 1) / n_experts;
            J_max = avg <= 16 ? 32 : avg <= 32 ? 64 : 128;
        }
    }

    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    static constexpr int J_candidates[] = { 8, 16, 24, 32, 48, 64, 128 }; // must match the switch below
    for (const int J : J_candidates) {
        if (J > J_max) {
            break;
        }

        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }

        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }

        const int ntiles_x = (args.ncols_opt + config.J - 1) / config.J;

        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }

    switch (J_best) {
        case   8:
            launch_mul_mat_q_moe_grouped<type,   8, fallback>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case  16:
            launch_mul_mat_q_moe_grouped<type,  16, fallback>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case  24:
            launch_mul_mat_q_moe_grouped<type,  24, fallback>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case  32:
            launch_mul_mat_q_moe_grouped<type,  32, fallback>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case  48:
            launch_mul_mat_q_moe_grouped<type,  48, fallback>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case  64:
            launch_mul_mat_q_moe_grouped<type,  64, fallback>(ctx, args, tile_offsets, n_experts, stream);
            break;
        case 128:
            launch_mul_mat_q_moe_grouped<type, 128, fallback>(ctx, args, tile_offsets, n_experts, stream);
            break;
        default:
            fprintf(stderr, "J_best=%d\n", J_best);
            GGML_ABORT("fatal error");
            break;
    }
}

// Not static: explicitly instantiated in template-instances/mmq-moe-grouped-instance-*.cu
// (same ODR scheme as mul_mat_q_case / DECL_MMQ_CASE in mmq.cuh).
template <ggml_type type>
void mul_mat_q_moe_grouped_case(
        ggml_backend_cuda_context & ctx, const mmq_args & args, const int32_t * tile_offsets,
        const int n_experts, cudaStream_t stream) {
    if (args.nrows_x % 128 == 0) {
        constexpr bool fallback = false;
        mul_mat_q_moe_grouped_switch_J<type, fallback>(ctx, args, tile_offsets, n_experts, stream);
    } else {
        constexpr bool fallback = true;
        mul_mat_q_moe_grouped_switch_J<type, fallback>(ctx, args, tile_offsets, n_experts, stream);
    }
}

#define DECL_MMQ_MOE_GROUPED_CASE(type)                                                                \
    template void mul_mat_q_moe_grouped_case<type>(                                                   \
            ggml_backend_cuda_context & ctx, const mmq_args & args, const int32_t * tile_offsets,     \
            const int n_experts, cudaStream_t stream)

extern DECL_MMQ_MOE_GROUPED_CASE(GGML_TYPE_Q2_0);
extern DECL_MMQ_MOE_GROUPED_CASE(GGML_TYPE_Q4_0);
extern DECL_MMQ_MOE_GROUPED_CASE(GGML_TYPE_Q6_K);
extern DECL_MMQ_MOE_GROUPED_CASE(GGML_TYPE_IQ2_XXS);
extern DECL_MMQ_MOE_GROUPED_CASE(GGML_TYPE_IQ2_XS);
extern DECL_MMQ_MOE_GROUPED_CASE(GGML_TYPE_IQ2_S);
extern DECL_MMQ_MOE_GROUPED_CASE(GGML_TYPE_IQ3_XXS);
extern DECL_MMQ_MOE_GROUPED_CASE(GGML_TYPE_IQ3_S);
extern DECL_MMQ_MOE_GROUPED_CASE(GGML_TYPE_IQ4_NL);
extern DECL_MMQ_MOE_GROUPED_CASE(GGML_TYPE_IQ4_XS);

// ---------------------------------------------------------------------------------------------

// Environment gate for the whole grouped MoE MMQ path (default off).
bool ggml_cuda_moe_grouped_enabled();

// Returns true when the op was handled. When false is returned nothing has been launched and
// the caller must fall back to the regular ggml_cuda_mul_mat_q ids path.
bool ggml_cuda_mul_mat_q_moe_grouped(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst);
