#pragma once
// =============================================================================
// kernel_sm70_moe.cuh -- restructured MoE quantized GEMV (Q2_0 down projection)
// for V100 / sm_70. Wired in at mul_mat_vec_q_moe_launch() behind
// GGML_SM70_MOE_REWORK (default 0 = stock path).
//
// Microbenchmark evidence (2026-10-04, ac922 V100 dev0, bench_moe_rework.cu,
// fork kernel extracted verbatim as baseline, correctness vs double-precision
// CPU reference transcribed from the fork's dequantize_row_q2_0):
//
//   Q2_0 down, 10 experts x N=2560 (per-card island shapes):
//     K=256 nc5: fork 38.54us -> 12.89us (3.00x); R17 in-situ fork 41.4us
//     K=192 nc5: fork 38.48us -> 11.73us (3.28x)
//     K=256 nc1: fork 20.74us ->  6.71us (3.09x)
//     K=256 nc5, per-token distinct experts (production MUL_MAT_ID routing):
//                fork 39.53us -> 20.74us (1.91x, 57% of same-size copy bound)
//   All 50 correctness checks (shapes x nc x ids-modes x configs) pass with
//   max rel err 2.8e-5 against the CPU reference (gate 2e-4); the fork
//   baseline passes the same gate on the same data.
//
// WHY THE FORK KERNEL IS SLOW HERE (qmmvq-audit/REPORT.md + phase timing):
// grid shredding (12800 blocks of 160 threads; the empty-kernel launch floor
// at grid 12800 is 22.2us vs 3.3us at <=1280) and, inside each warp, only
// 8/32 lanes have work for K-sliced shapes (K=256 -> 4 Q2_0 blocks per row,
// the K-loop trips once).
//
// STRUCTURE:
//   grid  = (ceil(nrows/128), slots, tokens);  block = (32, 4 warps) = 128 thr
//   One CUDA block owns one (slot, token) pair and 128 rows. Each warp
//   streams its 32-row span into shared memory and decodes lane->row.
//   Grid: 12800 -> 1000 blocks (nc5); schedule floor 22.2 -> ~3.5us.
//
//   Two-phase warp loop, decoupling the DRAM pattern from the decode pattern
//   (Q2_0 rows are 54/72B and can never be 16B-aligned):
//     A) STAGE: the warp's 32 rows are ONE contiguous global region; the warp
//        copies its aligned 16B window with fully-coalesced uint4 loads.
//        (Per-row ~5-lane bursts were measured 4x slower than this span form;
//        the window over-reads <=15B into neighbouring tensor data, and those
//        bytes are never decoded.)
//     B) DECODE: fork vec_dot_q2_0_q8_1 VERBATIM from the smem tile (bit-exact
//        math), with the token's q8_1 activations staged once in smem so each
//        weight byte's paired activation comes from shared memory instead of
//        L1-relayed global loads.
//
//   Per-token blocks (grid.z = token) are REQUIRED: MUL_MAT_ID semantics are
//   ids[slot + token*ids_stride] -- each token routes to its own experts, so
//   a warp covering all tokens of a slot would read the wrong weights. The
//   fork kernel reads per-(slot,token) experts the same way.
//
//   Accepted trade-offs (all measured, see bench):
//     - weight crumbs are re-decoded per token (one token per block). With
//       per-token distinct experts there is nothing to share; with repeated
//       experts L2 absorbs the re-reads (both modes measured above).
//     - packed 72B row stride in smem can cost up to 2-way bank conflicts
//       (conflict-free needs an odd word stride, which breaks uint4-aligned
//       staging stores).
//     - float accumulation order differs from the fork's warp reduce
//       (deterministic; token-level parity checked over 512 greedy tokens).
//
// Dependencies (must be included earlier in the TU): common.cuh (block types,
// ggml_cuda_kernel_launch), vecdotq.cuh (vec_dot_q2_0_q8_1, get_int_b2/b4).
// =============================================================================

namespace moe_sm70 {

constexpr int MOE_WARP_SZ = 32;
constexpr int MOE_Q8BLK   = (int)sizeof(block_q8_1);  // 36 = 2*half + 32 qs

// Q2_0 block: 64 elements, 18 bytes (half d + qs[16] at 2 bits/element).
// vec_dot_q2_0_q8_1(vbq, y, kbx, iqs) computes one 32-element chunk per call
// with iqs in {0,1}; the y q8_1 base for weight block kb is 2*kb.
constexpr int Q2_0_QK    = 64;
constexpr int Q2_0_BS    = 18;
constexpr int Q2_0_CALLS = 2;

// 32 rows per warp -> lane->row decode with no cross-lane reduction, and the
// whole span (1728/2304B) is a single staging window. 4 warps per block was
// the best measured config for both K=192 and K=256 (8 warps equal-or-worse).
constexpr int T_R = 32;
constexpr int NW  = 4;
constexpr int NT  = NW*MOE_WARP_SZ;          // 128 threads
constexpr int ROWS_PER_BLOCK = T_R*NW;       // 128 rows per block

// aligned-window cooperative copy: threads stream [align16_down(src),
// align16_up(src+len)) into dst with uint4s; returns the byte shift of the
// true segment start inside the window (0..15; even here, since every offset
// is a multiple of the 2-byte block stride).
static __device__ __forceinline__ int moe_stage_window(
    const char * src, const int len, char * dst, const int t, const int nthreads) {
    const uintptr_t a0 = (uintptr_t)src & ~(uintptr_t)15;
    const int shift = (int)((uintptr_t)src - a0);
    const uintptr_t a1 = ((uintptr_t)src + len + 15) & ~(uintptr_t)15;
    const int n_u4 = (int)((a1 - a0) >> 4);
    const uint4 * s4 = (const uint4 *)a0;
    uint4 * d4 = (uint4 *)dst;
    for (int i = t; i < n_u4; i += nthreads) d4[i] = s4[i];
    return shift;
}

template<int KB>  // Q2_0 k-blocks per row: 3 (K=192) or 4 (K=256)
__launch_bounds__(NT)
__global__ void moe_matvec_q2_0_sm70(
        const void * GGML_CUDA_RESTRICT vx_ptr, const void * GGML_CUDA_RESTRICT vy_ptr,
        const int32_t * GGML_CUDA_RESTRICT ids, float * GGML_CUDA_RESTRICT dst_ptr,
        const uint32_t ncols_x,             // K (192 or 256)
        const uint32_t nrows_x,             // N (2560 for the down projections)
        const uint32_t stride_row_x,        // blocks between weight rows (== K/64)
        const uint32_t stride_col_y,        // q8_1 blocks between token rows of y
        const uint32_t stride_col_dst,      // elements between token rows of dst
        const uint32_t stride_channel_x,    // blocks between experts
        const uint32_t stride_channel_y,    // q8_1 blocks between y channels
        const uint32_t stride_channel_dst,  // elements between dst slots
        const uint32_t ids_stride,          // ids elements between token rows
        const uint32_t nchannels_y)         // 10 for down (per-slot activations)
{
    constexpr int BS = Q2_0_BS;

    // smem layout: y row | NW warp tiles | per-row outputs
    extern __shared__ char smem_raw[];
    const int y_bytes = (int)(ncols_x/32) * MOE_Q8BLK;
    const int y_pitch = (y_bytes + 31) & ~15;
    char * y_s   = smem_raw;
    char * tiles = y_s + y_pitch;
    const int warp_tile_bytes = (T_R*KB*BS + 31) & ~15;  // span + window slack
    float * out_s = (float *)(tiles + (size_t)NW*warp_tile_bytes);

    const int tid  = threadIdx.y*MOE_WARP_SZ + threadIdx.x;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int ch   = blockIdx.y;   // expert slot
    const int tok  = blockIdx.z;   // token

    // per-(slot,token) expert -- the reason blocks are per-token
    const int chx = ids[ch + tok*ids_stride];
    const int chy = nchannels_y > 1 ? ch % (int)nchannels_y : 0;

    // ---- block-wide staging: this token's y row ----
    // (36B blocks: the row start is never 16B-aligned; the window shift is a
    // multiple of 4 so the decoder's 4B qs reads stay naturally aligned)
    const char * ysrc = (const char *) vy_ptr
        + ((size_t)chy*stride_channel_y + (size_t)tok*stride_col_y) * MOE_Q8BLK;
    const int y_shift = moe_stage_window(ysrc, y_bytes, y_s, tid, NT);
    const block_q8_1 * y0 = (const block_q8_1 *)(y_s + y_shift);

    const int r0 = blockIdx.x*ROWS_PER_BLOCK + warp*T_R;

    // ---- warp staging: the whole 32-row span as one aligned window ----
    // wbase is the byte offset of the span start in the expert tensor; its
    // low 4 bits are the window shift the decoder re-adds. Rows past nrows
    // are clipped; the <=15B window over-read lands in neighbouring tensor
    // data (or buffer slack) and is never decoded.
    const size_t wbase = ((size_t)chx*stride_channel_x + (size_t)r0*stride_row_x) * BS;
    char * tile = tiles + (size_t)warp*warp_tile_bytes;
    if (r0 < (int)nrows_x) {
        const int valid = min(T_R, (int)nrows_x - r0);
        moe_stage_window((const char *)vx_ptr + wbase, valid*stride_row_x*BS, tile, lane, MOE_WARP_SZ);
    }

    // y staging must complete before any warp decodes (y_s is block-shared)
    __syncthreads();

    // ---- decode: lane -> row, fork vec_dot_q2_0_q8_1 verbatim ----
    float acc = 0.f;
    if (r0 + lane < (int)nrows_x) {
        const char * wrow = tile + (int)(wbase & (size_t)15) + (size_t)lane*KB*BS;
        for (int kb = 0; kb < KB; ++kb) {
            const block_q8_1 * yb = y0 + kb*2;  // 2 q8_1 chunks per weight block
            #pragma unroll
            for (int c = 0; c < Q2_0_CALLS; ++c) {
                acc += vec_dot_q2_0_q8_1(wrow, yb, kb, c);
            }
        }
    }
    out_s[warp*T_R + lane] = acc;  // one warp uniquely owns each row

    __syncthreads();

    // ---- epilogue: dst[ch*stride_channel_dst + tok*stride_col_dst + row] ----
    for (int r = tid; r < ROWS_PER_BLOCK; r += NT) {
        const int row_g = blockIdx.x*ROWS_PER_BLOCK + r;
        if (row_g >= (int)nrows_x) break;
        dst_ptr[(size_t)ch*stride_channel_dst + (size_t)tok*stride_col_dst + row_g] = out_s[r];
    }
}

// host-side launchability check. K whitelist {192, 256}: exactly the Q2_0
// down-projection slices this kernel was validated on (224 is not a legal
// Q2_0 K -- 224/64 is not integral; those slices are IQ4_NL in this model).
static inline bool moe_sm70_q2_0_valid(const uint32_t ncols_x, const uint32_t nrows_x,
                                       const uint32_t stride_row_x, const uint32_t ncols_dst) {
    if (ncols_x != 192 && ncols_x != 256) return false;
    if (stride_row_x != ncols_x/Q2_0_QK) return false;  // rows contiguous
    if (ncols_dst < 2 || ncols_dst > 8) return false;   // decode / MTP-verify window
    if (nrows_x == 0) return false;
    return true;
}

// Launch; returns false (without launching) when a guard fails so the caller
// can fall through to the stock path.
static inline bool moe_sm70_q2_0_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t nchan_dst,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const uint32_t nchannels_y,
        cudaStream_t stream) {
    if (!moe_sm70_q2_0_valid(ncols_x, nrows_x, stride_row_x, ncols_dst)) return false;

    const int kb = (int)(ncols_x/Q2_0_QK);
    const int y_pitch = (((int)(ncols_x/32)*MOE_Q8BLK) + 31) & ~15;
    const size_t smem = y_pitch + (size_t)NW*((T_R*kb*Q2_0_BS + 31) & ~15)
                      + (size_t)ROWS_PER_BLOCK*4;
    if (smem >= 48*1024) return false;  // ~10KB here; under the static limit

    const dim3 grid((nrows_x + ROWS_PER_BLOCK - 1)/ROWS_PER_BLOCK, nchan_dst, ncols_dst);
    const dim3 block(MOE_WARP_SZ, NW);
    const ggml_cuda_kernel_launch_params lp(grid, block, smem, stream);

    if (ncols_x == 256) {
        ggml_cuda_kernel_launch((moe_matvec_q2_0_sm70<4>), lp,
            vx, vy, ids, dst, ncols_x, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ids_stride, nchannels_y);
    } else {
        ggml_cuda_kernel_launch((moe_matvec_q2_0_sm70<3>), lp,
            vx, vy, ids, dst, ncols_x, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ids_stride, nchannels_y);
    }
    return true;
}

} // namespace moe_sm70
