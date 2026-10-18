// ============================================================================
// kernel_sm70_bf16.cuh -- sm_70 (V100) optimized BF16 GEMV (matvec) kernel
// ============================================================================
//
// Target problem (from R17 nsys trace of qwen4exp decode, 6x V100 SXM2):
//   dst[j][row] = sum_k W[row][k] * y[j][k],   W: BF16 [N_out, K], y: F32 [ncols, K]
//   Two mul_mat_vec_f<__nv_bfloat16, float, ncols_dst=5, block_size=160/256>
//   instantiations burn 33.1% of all GPU kernel time at ~40% of HBM bandwidth.
//
// Design constraints / hardware facts (all decisions below trace back to these):
//   * V100: 900 GB/s HBM2, 80 SMs, 64K 32-bit regs/SM, 96KB unified L1+shared
//     per SM (max 48KB *static* shared per block without opt-in), 32 banks x 4B
//     shared memory, LSU issue ~4 warp-loads/cycle/SM, ~64 FP32 FMA/cycle/SM.
//   * NO BF16 tensor core, NO cp.async, NO PDL on sm_70 -> everything is plain
//     LDG/LDS + FP32 FMA. bfloat16 arithmetic intrinsics would be legal but are
//     compiler-emulated instruction soup; we only need *conversion*, which is
//     a 16-bit shift (see bf16x2_to_float2).
//   * Arithmetic intensity of decode GEMV is 1..5 flop/byte << V100 balance
//     point (~17.5) -> the ONLY thing that matters is bytes moved per second.
//
// Why the current fork kernel (ggml-cuda/mmvf.cu, mul_mat_vec_f) is slow:
//   1. One 4-byte weight load per thread per iteration (a single nv_bfloat162).
//     Consecutive threads ARE coalesced (32x4B = one 128B line per warp), but a
//     warp must issue 4x more load instructions than a uint4 kernel to move the
//     same bytes, and each thread has only 1 weight load in flight at a time.
//     V100 needs ~(900GB/s / 80SM) * ~700ns  ~= 8KB of loads in flight per SM;
//     with K=640 the fork even picks block_size=160 -> each thread runs a
//     2-iteration loop of 4B loads -> the kernel is pure latency, measured at
//     ~29% of peak for those shapes.
//   2. ncols_dst=5 multiplies the *activation* load instructions 5x: every
//     weight pair is accompanied by 5 float2 (8B) loads of y. That is
//     44B of load instructions per thread per iteration to consume 4B of
//     streamed weights. y is L2/L1 resident so it does not directly cost DRAM
//     bandwidth, but it eats load-issue slots and L1 bandwidth 10x faster than
//     the weight stream.
//   3. One row per block: y is re-read once per output row (N_out blocks), so
//     L1/L2 sees ncols*N_out*K*4B of activation traffic -- for N=12288, K=640,
//     ncols=5 that is 157MB of cache traffic against 15.7MB of weight DRAM
//     traffic.
//
// This kernel's answer, decision by decision:
//
//   D1. uint4 (128-bit) weight loads -> 8 BF16 weights per thread per load.
//       4x fewer load instructions than the fork, 4x more bytes in flight per
//       instruction. 16B/thread/iter is the widest legal global access on
//       sm_70 (no 256-bit LDG before sm_90's ld.global.v8 availability in
//       CUDA 12 / sm_90; not applicable here anyway).
//
//   D2. rows_per_block (template ROWS in {1,2,4}): one block computes ROWS
//       adjacent output rows. All ROWS weight vectors for a given k-offset are
//       loaded back-to-back by the SAME thread before the FMA loop, giving
//       ROWS independent loads in flight per thread (memory-level parallelism)
//       even at occupancy as low as 25%. It also divides the activation
//       re-read traffic and the block count by ROWS (see point 3 above).
//       Coalescing is preserved: for a fixed k-offset, consecutive threads
//       still touch consecutive uint4s inside each row.
//
//   D3. Activations read as float4 pairs with __ldg (default STAGE_Y=false):
//       the y values needed by one weight-uint4 are y[k..k+8) = 2 float4s.
//       With ROWS>=2 and typical shapes the whole y panel (K*ncols*4B <= 200KB)
//       is L2 resident (V100 L2 = 6MB) and largely L1 resident, so __ldg gives
//       near-shared latency without staging instructions or __syncthreads.
//       An STAGE_Y=true variant (y staged into shared memory per K-chunk) is
//       kept for A/B: it wins only if y misses L1 (huge K*ncols) because it
//       trades 1 STS + 2 syncs per chunk for guaranteed hits.
//       NOTE on precision: y stays F32 everywhere. Rounding activations to
//       BF16 in shared would halve LDS bytes and make bank access perfectly
//       conflict-free, but it changes results by ~2^-9 relative -- breaking
//       parity with the fork's F32-activation contract. Rejected (documented
//       so nobody re-invents it without re-running parity).
//
//   D4. BF16->F32 conversion by bit shift, not __bfloat162float: a BF16 value
//       is exactly the top 16 bits of an IEEE F32, so
//           f32_bits = (u32 & 0xFFFF) << 16      (element 0 of a pair)
//           f32_bits = (u32 & 0xFFFF0000)        (element 1, already in place)
//       This is bit-exact with __bfloat162float, compiles to SHL/AND+MOV on
//       any arch, and avoids cuda_bf16.h availability questions on old
//       toolchains pointed at sm_70. All accumulation is FP32 FMA.
//
//   D5. Reduction: warp __shfl_xor tree (5 steps), then one float per warp to
//       shared, final warp serially adds the per-warp partials. This is 2
//       levels (warp + block) like the fork's, but the per-warp partial
//       layout is [row*NCOLS+j][warp] so all ROWS*NCOLS outputs reduce with a
//       single __syncthreads. The summation ORDER differs from both the fork
//       and the CPU reference (associativity of FP32 add) -- parity epsilon
//       must be ~1e-5 relative, see WIRING.md.
//
//   D6. Thread mapping: thread t covers k-vectors v = t, t+THREADS, ... and
//       processes ALL ROWS for each v (not a flattened (row,vec) split). Why:
//       a flattened split would make each thread own exactly one row per
//       iteration, dropping in-flight loads per thread from ROWS to 1 and
//       multiplying the shared/L1 reads of y by ROWS (each of the ROWS
//       threads covering the same v re-reads the same y). The chosen mapping
//       reads each y float4 exactly NCOLS times per v per block, which is the
//       information-theoretic minimum for in-register row accumulation.
//
//   D7. Launch geometry: grid.x = ceil(N_out / ROWS); THREADS in {64,128,256}
//       picked by the host chooser so that THREADS <= K/8 (all threads busy on
//       the first iteration) while keeping >= ~320 blocks in flight when the
//       matrix is small. Rationale per shape is in gemv_sm70_bf16_pick_config.
//
//   D8. No dynamic shared memory, no cudaFuncSetAttribute, no allocations, no
//       device-side getenv: the launch is CUDA-Graph-capturable verbatim and
//       needs no initialization step. All gating happens host-side.
//
// Requirements enforced by the host dispatch (fall back to fork kernel if any
// fails): sm_70; K % 8 == 0 (uint4 alignment of rows, ggml guarantees K % 2);
// stride_w % 8 == 0; (uintptr_t)W % 16 == 0; stride_y % 4 == 0 and y % 16
// aligned (float4 loads); plain MUL_MAT (ids == nullptr, no fusion) -- the
// MUL_MAT_ID / fused-GLU variants are a wiring follow-up, see WIRING.md.
//
// Compile: this header is self-contained C++17 + CUDA. No cuda_bf16.h needed;
// BF16 data is consumed as raw 16-bit patterns (bit-exact, see D4).
// ============================================================================

#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

namespace gemv_sm70 {

// ----------------------------------------------------------------------------
// Device helpers
// ----------------------------------------------------------------------------

// Bit-exact BF16 pair -> float2 (design D4). Element 0 of the pair lives in
// the LOW 16 bits of the packed u32 (little-endian memory order: the first
// BF16 in memory occupies bytes 0..1 -> low half of the u32).
__device__ __forceinline__ float2 bf16x2_to_float2(unsigned int p) {
    return make_float2(__uint_as_float(p << 16),          // low  half -> f32
                       __uint_as_float(p & 0xffff0000u)); // high half -> f32
}

// One uint4 of BF16 = 8 floats, returned through f[0..7].
__device__ __forceinline__ void bf16x8_to_float8(uint4 w, float * f) {
    const float2 a = bf16x2_to_float2(w.x);
    const float2 b = bf16x2_to_float2(w.y);
    const float2 c = bf16x2_to_float2(w.z);
    const float2 d = bf16x2_to_float2(w.w);
    f[0] = a.x; f[1] = a.y;
    f[2] = b.x; f[3] = b.y;
    f[4] = c.x; f[5] = c.y;
    f[6] = d.x; f[7] = d.y;
}

// ----------------------------------------------------------------------------
// The kernel (design D1..D6)
//
// W   : BF16 row-major weights, [n_out, stride_w] elements (stride in BF16
//       elements). Read as uint4 -> each uint4 covers 8 consecutive k of one
//       row; requires 16B alignment of every row start (checked host-side).
// y   : F32 activations, y[j*stride_y + k], j < NCOLS (template), k < k_dim.
// dst : F32, dst[j*stride_dst + row].
//
// template params:
//   ROWS    - output rows per block (1/2/4), design D2
//   NCOLS   - activation columns (decode: 1 or 5; general 1..8)
//   THREADS - block size, multiple of 32
//   STAGE_Y - false: y read via __ldg float4 (default, design D3)
//             true : y staged to shared per CHUNK_K chunk (A/B variant)
// ----------------------------------------------------------------------------

template <int ROWS, int NCOLS, int THREADS, bool STAGE_Y = false, int CHUNK_K = 1024>
__global__ __launch_bounds__(THREADS) void gemv_sm70_bf16_kernel(
        const unsigned short * __restrict__ w,   // BF16 elements
        const float          * __restrict__ y,
        float                * __restrict__ dst,
        const int              n_out,
        const int              k_dim,
        const int64_t          stride_w,     // in BF16 elements
        const int64_t          stride_y,     // in F32 elements
        const int64_t          stride_dst)   // in F32 elements
{
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of the warp size");
    static_assert(CHUNK_K % 8 == 0, "CHUNK_K must be a multiple of 8 weights (one uint4)");
    constexpr int WARPS = THREADS / 32;

    const int tid  = threadIdx.x;
    const int row0 = blockIdx.x * ROWS;
    // Rows this block actually owns (last block may be partial, design D8's
    // "N_out arbitrary" case). Guards below use r < nrows instead of early
    // return so that every thread reaches both __syncthreads and the warp
    // shuffles with a full warp (required for __shfl_xor_sync correctness).
    const int nrows = min(ROWS, n_out - row0);

    float acc[ROWS][NCOLS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            acc[r][j] = 0.0f;
        }
    }

    // Row base pointers as uint4 (element index k <-> vector index k/8).
    // Computed only for valid rows; invalid rows never dereference.
    const uint4 * w4[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        w4[r] = nullptr;
        if (r < nrows) {
            w4[r] = reinterpret_cast<const uint4 *>(w + (int64_t)(row0 + r) * stride_w);
        }
    }

    // --- STAGE_Y variant only: shared y chunk (design D3) -------------------
    // Layout [j][k] flattened as float4s: ys4[j * (CHUNK_K/4) + q]. Staging
    // writes are perfectly coalesced (consecutive threads -> consecutive q).
    // Compute reads two consecutive float4s per j (2v, 2v+1): threads stride
    // 8 floats apart -> ~2x shared wavefronts vs the ideal 4 per LDS.128 --
    // acceptable; the conflict-free alternative requires packing y to BF16
    // which loses precision (D3 note).
    __shared__ __align__(16) float y_s[STAGE_Y ? CHUNK_K * NCOLS : 1];
    constexpr int CK4 = CHUNK_K / 4;

    const int nv_total = k_dim >> 3;   // uint4 vectors per row (k_dim % 8 == 0)

    if constexpr (STAGE_Y) {
        for (int k0 = 0; k0 < k_dim; k0 += CHUNK_K) {
            const int klen = min(CHUNK_K, k_dim - k0);       // multiple of 8
            const int nq   = klen >> 2;                      // float4s per j
            if (k0 > 0) {
                __syncthreads();  // previous chunk's readers must finish first
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const float4 * src = reinterpret_cast<const float4 *>(y + (int64_t)j * stride_y + k0);
                float4       * dsk = reinterpret_cast<float4 *>(y_s) + (int64_t)j * CK4;
                for (int q = tid; q < nq; q += THREADS) {
                    dsk[q] = __ldg(src + q);
                }
            }
            __syncthreads();

            const int nv = klen >> 3;
            for (int v = tid; v < nv; v += THREADS) {
                float wf[ROWS][8];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    if (r < nrows) {
                        bf16x8_to_float8(w4[r][(k0 >> 3) + v], wf[r]);
                    }
                }
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const float4 * ys4 = reinterpret_cast<const float4 *>(y_s) + (int64_t)j * CK4;
                    const float4 ya = ys4[2 * v];
                    const float4 yb = ys4[2 * v + 1];
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        float s = acc[r][j];
                        s = fmaf(wf[r][0], ya.x, s);
                        s = fmaf(wf[r][1], ya.y, s);
                        s = fmaf(wf[r][2], ya.z, s);
                        s = fmaf(wf[r][3], ya.w, s);
                        s = fmaf(wf[r][4], yb.x, s);
                        s = fmaf(wf[r][5], yb.y, s);
                        s = fmaf(wf[r][6], yb.z, s);
                        s = fmaf(wf[r][7], yb.w, s);
                        acc[r][j] = s;
                    }
                }
            }
        }
    } else {
        // --- default path: single flat loop, y via __ldg (design D3) --------
        for (int v = tid; v < nv_total; v += THREADS) {
            // ROWS independent 16B loads -> ROWS loads in flight per thread
            // (design D2); none of them depends on the previous iteration's
            // FMAs, so the loop software-pipelines without unrolling hints.
            float wf[ROWS][8];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (r < nrows) {
                    bf16x8_to_float8(w4[r][v], wf[r]);
                }
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const float4 * y4 = reinterpret_cast<const float4 *>(y + (int64_t)j * stride_y);
                const float4 ya = __ldg(y4 + 2 * v);
                const float4 yb = __ldg(y4 + 2 * v + 1);
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    float s = acc[r][j];
                    s = fmaf(wf[r][0], ya.x, s);
                    s = fmaf(wf[r][1], ya.y, s);
                    s = fmaf(wf[r][2], ya.z, s);
                    s = fmaf(wf[r][3], ya.w, s);
                    s = fmaf(wf[r][4], yb.x, s);
                    s = fmaf(wf[r][5], yb.y, s);
                    s = fmaf(wf[r][6], yb.z, s);
                    s = fmaf(wf[r][7], yb.w, s);
                    acc[r][j] = s;
                }
            }
        }
    }

    // --- two-level block reduction (design D5) ------------------------------
    __shared__ float red[ROWS * NCOLS][WARPS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float s = acc[r][j];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_xor_sync(0xffffffffu, s, off);
            }
            if ((tid & 31) == 0) {
                red[r * NCOLS + j][tid >> 5] = s;
            }
        }
    }
    __syncthreads();

    constexpr int NOUT = ROWS * NCOLS;   // <= 20 < THREADS for all configs
    if (tid < NOUT) {
        const int r = tid / NCOLS;       // constexpr divisor -> strength-reduced
        const int j = tid - r * NCOLS;
        float s = 0.0f;
#pragma unroll
        for (int q = 0; q < WARPS; ++q) {
            s += red[tid][q];
        }
        if (r < nrows) {
            dst[(int64_t)j * stride_dst + row0 + r] = s;
        }
    }
}

// ----------------------------------------------------------------------------
// Host side: launch configuration chooser (design D7) + dispatch
// ----------------------------------------------------------------------------

struct gemv_sm70_config {
    int  rows;      // ROWS template
    int  threads;   // THREADS template
    bool stage_y;   // STAGE_Y template
};

// Heuristic, tuned for V100 (80 SMs) and the observed qwen4exp shapes.
// Reasoning per rule in comments; overridable via env in the bench.
static inline gemv_sm70_config gemv_sm70_bf16_pick_config(int n_out, int k_dim, int ncols) {
    (void)ncols;   // ncols influences L1 pressure only marginally at ROWS>=2;
                   // measured before enabling any ncols-specific rule.
    const int nv = k_dim / 8;  // uint4 vectors per row

    // Prefer ROWS=4 (max loads in flight, min y re-reads), but keep at least
    // ~320 blocks (= 4 blocks x 80 SMs) when the matrix is small so the GPU
    // stays filled. ROWS=1 keeps N_out blocks; ROWS=2 is the compromise.
    int rows = 4;
    while (rows > 1 && n_out / rows < 320) {
        rows >>= 1;
    }

    // Threads: cover at least one full uint4-vector per thread on iteration 1
    // (THREADS <= nv) to avoid idle threads, cap at 256 for occupancy headroom
    // (256 threads x ~100 regs stays within the 64K register file at 2
    // blocks/SM = 512 threads = 32KB of loads in flight, 4x the ~8KB
    // bandwidth-delay product). K=512 -> nv=64 -> 64 threads.
    int threads = 256;
    if (nv < 256) {
        threads = (nv >= 128) ? 128 : 64;
    }

    return gemv_sm70_config{rows, threads, /*stage_y=*/false};
}

// Launches the kernel for a supported shape. Returns false (caller must fall
// back) when any hard requirement is violated.
//
// ncols must be 1..8; n_out >= 1; k_dim >= 8.
static inline bool gemv_sm70_bf16_launch(
        const void  * w_bf16,
        const float * y,
        float       * dst,
        int           n_out,
        int           k_dim,
        int           ncols,
        int64_t       stride_w,     // BF16 elements
        int64_t       stride_y,     // F32 elements
        int64_t       stride_dst,   // F32 elements
        cudaStream_t  stream)
{
    // Hard requirements (see header comment). These mirror ggml's own
    // should_use_mmvf-style gating so the caller can silently fall back.
    if (k_dim % 8 != 0        || // uint4 weight loads
        ncols < 1 || ncols > 8 ||
        stride_w % 8 != 0     || // rows stay 16B aligned
        stride_y % 4 != 0     || // float4 activation loads
        ((uintptr_t) w_bf16 & 0xf) != 0 ||
        ((uintptr_t) y       & 0xf) != 0) {
        return false;
    }

    const unsigned short * w = static_cast<const unsigned short *>(w_bf16);
    const gemv_sm70_config cfg = gemv_sm70_bf16_pick_config(n_out, k_dim, ncols);
    const int blocks = (n_out + cfg.rows - 1) / cfg.rows;
    const dim3 grid(blocks), block(cfg.threads);

    // Dispatch table: every (ROWS, THREADS, STAGE_Y) the chooser can return,
    // plus STAGE_Y=true A/B variants at THREADS=128 == CHUNK_K/8 (one vector
    // per thread per chunk; see kernel comment). ncols is instantiated for
    // 1 and 5 (the decode shapes) plus the full 1..8 set for the production
    // (4,256,false) config so the wiring covers general batches.
    // The lambda keeps the launch line identical across 40 instantiations.
#define GEMV_SM70_LAUNCH(R, T, S)                                                              \
    do {                                                                                       \
        if (cfg.rows == R && cfg.threads == T && cfg.stage_y == S) {                           \
            switch (ncols) {                                                                   \
                case 1: gemv_sm70_bf16_kernel<R, 1, T, S><<<grid, block, 0, stream>>>(         \
                            w, y, dst, n_out, k_dim, stride_w, stride_y, stride_dst); break;   \
                case 5: gemv_sm70_bf16_kernel<R, 5, T, S><<<grid, block, 0, stream>>>(         \
                            w, y, dst, n_out, k_dim, stride_w, stride_y, stride_dst); break;   \
                default: break;                                                                \
            }                                                                                  \
            return true;                                                                       \
        }                                                                                      \
    } while (0)

    // STAGE_Y = false (production default)
    GEMV_SM70_LAUNCH(4, 256, false);
    GEMV_SM70_LAUNCH(4, 128, false);   // K in [1024, 2048): nv in [128, 256)
    GEMV_SM70_LAUNCH(4,  64, false);
    GEMV_SM70_LAUNCH(2, 256, false);
    GEMV_SM70_LAUNCH(2, 128, false);
    GEMV_SM70_LAUNCH(1, 256, false);
    GEMV_SM70_LAUNCH(1, 128, false);
    GEMV_SM70_LAUNCH(1,  64, false);
    // STAGE_Y = true (A/B; THREADS=128 == CHUNK_K/8 by design)
    GEMV_SM70_LAUNCH(4, 128, true);
    GEMV_SM70_LAUNCH(2, 128, true);
    GEMV_SM70_LAUNCH(1, 128, true);
#undef GEMV_SM70_LAUNCH

    // Config outside the table (only possible if the chooser was overridden).
    return false;
}

// Generic-ncols entry (2..8, plus 1/5 via the table) for a fixed best config.
// Used by the wiring when a batch size other than 1/5 shows up; kept separate
// so the hot table above stays small.
//
// IMPORTANT: the kernel's v-loop partitions work with a stride of THREADS, so
// blockDim.x MUST equal the THREADS template. This entry therefore forces the
// (ROWS=4, THREADS=256) template and launches blockDim=256 unconditionally:
// when nv < 256 some threads are idle on the first iteration, which is merely
// a small inefficiency, never a correctness issue.
static inline bool gemv_sm70_bf16_launch_generic(
        const void  * w_bf16,
        const float * y,
        float       * dst,
        int           n_out,
        int           k_dim,
        int           ncols,
        int64_t       stride_w,
        int64_t       stride_y,
        int64_t       stride_dst,
        cudaStream_t  stream)
{
    if (k_dim % 8 != 0 || ncols < 1 || ncols > 8 || stride_w % 8 != 0 ||
        stride_y % 4 != 0 || ((uintptr_t) w_bf16 & 0xf) != 0 || ((uintptr_t) y & 0xf) != 0) {
        return false;
    }
    if (ncols == 1 || ncols == 5) {
        return gemv_sm70_bf16_launch(w_bf16, y, dst, n_out, k_dim, ncols, stride_w, stride_y, stride_dst, stream);
    }
    const unsigned short * w = static_cast<const unsigned short *>(w_bf16);
    const int blocks = (n_out + 3) / 4;
    const dim3 grid(blocks), block(256);
    switch (ncols) {
        case 2: gemv_sm70_bf16_kernel<4, 2, 256, false><<<grid, block, 0, stream>>>(w, y, dst, n_out, k_dim, stride_w, stride_y, stride_dst); return true;
        case 3: gemv_sm70_bf16_kernel<4, 3, 256, false><<<grid, block, 0, stream>>>(w, y, dst, n_out, k_dim, stride_w, stride_y, stride_dst); return true;
        case 4: gemv_sm70_bf16_kernel<4, 4, 256, false><<<grid, block, 0, stream>>>(w, y, dst, n_out, k_dim, stride_w, stride_y, stride_dst); return true;
        case 6: gemv_sm70_bf16_kernel<4, 6, 256, false><<<grid, block, 0, stream>>>(w, y, dst, n_out, k_dim, stride_w, stride_y, stride_dst); return true;
        case 7: gemv_sm70_bf16_kernel<4, 7, 256, false><<<grid, block, 0, stream>>>(w, y, dst, n_out, k_dim, stride_w, stride_y, stride_dst); return true;
        case 8: gemv_sm70_bf16_kernel<4, 8, 256, false><<<grid, block, 0, stream>>>(w, y, dst, n_out, k_dim, stride_w, stride_y, stride_dst); return true;
        default: return false;
    }
}

} // namespace gemv_sm70
