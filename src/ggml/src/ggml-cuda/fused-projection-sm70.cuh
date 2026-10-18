// ============================================================================
// fused_projection_sm70.cuh -- sm_70 (V100) fused per-layer BF16 GEMV group
// ============================================================================
//
// Target problem (qwen4exp decode on 6x V100 SXM2, R17 nsys):
//   One decoder layer issues ~16-17 small BF16 MUL_MATs (hyper-connection
//   down/up/inject, GDN wqkv/gate/beta/alpha/out, QSA wq/wk/wv/indexer/wo,
//   MoE router + shared-expert trio). Each is a separate kernel launch of
//   2-25 us; small ones run at <45% of HBM bandwidth because their tail
//   (few blocks) cannot fill 80 SMs. Summed: the BF16 matvec family is 33%
//   of all GPU kernel time while the GPU is only ~65% busy.
//
// What fuses and what cannot:
//   The projections of one layer form a dependency CHAIN, not a set:
//     wave 1: hc_down + hc_inject      (share y = normed wide stream, K=hc_dim)
//     wave 2: hc_up                     (y = silu(down out), K=hc_lr)
//     wave 3: wqkv/gate/beta/alpha      (share y = mixed hidden, K=n_embd)
//             [QSA: wq/wk/wv/index_q/index_k, same y]
//     wave 4: ssm_out / wo              (y = mixer output)
//     wave 5: hc_ffn_down + inject      (= wave 1 shapes)
//     wave 6: hc_ffn_up                 (= wave 2 shapes)
//     wave 7: router + shexp up/gate + shexp router (share y = mixed hidden)
//     wave 8: shexp down
//   A single launch per LAYER would need in-kernel cross-block dependencies
//   (Luce-style megakernel; rejected for v1 -- see F2). What IS legal and
//   what this kernel does: ONE LAUNCH PER WAVE -- all MUL_MATs that read the
//   same activation y with no producer/consumer edge between them.
//   Per layer this collapses ~16-17 launches into 8.
//
// Key structural fact this exploits (and requires): every matvec in one wave
// shares ONE activation y, hence ONE k_dim. So y/k_dim/ncols are scalar
// kernel arguments, and the per-matrix task descriptor only carries
// (w, dst, n_out, stride_w, stride_dst). Different waves never mix K.
//
// Design decisions, each with its why:
//
//   F1. Task table = array of per-matrix descriptors + cumulative row-block
//       offsets (block0[0..n]). grid.x = total row blocks = sum_i
//       ceil(n_out_i / ROWS). Each block binary-searches blockIdx.x in
//       block0[] (log2(n) <= 5 uniform L1 hits, once per block) and then
//       runs the SAME proven per-block GEMV body as kernel_sm70_bf16.cuh.
//       Why a search and not grid.y = matrix index: a 1-D grid lets the
//       hardware scheduler fill SMs from one global block pool -- a 2-D grid
//       would still work, but the 1-D flattening is what makes small
//       matrices' blocks backfill the tail of big matrices in the same wave.
//
//   F2. No cross-wave fusion / no device-side dependencies (v1). The Luce
//       megakernel (persistent CTAs spinning on ready-flags, moe-fused.cu
//       one-block whole-FFN style) would fold waves 1+2 etc. into one
//       launch, but on sm_70 that means either cooperative launch with
//       grid <= resident-block limit or deadlock risk, plus nondeterministic
//       reduction order across replays. The fork's parity culture rejected
//       exactly that for mmvf_multi ("changed summation order moves the
//       greedy text"). v1 keeps every wave an ordinary data-parallel launch:
//       CUDA-Graph-capturable verbatim, no spin loops, no atomics.
//
//   F3. Two table flavors, both CUDA-Graph compatible:
//       - dynamic: table lives in a small device buffer (pointer argument).
//         Contents may be updated between replays without re-capture. This
//         is the production flavor: the fork's graph replays freeze buffer
//         addresses, so a wave table (weights + dst scratch addresses) is
//         stable; the host rebuilds it only when the scheduler hands out
//         different addresses.
//       - static: the table is passed BY VALUE as a kernel parameter
//         (kernel params live in the constant bank; <= 32,764 bytes since
//         CUDA 12.1 -- our worst case is ~1 KB for 24 tasks). Capture
//         embeds the whole table into the graph node; nothing can drift.
//         Dynamic indexing of a by-value struct can spill it to local
//         memory; that cost is O(log n) scalar loads ONCE per block (the
//         hot loop touches only resolved scalars), so it is accepted. The
//         bench A/Bs both flavors so the choice is data, not faith.
//
//   F4. Memory pattern per block is bit-for-bit the single-matrix kernel's
//       (kernel_sm70_bf16.cuh D1-D6): uint4 weight loads (8 BF16), y via
//       __ldg float4 pairs, ROWS adjacent rows per block for load-level
//       parallelism, warp-shuffle + shared two-level reduction, FP32 FMA
//       with shift-based BF16->F32 conversion. Why: that kernel measured
//       61-90% of HBM peak on the big shapes; fusion must not regress the
//       per-block memory pipeline, only remove the launch/tail overhead
//       around it. Same reduction order as the single kernel when ROWS/
//       THREADS match => bit-identical results vs per-matrix launches.
//
//   F5. ROWS chosen per WAVE (not per matrix): rows = 4, halved while
//       total_rows/rows < 320 (keep ~4 blocks x 80 SMs filled). Same rule
//       as the single kernel's chooser, applied to the wave's total row
//       count. THREADS chosen per wave from k_dim (256/128/64 so that
//       THREADS <= k_dim/8 when possible). Both are launch-level (not
//       task-level) template parameters -- legal because one launch = one
//       wave = one k_dim.
//
//   F6. y sharing: every task in the wave reads the same y through the same
//       __ldg pattern, so y is L1/L2-resident once the first blocks touch
//       it -- small-matrix blocks then hit cache instead of replaying cold
//       misses per launch. The literal "read y once" would need persistent
//       blocks (F2); the cache-level reuse is what v1 gets.
//
//   F7. No K-split in v1. Waves with few long rows (hc mix: ~55 rows x
//       K=10240 after 6-way TP) have too few blocks to fill the GPU
//       (~0.7 waves). K-split needs either atomic float adds (nondetermin-
//       istic ORDER across replays: rejected, see F2) or a two-phase
//       reduce (a second launch: eats the fusion win). Documented as the
//       known follow-up for those two waves only.
//
//   F8. Zero-row tasks (6-way TP can hand a GPU 0 rows of a tiny matrix,
//       e.g. hc_inject N=4) are dropped host-side at build time.
//
//   F9. Requirements (host-enforced, fall back to per-matrix launches on
//       any violation): sm_70; k_dim % 8 == 0 and stride_w % 8 == 0 (uint4
//       row alignment); stride_y % 4 == 0 (float4 y); w, y 16-byte aligned;
//       every task n_out >= 1 and identical k_dim; ncols 1..8; n_task
//       1..FP_MAX_TASKS; total_blocks < 2^31.
//
//   F10. No dynamic shared memory, no cudaFuncSetAttribute, no allocations
//       in the launcher hot path (the dynamic table buffer is owned by the
//       caller-owned fused_proj_table, built once per wave per graph shape):
//       	launch is CUDA-Graph-capturable verbatim.
//
// Relation to prior art found in the trees:
//   * fork ggml-cuda/mmvf-multi.cu: same idea, capped at 4 segments, ONE
//     shared x, K <= 3072, T <= 4, total_rows <= 2048, one block per 1/4
//     rows with warps splitting K (a different, latency-oriented body);
//     measured no win and OFF by default (GGML_CUDA_MMVF_MULTI). This
//     kernel targets the big-shape waves the mmvf_multi shape gates exclude
//     (K=10240, N=1707), keeps the bandwidth-oriented uint4 body, and lifts
//     the segment cap to 24.
//   * Luce server/deps/llama.cpp/ggml/src/ggml-cuda/moe-fused.cu: one
//     block computes ALL experts (loop over experts, extern-shared staging
//     between the gate/up phase and the down phase, <<<1, block>>> grid).
//     Grid=1 is right for its small gfx1151 and tiny ff_dim; it would
//     serialize a V100. We take its lesson -- dependents share a launch by
//     staging in shared, phases sequential inside the kernel -- only where
//     it is free (we do not need it: waves are already dependency-free).
//
// Compile: self-contained C++17 + CUDA, no cuda_bf16.h (raw 16-bit
// patterns, bit-exact conversion), mirrors kernel_sm70_bf16.cuh.
// ============================================================================

#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

namespace fusedproj_sm70 {

// ----------------------------------------------------------------------------
// Task table
// ----------------------------------------------------------------------------

// Worst case = one full LAYER if a future variant ever merges waves: GDN 16
// projections + PLE 2 + MTP eh_proj 1 => 19. 24 leaves margin. The static
// flavor's by-value size stays ~1 KB (see F3).
#define FP_MAX_TASKS 24

// Device-side descriptor block, structure-of-arrays (F1):
//   - the binary search touches ONLY block0[] (contiguous ints, one or two
//     128-byte lines for 24 tasks) so the search stream stays cache-hot;
//   - the resolved task's fields (w/dst/n_out/strides) are then pulled as
//     scalars; no strided access ever enters the hot loop.
struct fused_proj_tasks {
    int n;                       // tasks present (after F8 zero-row dropping)
    int pad0;
    const unsigned short * w[FP_MAX_TASKS];  // BF16 row-major weights
    float       * dst[FP_MAX_TASKS];         // F32 [stride_dst, ncols]
    int           n_out[FP_MAX_TASKS];       // rows of THIS gpu's shard
    int64_t       stride_w[FP_MAX_TASKS];    // BF16 elements between rows
    int64_t       stride_dst[FP_MAX_TASKS];  // F32 elements between columns
    int           block0[FP_MAX_TASKS + 1];  // cumulative first block index
};

// Host-side input descriptor: what a graph walker extracts per MUL_MAT.
// k_dim lives here only for validation -- the wave shares one y/k_dim,
// passed separately to the launcher (F1).
struct fused_proj_desc {
    const void * w;          // BF16 weights, rows of k_dim elements
    float      * dst;        // F32 output, [n_out, ncols] column stride below
    int          n_out;
    int          k_dim;      // must equal the wave's k_dim (checked)
    int64_t      stride_w;   // BF16 elements
    int64_t      stride_dst; // F32 elements
};

// ----------------------------------------------------------------------------
// Device helpers (bit-exact BF16 -> F32; same as kernel_sm70_bf16.cuh D4)
// ----------------------------------------------------------------------------

__device__ __forceinline__ float2 fp_bf16x2_to_float2(unsigned int p) {
    return make_float2(__uint_as_float(p << 16),
                       __uint_as_float(p & 0xffff0000u));
}

__device__ __forceinline__ void fp_bf16x8_to_float8(uint4 wv, float * f) {
    const float2 a = fp_bf16x2_to_float2(wv.x);
    const float2 b = fp_bf16x2_to_float2(wv.y);
    const float2 c = fp_bf16x2_to_float2(wv.z);
    const float2 d = fp_bf16x2_to_float2(wv.w);
    f[0] = a.x; f[1] = a.y;
    f[2] = b.x; f[3] = b.y;
    f[4] = c.x; f[5] = c.y;
    f[6] = d.x; f[7] = d.y;
}

// ----------------------------------------------------------------------------
// The per-block GEMV body (F4): identical memory pipeline and reduction
// order to gemv_sm70_bf16_kernel. Split out so the dynamic-table and
// static-table kernels share one body; row0/nrows/w/dst/strides arrive
// already resolved by the caller's block-claim code.
//
// W   : BF16 row-major weights, rows of stride_w elements; row r starts at
//       w + (row0+r)*stride_w, read as uint4 (16B-aligned, host-checked).
// y   : F32 activations shared by the WHOLE wave, y[j*stride_y + k].
// dst : F32, dst[j*stride_dst + row].
// ----------------------------------------------------------------------------

template <int ROWS, int NCOLS, int THREADS>
__device__ __forceinline__ void fused_proj_gemv_block(
        const unsigned short * __restrict__ w,
        const float          * __restrict__ y,
        float                * __restrict__ dst,
        const int              n_out,
        const int              k_dim,
        const int64_t          stride_w,
        const int64_t          stride_y,
        const int64_t          stride_dst,
        const int              row0)
{
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of the warp size");
    constexpr int WARPS = THREADS / 32;

    const int tid  = threadIdx.x;
    // Rows owned by this block (last block of a task may be partial). Guards
    // use r < nrows instead of early return so every thread reaches both
    // __syncthreads and the shuffles with a full warp.
    const int nrows = min(ROWS, n_out - row0);

    float acc[ROWS][NCOLS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            acc[r][j] = 0.0f;
        }
    }

    const uint4 * w4[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        w4[r] = nullptr;
        if (r < nrows) {
            w4[r] = reinterpret_cast<const uint4 *>(w + (int64_t)(row0 + r) * stride_w);
        }
    }

    const int nv_total = k_dim >> 3;   // uint4 vectors per row (k_dim % 8 == 0)

    for (int v = tid; v < nv_total; v += THREADS) {
        // ROWS independent 16B loads in flight per thread (single-kernel D2).
        float wf[ROWS][8];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r < nrows) {
                fp_bf16x8_to_float8(w4[r][v], wf[r]);
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

    // Two-level reduction (single-kernel D5); per-warp partial layout
    // [row*NCOLS+j][warp] so all ROWS*NCOLS outputs use one __syncthreads.
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

    constexpr int NOUT = ROWS * NCOLS;   // <= 32 < THREADS for all configs
    if (tid < NOUT) {
        const int r = tid / NCOLS;
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

// Resolve which task this block works on: the unique t with
// block0[t] <= blockIdx.x < block0[t+1]. Binary search over a contiguous
// int array (<= 25 entries); every thread runs it redundantly with uniform
// data (broadcast loads), which is cheaper than a shared-memory broadcast
// because it skips the __syncthreads.
__device__ __forceinline__ int fused_proj_find_task(const int * block0, int n, int b) {
    // Plain loads, NOT __ldg: the static-table flavor passes block0 by value in
    // kernel PARAMETER space (the const bank), and __ldg on a parameter-space
    // address raises "operation not supported on global/shared address space"
    // (a sticky error that then poisons every later CUDA call -- found on V100,
    // CUDA 12.4). __ldg bought nothing here anyway: <= 5 reads per block, and
    // the dynamic table is L1-resident after the first block touches it.
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (b >= block0[mid + 1]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// ----------------------------------------------------------------------------
// Kernel 1: dynamic table (device buffer; F3 dynamic flavor)
// ----------------------------------------------------------------------------

template <int ROWS, int NCOLS, int THREADS>
__global__ __launch_bounds__(THREADS) void fused_projection_sm70_kernel(
        const fused_proj_tasks * __restrict__ tab,
        const float             * __restrict__ y,
        const int                 k_dim,
        const int64_t             stride_y)
{
    const int t = fused_proj_find_task(tab->block0, tab->n, blockIdx.x);
    const int row0 = (blockIdx.x - tab->block0[t]) * ROWS;
    // The per-task field loads are uniform (same t for every thread in the
    // block) and cold once per block: plain loads through L1 are enough, and
    // avoiding __ldg here sidesteps its pointer-type overload availability.
    fused_proj_gemv_block<ROWS, NCOLS, THREADS>(
            tab->w[t],
            y,
            tab->dst[t],
            tab->n_out[t],
            k_dim,
            tab->stride_w[t],
            stride_y,
            tab->stride_dst[t],
            row0);
}

// ----------------------------------------------------------------------------
// Kernel 2: static table (by-value kernel parameter = constant bank; F3)
//
// Same table struct, but NT is a compile-time bound so the whole descriptor
// set rides in the launch parameters and is frozen into a captured CUDA
// graph node. Dynamic indexing of the by-value struct is O(log NT) scalar
// loads once per block (see F3); the hot loop uses resolved scalars only.
// ----------------------------------------------------------------------------

template <int NT>
struct fused_proj_static_table {
    int n;
    int pad0;
    const unsigned short * w[NT];
    float       * dst[NT];
    int           n_out[NT];
    int64_t       stride_w[NT];
    int64_t       stride_dst[NT];
    int           block0[NT + 1];
};

template <int ROWS, int NCOLS, int THREADS, int NT>
__global__ __launch_bounds__(THREADS) void fused_projection_sm70_static_kernel(
        const fused_proj_static_table<NT> tab,
        const float * __restrict__ y,
        const int     k_dim,
        const int64_t stride_y)
{
    static_assert(NT >= 1 && NT <= FP_MAX_TASKS, "NT out of range");
    const int t = fused_proj_find_task(tab.block0, tab.n, blockIdx.x);
    const int row0 = (blockIdx.x - tab.block0[t]) * ROWS;
    fused_proj_gemv_block<ROWS, NCOLS, THREADS>(
            tab.w[t], y, tab.dst[t], tab.n_out[t], k_dim,
            tab.stride_w[t], stride_y, tab.stride_dst[t], row0);
}

// ----------------------------------------------------------------------------
// Host side: config chooser (F5), table builder, launchers
// ----------------------------------------------------------------------------

struct fused_proj_config {
    int rows;      // ROWS template
    int threads;   // THREADS template
};

// Same philosophy as gemv_sm70_bf16_pick_config, applied at WAVE granularity:
// fill ~320 blocks (4 x 80 SMs) from the wave's TOTAL rows, then size the
// block so THREADS <= k_dim/8 (all threads busy on iteration one).
static inline fused_proj_config fused_proj_pick_config(int k_dim, int total_rows) {
    int rows = 4;
    while (rows > 1 && total_rows / rows < 320) {
        rows >>= 1;
    }
    const int nv = k_dim / 8;
    int threads = 256;
    if (nv < 256) {
        threads = (nv >= 128) ? 128 : 64;
    }
    return fused_proj_config{rows, threads};
}

// Validate a wave's descriptors against the kernel's hard requirements (F9)
// and fill a device-layout table. Returns false (caller falls back to
// per-matrix launches) on any violation. Does NOT touch the GPU: the same
// filled struct feeds the one-shot memcpy path and the persistent table.
static inline bool fused_proj_fill_table(
        const fused_proj_desc * tasks,
        int                     n_task,
        int                     k_dim,
        int                     ROWS,        // chosen config rows
        fused_proj_tasks     * out)
{
    if (n_task < 1 || n_task > FP_MAX_TASKS) {
        return false;
    }
    if (k_dim % 8 != 0) {
        return false;   // uint4 weight loads
    }
    int total_blocks = 0;
    out->n = 0;
    for (int i = 0; i < n_task; ++i) {
        const fused_proj_desc & d = tasks[i];
        if (d.n_out < 1) {
            continue;   // F8: zero-row TP shard
        }
        if (d.k_dim != k_dim           ||
            d.stride_w % 8 != 0        ||   // rows stay 16B aligned
            ((uintptr_t) d.w & 0xf) != 0) {
            return false;
        }
        const int t = out->n;
        out->w[t]         = static_cast<const unsigned short *>(d.w);
        out->dst[t]       = d.dst;
        out->n_out[t]     = d.n_out;
        out->stride_w[t]  = d.stride_w;
        out->stride_dst[t] = d.stride_dst;
        out->block0[t]    = total_blocks;
        total_blocks     += (d.n_out + ROWS - 1) / ROWS;
        ++out->n;
    }
    if (out->n < 1) {
        return false;   // every task was a zero-row shard
    }
    out->block0[out->n] = total_blocks;   // sentinel (search upper bound)
    return true;
}

// Full requirement check for the wave-level scalars (F9).
static inline bool fused_proj_wave_ok(const float * y, int ncols, int64_t stride_y) {
    return ncols >= 1 && ncols <= 8 &&
           stride_y % 4 == 0 &&          // float4 activation loads
           ((uintptr_t) y & 0xf) == 0;
}

// Dispatch macro shared by both launcher flavors: every (ROWS, THREADS) the
// chooser can return, ncols instantiated 1..8. The lambda-in-macro keeps the
// launch line identical across 72 instantiations per kernel flavor.
// NOTE: no // comments inside the macro body after code -- a backslash after
// a line comment does not continue the macro (verified the hard way by the
// clang syntax check harness).
#define FP_LAUNCH_KERNEL(KERNEL_MACRO)                                                            \
    do {                                                                                          \
        if (cfg.rows == 4 && cfg.threads == 256) { KERNEL_MACRO(4, 256); }                        \
        else if (cfg.rows == 4 && cfg.threads == 128) { KERNEL_MACRO(4, 128); }                   \
        else if (cfg.rows == 4 && cfg.threads ==  64) { KERNEL_MACRO(4,  64); }                   \
        else if (cfg.rows == 2 && cfg.threads == 256) { KERNEL_MACRO(2, 256); }                   \
        else if (cfg.rows == 2 && cfg.threads == 128) { KERNEL_MACRO(2, 128); }                   \
        else if (cfg.rows == 2 && cfg.threads ==  64) { KERNEL_MACRO(2,  64); }                   \
        else if (cfg.rows == 1 && cfg.threads == 256) { KERNEL_MACRO(1, 256); }                   \
        else if (cfg.rows == 1 && cfg.threads == 128) { KERNEL_MACRO(1, 128); }                   \
        else if (cfg.rows == 1 && cfg.threads ==  64) { KERNEL_MACRO(1,  64); }                   \
        else { return false; }                                                                    \
    } while (0)

// One-shot dynamic-table launch (bench / bring-up path): copies the filled
// table to a caller-staging device buffer and launches. The production path
// should use fused_proj_table below, which keeps the device copy alive.
static inline bool fused_projection_sm70_launch(
        const fused_proj_desc * tasks,
        int                     n_task,
        const float           * y,
        int                     k_dim,
        int                     ncols,
        int64_t                 stride_y,
        fused_proj_tasks      * d_tab_staging,   // device buffer, >= sizeof(fused_proj_tasks)
        cudaStream_t           stream)
{
    if (!fused_proj_wave_ok(y, ncols, stride_y)) {
        return false;
    }
    int total_rows = 0;
    for (int i = 0; i < n_task; ++i) {
        if (tasks[i].n_out > 0) {
            total_rows += tasks[i].n_out;
        }
    }
    const fused_proj_config cfg = fused_proj_pick_config(k_dim, total_rows);

    fused_proj_tasks h_tab;
    if (!fused_proj_fill_table(tasks, n_task, k_dim, cfg.rows, &h_tab)) {
        return false;
    }
    if (cudaMemcpyAsync(d_tab_staging, &h_tab, sizeof(h_tab),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }
    const dim3 grid(h_tab.block0[h_tab.n]), block(cfg.threads);

#define FP_DYN_LAUNCH(R, T)                                                                       \
    do {                                                                                          \
        switch (ncols) {                                                                          \
            case 1: fused_projection_sm70_kernel<R, 1, T><<<grid, block, 0, stream>>>(            \
                        d_tab_staging, y, k_dim, stride_y); return true;                          \
            case 2: fused_projection_sm70_kernel<R, 2, T><<<grid, block, 0, stream>>>(            \
                        d_tab_staging, y, k_dim, stride_y); return true;                          \
            case 3: fused_projection_sm70_kernel<R, 3, T><<<grid, block, 0, stream>>>(            \
                        d_tab_staging, y, k_dim, stride_y); return true;                          \
            case 4: fused_projection_sm70_kernel<R, 4, T><<<grid, block, 0, stream>>>(            \
                        d_tab_staging, y, k_dim, stride_y); return true;                          \
            case 5: fused_projection_sm70_kernel<R, 5, T><<<grid, block, 0, stream>>>(            \
                        d_tab_staging, y, k_dim, stride_y); return true;                          \
            case 6: fused_projection_sm70_kernel<R, 6, T><<<grid, block, 0, stream>>>(            \
                        d_tab_staging, y, k_dim, stride_y); return true;                          \
            case 7: fused_projection_sm70_kernel<R, 7, T><<<grid, block, 0, stream>>>(            \
                        d_tab_staging, y, k_dim, stride_y); return true;                          \
            case 8: fused_projection_sm70_kernel<R, 8, T><<<grid, block, 0, stream>>>(            \
                        d_tab_staging, y, k_dim, stride_y); return true;                          \
            default: return false;                                                                \
        }                                                                                          \
    } while (0)

    FP_LAUNCH_KERNEL(FP_DYN_LAUNCH);
#undef FP_DYN_LAUNCH
}

// Static-table launch: table embedded in the launch parameters (F3). NT is
// the compile-time task bound chosen by the caller (waves have <= 8 tasks;
// NT = 8 covers every real wave, NT can go to FP_MAX_TASKS).
template <int NT = 8>
static inline bool fused_projection_sm70_launch_static(
        const fused_proj_desc * tasks,
        int                     n_task,
        const float           * y,
        int                     k_dim,
        int                     ncols,
        int64_t                 stride_y,
        cudaStream_t           stream)
{
    static_assert(NT >= 1 && NT <= FP_MAX_TASKS, "NT out of range");
    if (!fused_proj_wave_ok(y, ncols, stride_y)) {
        return false;
    }
    int total_rows = 0;
    for (int i = 0; i < n_task; ++i) {
        if (tasks[i].n_out > 0) {
            total_rows += tasks[i].n_out;
        }
    }
    const fused_proj_config cfg = fused_proj_pick_config(k_dim, total_rows);

    fused_proj_tasks h_tab;
    if (!fused_proj_fill_table(tasks, n_task, k_dim, cfg.rows, &h_tab)) {
        return false;
    }
    fused_proj_static_table<NT> tab;
    tab.n = h_tab.n;
    for (int i = 0; i < h_tab.n; ++i) {
        tab.w[i]         = h_tab.w[i];
        tab.dst[i]       = h_tab.dst[i];
        tab.n_out[i]     = h_tab.n_out[i];
        tab.stride_w[i]  = h_tab.stride_w[i];
        tab.stride_dst[i] = h_tab.stride_dst[i];
        tab.block0[i]    = h_tab.block0[i];
    }
    tab.block0[tab.n] = h_tab.block0[h_tab.n];
    const dim3 grid(tab.block0[tab.n]), block(cfg.threads);

#define FP_STATIC_LAUNCH(R, T)                                                                    \
    do {                                                                                          \
        switch (ncols) {                                                                          \
            case 1: fused_projection_sm70_static_kernel<R, 1, T, NT><<<grid, block, 0, stream>>>( \
                        tab, y, k_dim, stride_y); return true;                                    \
            case 2: fused_projection_sm70_static_kernel<R, 2, T, NT><<<grid, block, 0, stream>>>( \
                        tab, y, k_dim, stride_y); return true;                                    \
            case 3: fused_projection_sm70_static_kernel<R, 3, T, NT><<<grid, block, 0, stream>>>( \
                        tab, y, k_dim, stride_y); return true;                                    \
            case 4: fused_projection_sm70_static_kernel<R, 4, T, NT><<<grid, block, 0, stream>>>( \
                        tab, y, k_dim, stride_y); return true;                                    \
            case 5: fused_projection_sm70_static_kernel<R, 5, T, NT><<<grid, block, 0, stream>>>( \
                        tab, y, k_dim, stride_y); return true;                                    \
            case 6: fused_projection_sm70_static_kernel<R, 6, T, NT><<<grid, block, 0, stream>>>( \
                        tab, y, k_dim, stride_y); return true;                                    \
            case 7: fused_projection_sm70_static_kernel<R, 7, T, NT><<<grid, block, 0, stream>>>( \
                        tab, y, k_dim, stride_y); return true;                                    \
            case 8: fused_projection_sm70_static_kernel<R, 8, T, NT><<<grid, block, 0, stream>>>( \
                        tab, y, k_dim, stride_y); return true;                                    \
            default: return false;                                                                \
        }                                                                                          \
    } while (0)

    FP_LAUNCH_KERNEL(FP_STATIC_LAUNCH);
#undef FP_STATIC_LAUNCH
}

// ----------------------------------------------------------------------------
// Persistent wave table (production path; F10)
//
// One object per (layer, wave, graph shape). build() uploads the descriptor
// set once; launch() only issues the kernel -- no memcpy in the steady state,
// so a captured CUDA graph replays a single kernel node per wave. rebuild()
// is needed only when the scheduler changes buffer addresses (detected by
// sig(): a cheap fingerprint of the wave's pointers and shapes).
// ----------------------------------------------------------------------------

class fused_proj_table {
public:
    fused_proj_table() : d_tab_(nullptr) {}
    ~fused_proj_table() { free(); }

    fused_proj_table(const fused_proj_table &) = delete;
    fused_proj_table & operator=(const fused_proj_table &) = delete;

    // Allocates the device table and uploads the descriptors. Must be called
    // outside graph capture (cudaMalloc/cudaMemcpyAsync pre-capture, or on a
    // non-capturing stream).
    bool build(const fused_proj_desc * tasks, int n_task, int k_dim) {
        k_dim_ = k_dim;
        int total_rows = 0;
        for (int i = 0; i < n_task; ++i) {
            if (tasks[i].n_out > 0) {
                total_rows += tasks[i].n_out;
            }
        }
        cfg_ = fused_proj_pick_config(k_dim, total_rows);
        if (!fused_proj_fill_table(tasks, n_task, k_dim, cfg_.rows, &h_tab_)) {
            return false;
        }
        if (!d_tab_) {
            if (cudaMalloc(&d_tab_, sizeof(fused_proj_tasks)) != cudaSuccess) {
                d_tab_ = nullptr;
                return false;
            }
        }
        if (cudaMemcpy(d_tab_, &h_tab_, sizeof(h_tab_), cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        return true;
    }

    // Fingerprint: identical fingerprints mean the launch is byte-for-byte
    // repeatable (same pointers/shapes/config), so a captured graph node and
    // the uploaded table still agree.
    bool matches(const fused_proj_desc * tasks, int n_task, int k_dim) const {
        if (k_dim != k_dim_ || d_tab_ == nullptr) {
            return false;
        }
        int total_rows = 0;
        for (int i = 0; i < n_task; ++i) {
            if (tasks[i].n_out > 0) {
                total_rows += tasks[i].n_out;
            }
        }
        const fused_proj_config cfg = fused_proj_pick_config(k_dim, total_rows);
        if (cfg.rows != cfg_.rows || cfg.threads != cfg_.threads) {
            return false;
        }
        fused_proj_tasks ref;
        if (!fused_proj_fill_table(tasks, n_task, k_dim, cfg.rows, &ref)) {
            return false;
        }
        if (ref.n != h_tab_.n) {
            return false;
        }
        for (int i = 0; i < ref.n; ++i) {
            if (ref.w[i] != h_tab_.w[i] || ref.dst[i] != h_tab_.dst[i] ||
                ref.n_out[i] != h_tab_.n_out[i] || ref.stride_w[i] != h_tab_.stride_w[i] ||
                ref.stride_dst[i] != h_tab_.stride_dst[i]) {
                return false;
            }
        }
        return true;
    }

    // Steady-state launch: one kernel, no H2D copy, capturable verbatim.
    bool launch(const float * y, int ncols, int64_t stride_y, cudaStream_t stream) const {
        if (!fused_proj_wave_ok(y, ncols, stride_y) || d_tab_ == nullptr) {
            return false;
        }
        const fused_proj_config cfg = cfg_;   // FP_LAUNCH_KERNEL reads `cfg`
        const dim3 grid(h_tab_.block0[h_tab_.n]), block(cfg_.threads);

#define FP_TBL_LAUNCH(R, T)                                                                       \
    do {                                                                                          \
        switch (ncols) {                                                                          \
            case 1: fused_projection_sm70_kernel<R, 1, T><<<grid, block, 0, stream>>>(            \
                        d_tab_, y, k_dim_, stride_y); return true;                                \
            case 2: fused_projection_sm70_kernel<R, 2, T><<<grid, block, 0, stream>>>(            \
                        d_tab_, y, k_dim_, stride_y); return true;                                \
            case 3: fused_projection_sm70_kernel<R, 3, T><<<grid, block, 0, stream>>>(            \
                        d_tab_, y, k_dim_, stride_y); return true;                                \
            case 4: fused_projection_sm70_kernel<R, 4, T><<<grid, block, 0, stream>>>(            \
                        d_tab_, y, k_dim_, stride_y); return true;                                \
            case 5: fused_projection_sm70_kernel<R, 5, T><<<grid, block, 0, stream>>>(            \
                        d_tab_, y, k_dim_, stride_y); return true;                                \
            case 6: fused_projection_sm70_kernel<R, 6, T><<<grid, block, 0, stream>>>(            \
                        d_tab_, y, k_dim_, stride_y); return true;                                \
            case 7: fused_projection_sm70_kernel<R, 7, T><<<grid, block, 0, stream>>>(            \
                        d_tab_, y, k_dim_, stride_y); return true;                                \
            case 8: fused_projection_sm70_kernel<R, 8, T><<<grid, block, 0, stream>>>(            \
                        d_tab_, y, k_dim_, stride_y); return true;                                \
            default: return false;                                                                \
        }                                                                                          \
    } while (0)

        FP_LAUNCH_KERNEL(FP_TBL_LAUNCH);
#undef FP_TBL_LAUNCH
    }

    int total_blocks() const { return d_tab_ ? h_tab_.block0[h_tab_.n] : 0; }
    const fused_proj_config & config() const { return cfg_; }

private:
    void free() {
        if (d_tab_) {
            cudaFree(d_tab_);
            d_tab_ = nullptr;
        }
    }
    fused_proj_tasks     h_tab_{};
    fused_proj_tasks   * d_tab_ = nullptr;
    fused_proj_config    cfg_{4, 256};
    int                  k_dim_ = 0;
};

} // namespace fusedproj_sm70
