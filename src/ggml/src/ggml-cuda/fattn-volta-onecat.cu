// SPDX-License-Identifier: Apache-2.0
// FlashAttention decode backend for Volta (SM70), ported from the 1Cat-vLLM
// Flash-V100 kernels (onecat-kernels/attention/sm70_grouped_long/kernel/
// grouped-attention.cu, Apache-2.0, Copyright (c) 2026, 1CatAI).
//
// Ported device kernels:
//   * flash_attention_decode_xqa_tc_partition_kernel_256_wide
//       (FP16-KV instantiation, GROUP_SIZE=6, generic runtime page size,
//        no seq-len routing, no QK software pipeline, no FP8 paths)
//       -> onecat_xqa_partition_kernel<PARTITION_SIZE>
//   * flash_attention_decode_reduce_kernel (__half partial variant)
//       -> onecat_reduce_f32_kernel<PARTITION_SIZE> (writes FP32 like ggml)
//
// ggml-side changes to the ported partition kernel:
//   * seq_lens/active_num_partitions device arrays are replaced by one
//     uniform runtime n_kv_all argument (ggml decode has no varlen batch),
//   * an additive F16 mask (same semantics as fattn-mma-f16: score += mask,
//     non-finite masks to -inf) is applied to the FP32 score tile before the
//     online softmax, applied per-thread to avoid a new barrier,
//   * k_scale/v_scale are dropped (FP16 KV only).
//
// The paged-KV block table of the original kernel is kept as the addressing
// mechanism, but the adapter feeds it a degenerate single-page-per-stream
// table: page i == stream i, page size == padded n_kv, block stride ==
// K->nb[3]/2. This reproduces ggml's contiguous per-stream KV cache rows
// exactly without touching the kernel's addressing math.

#include "fattn-volta-onecat.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "fattn-common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace ggml_onecat_v100 {

// Interface assumptions of the original Flash-V100 XQA kernel that this port
// preserves (see patches/onecat-flashv100-NOTES.md for the full table):
//   * Q is FP16 [batch, heads_q, 256], batch == query stream,
//   * K/V are FP16 paged caches with runtime page size,
//     token_offset -> page_ids[token_offset / page_size] * block_stride
//     + (token_offset % page_size) * token_stride + kv_head * head_stride,
//   * one query token per row set, GROUP_SIZE=6 query heads per CTA,
//     q_head_base = kv_head_idx * 6,
//   * online softmax in FP32, probabilities in FP32 for the PV product,
//   * split-K partitions of PARTITION_SIZE tokens, merged by a reduce kernel.
constexpr int kWarpSize                = 32;
constexpr int kThreads                 = 256; // kXQATC256WideThreads
constexpr int kWarps                   = kThreads / kWarpSize;
constexpr int kBlockN                  = 128; // kXQATCBlockN
constexpr int kBlockM                  = 8;   // kXQATC256WideBlockM
constexpr int kGroupSize               = 6;   // query heads per CTA
constexpr int kQStride                 = 264; // padded, bank-conflict avoiding
constexpr int kKVStride                = 136; // padded, bank-conflict avoiding
constexpr int kPanelDim                = 128; // kXQATCStride (D / kNumPanels)
constexpr int kNumPanels               = 256 / kPanelDim;
constexpr int kPageIdsCapacity         = kBlockN / 4;
constexpr int kThreadsPerRow           = kWarpSize; // kXQATC256WideThreadsPerRow
constexpr float kNegInf                = -1.0e30f;  // kXQANegInf
constexpr int kMaxPartitions           = 512;

constexpr int kD = 256;

struct alignas(256) SmemLayout {
    alignas(16) __half q[kBlockM * kQStride];
    union {
        alignas(16) __half k[kBlockN * kKVStride];
        alignas(16) __half v[kBlockN * kKVStride];
    } reuse_kv;
    struct {
        alignas(16) float s[kBlockM * kBlockN];
        alignas(16) __half p[kBlockM * kBlockN];
    } reuse_sp;
    alignas(16) float row_max[kBlockM];
    alignas(16) float row_sum[kBlockM];
    alignas(16) int   page_ids[kPageIdsCapacity];
};

constexpr size_t kSmemBytes = sizeof(SmemLayout);
static_assert(kSmemBytes <= 48 * 1024, "onecat XQA smem must fit the Volta default budget");

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

// The upstream kernel used its own inline-PTX "volta" fragment shim
// (include/fused_mma.h); the standard nvcuda::wmma API emits the identical
// wmma.*.m8n32k16 instructions for these fragment shapes.
namespace wmma = nvcuda::wmma;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[NUM_WARPS];
    __shared__ float result;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    val = threadIdx.x < NUM_WARPS ? shared[lane] : 0.f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
        if (lane == 0) {
            result = val;
        }
    }
    __syncthreads();
    return result;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[NUM_WARPS];
    __shared__ float result;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    val = warp_reduce_max(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    val = threadIdx.x < NUM_WARPS ? shared[lane] : -1.0e20f;
    if (warp == 0) {
        val = warp_reduce_max(val);
        if (lane == 0) {
            result = val;
        }
    }
    __syncthreads();
    return result;
}

// Ported from load_xqa_tc_kv_panel (BLOCK_SIZE == 0 generic page size,
// FP16 KV, no FP8 pair load, no LUT).
__device__ __forceinline__ void onecat_load_kv_panel_fp16(
        __half * __restrict__ shared_kv, const __half * __restrict__ kv_cache,
        const int * __restrict__ page_ids, const int valid_kv_tile_rows,
        const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
        const int tile_page_offset, const int kv_tile_start, const int block_size,
        const int kv_head_idx, const int64_t block_stride, const int64_t token_stride,
        const int64_t head_stride, const int panel_offset, const int copy_thread_idx) {
    uint4 * shared_vec = reinterpret_cast<uint4 *>(shared_kv);
    const uint4 * cache_vec = reinterpret_cast<const uint4 *>(kv_cache);
    const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
    for (int copy_idx = copy_thread_idx; copy_idx < copy_count; copy_idx += kThreads) {
        const int row     = copy_idx / panel_d_stride_uint4;
        const int vec_col = copy_idx % panel_d_stride_uint4;
        const int token_offset = tile_page_offset + kv_tile_start + row;
        const int logical_block = token_offset / block_size;
        const int block_offset  = token_offset - logical_block * block_size;
        const int physical_block = page_ids[logical_block];
        const int64_t physical_offset =
            static_cast<int64_t>(physical_block) * block_stride +
            static_cast<int64_t>(block_offset) * token_stride +
            static_cast<int64_t>(kv_head_idx) * head_stride + panel_offset;
        shared_vec[row * kv_smem_stride_uint4 + vec_col] =
            __ldg(&cache_vec[physical_offset / 8 + vec_col]);
    }
}

// Ported from flash_attention_decode_xqa_tc_partition_kernel_256_wide
// (FP16-KV specialization). See the header comment for the ggml-side deltas.
template <int PARTITION_SIZE>
__global__ __launch_bounds__(kThreads, 1)
void onecat_xqa_partition_kernel(
        const __half * __restrict__ q, const __half * __restrict__ k_cache,
        const __half * __restrict__ v_cache, __half * __restrict__ tmp_out,
        float * __restrict__ max_logits, float * __restrict__ exp_sums,
        const int * __restrict__ block_table, const half * __restrict__ mask_h,
        const int n_kv_all, const int mask_batch_stride, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t q_stride0, const int64_t q_stride1,
        const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
        const int64_t tmp_out_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t k_block_stride,
        const int64_t k_token_stride, const int64_t k_head_stride,
        const int64_t v_block_stride, const int64_t v_token_stride,
        const int64_t v_head_stride, const float softmax_scale) {
    constexpr int WMMA_M = 8;
    constexpr int WMMA_N = 32;
    constexpr int WMMA_K = 16;
    constexpr int kPVPanelDim = kPanelDim;
    constexpr int kNumPVPanels = kD / kPVPanelDim;
    // FP16 KV keeps softmax P in fp32 through PV (kKeepPfp32 upstream).
    constexpr bool kKeepPfp32 = true;
    constexpr int q_global_stride_uint4 = kD / 8;
    constexpr int q_smem_stride_uint4   = kQStride / 8;
    constexpr int kv_smem_stride_uint4  = kKVStride / 8;
    constexpr int panel_d_stride_uint4  = kPanelDim / 8;
    constexpr int kAccumsPerThread      = kD / kWarpSize;
    static_assert(kGroupSize == 4 || kGroupSize == 6 || kGroupSize == 8,
                  "port keeps the upstream GROUP_SIZE constraint");

    const int batch_idx     = blockIdx.x;
    const int kv_head_idx   = blockIdx.y;
    const int partition_idx = blockIdx.z;

    if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
            partition_idx >= max_num_partitions) {
        return;
    }

    const int seq_len = n_kv_all;
    const int start_token_idx = partition_idx * PARTITION_SIZE;
    if (seq_len <= 0 || start_token_idx >= seq_len) {
        return;
    }
    const int seq_num_partitions = (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
    const int effective_num_partitions = min(max_num_partitions, seq_num_partitions);
    if (partition_idx >= effective_num_partitions) {
        return;
    }

    const int q_head_base = kv_head_idx * kGroupSize;
    if (q_head_base + kGroupSize > num_heads_q) {
        return;
    }

    const int tid     = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane_id = tid % kWarpSize;
    const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
    const int num_k_tiles  = (part_tokens + kBlockN - 1) / kBlockN;
    const int * block_table_seq = block_table + int64_t(batch_idx) * max_num_blocks;

    extern __shared__ char smem_raw[];
    auto & smem = *reinterpret_cast<SmemLayout *>(smem_raw);
    __half * sQ = smem.q;
    __half * sK = smem.reuse_kv.k;
    __half * sV = smem.reuse_kv.v;
    float * sS  = smem.reuse_sp.s;
    __half * sP = smem.reuse_sp.p;
    float row_max_reg = kNegInf;
    float row_sum_reg = 0.f;
    float out_acc[kAccumsPerThread];
#pragma unroll
    for (int i = 0; i < kAccumsPerThread; ++i) {
        out_acc[i] = 0.f;
    }

    const uint4 * q_vec = reinterpret_cast<const uint4 *>(q);
    uint4 * sQ_vec = reinterpret_cast<uint4 *>(sQ);
    for (int idx = tid; idx < kGroupSize * q_global_stride_uint4; idx += kThreads) {
        const int row     = idx / q_global_stride_uint4;
        const int vec_col = idx % q_global_stride_uint4;
        const int64_t q_offset =
            static_cast<int64_t>(batch_idx) * q_stride0 +
            static_cast<int64_t>(q_head_base + row) * q_stride1;
        sQ_vec[row * q_smem_stride_uint4 + vec_col] =
            __ldg(&q_vec[q_offset / 8 + vec_col]);
    }
    for (int idx = tid; idx < (kBlockM - kGroupSize) * q_global_stride_uint4; idx += kThreads) {
        const int row     = kGroupSize + idx / q_global_stride_uint4;
        const int vec_col = idx % q_global_stride_uint4;
        sQ_vec[row * q_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
        const int tile_token_start = start_token_idx + block_n * kBlockN;
        const int valid_k_rows =
            min(kBlockN, part_tokens - block_n * kBlockN);
        const int start_page = tile_token_start / block_size;
        const int tile_page_offset = tile_token_start - start_page * block_size;
        const int page_count =
            (tile_page_offset + valid_k_rows + block_size - 1) / block_size;

        for (int idx = tid; idx < page_count; idx += kThreads) {
            smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
        }
        __syncthreads();

        for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
             kv_tile_start += kBlockN) {
            const int valid_kv_tile_rows =
                min(kBlockN, valid_k_rows - kv_tile_start);
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                            wmma::row_major>
                qk_a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                            wmma::col_major>
                qk_b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
                qk_acc_frag;
            if (warp_id < (kBlockN / WMMA_N)) {
                wmma::fill_fragment(qk_acc_frag, 0.0f);
            }

#pragma unroll
            for (int panel_idx = 0; panel_idx < kNumPanels; ++panel_idx) {
                const int panel_offset = panel_idx * kPanelDim;
                onecat_load_kv_panel_fp16(
                    sK, k_cache, smem.page_ids, valid_kv_tile_rows,
                    panel_d_stride_uint4, kv_smem_stride_uint4, tile_page_offset,
                    kv_tile_start, block_size, kv_head_idx, k_block_stride,
                    k_token_stride, k_head_stride, panel_offset, threadIdx.x);
                for (int idx = tid + valid_kv_tile_rows * panel_d_stride_uint4;
                     idx < kBlockN * panel_d_stride_uint4; idx += kThreads) {
                    const int row     = idx / panel_d_stride_uint4;
                    const int vec_col = idx % panel_d_stride_uint4;
                    reinterpret_cast<uint4 *>(sK)[row * kv_smem_stride_uint4 + vec_col] =
                        make_uint4(0, 0, 0, 0);
                }
                __syncthreads();

                if (warp_id < (kBlockN / WMMA_N)) {
                    const int tile_n = warp_id * WMMA_N;
#pragma unroll
                    for (int k_tile = 0; k_tile < (kPanelDim / WMMA_K); ++k_tile) {
                        const int k_offset = k_tile * WMMA_K;
                        wmma::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                                kQStride);
                        wmma::load_matrix_sync(
                            qk_b_frag, sK + tile_n * kKVStride + k_offset, kKVStride);
                        wmma::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
                    }
                }
                __syncthreads();
            }

            if (warp_id < (kBlockN / WMMA_N)) {
#pragma unroll
                for (int i = 0; i < qk_acc_frag.num_elements; ++i) {
                    qk_acc_frag.x[i] *= softmax_scale;
                }
                wmma::store_matrix_sync(sS + kv_tile_start + warp_id * WMMA_N,
                                         qk_acc_frag, kBlockN, wmma::mem_row_major);
            }
            __syncthreads();
        }

        if (tid < kGroupSize * kThreadsPerRow) {
            const int row           = tid / kThreadsPerRow;
            const int thread_in_row = tid % kThreadsPerRow;
            const unsigned mask = 0xffffffffu;
            float * sS_row_f  = sS + row * kBlockN;
            __half * sP_row_h = sP + row * kBlockN;
            float * sP_row_f  = sS + row * kBlockN;
            const int vec_cols   = valid_k_rows >> 2;
            const int tail_start = vec_cols << 2;
            const int vec_col    = thread_in_row;

            // ggml additive F16 mask, applied per-thread on the columns this
            // thread is about to read (no extra barrier needed).
            if (mask_h != nullptr) {
                const half * mask_row =
                    mask_h + static_cast<int64_t>(batch_idx) * mask_batch_stride;
                if (vec_col < vec_cols) {
                    float4 v4 = reinterpret_cast<float4 *>(sS_row_f)[vec_col];
                    const float m0 = __half2float(__ldg(&mask_row[tile_token_start + vec_col * 4 + 0]));
                    const float m1 = __half2float(__ldg(&mask_row[tile_token_start + vec_col * 4 + 1]));
                    const float m2 = __half2float(__ldg(&mask_row[tile_token_start + vec_col * 4 + 2]));
                    const float m3 = __half2float(__ldg(&mask_row[tile_token_start + vec_col * 4 + 3]));
                    v4.x = isfinite(m0) ? v4.x + m0 : kNegInf;
                    v4.y = isfinite(m1) ? v4.y + m1 : kNegInf;
                    v4.z = isfinite(m2) ? v4.z + m2 : kNegInf;
                    v4.w = isfinite(m3) ? v4.w + m3 : kNegInf;
                    reinterpret_cast<float4 *>(sS_row_f)[vec_col] = v4;
                }
                for (int c = tail_start + thread_in_row; c < valid_k_rows;
                     c += kThreadsPerRow) {
                    const float m = __half2float(__ldg(&mask_row[tile_token_start + c]));
                    sS_row_f[c] = isfinite(m) ? sS_row_f[c] + m : kNegInf;
                }
            }

            float thread_max = kNegInf;
            __half2 packed_exp0 = __float22half2_rn(make_float2(0.f, 0.f));
            __half2 packed_exp1 = __float22half2_rn(make_float2(0.f, 0.f));
            if (vec_col < vec_cols) {
                const float4 v4 = reinterpret_cast<float4 *>(sS_row_f)[vec_col];
                thread_max =
                    fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
            }
#pragma unroll
            for (int c = tail_start + thread_in_row; c < valid_k_rows;
                 c += kThreadsPerRow) {
                thread_max = fmaxf(thread_max, sS_row_f[c]);
            }
#pragma unroll
            for (int o = kThreadsPerRow / 2; o > 0; o >>= 1) {
                thread_max =
                    fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, kWarpSize));
            }

            const float row_max = __shfl_sync(mask, thread_max, 0, kWarpSize);
            const float old_max = __shfl_sync(mask, row_max_reg, 0, kWarpSize);
            const float new_max = fmaxf(old_max, row_max);
            const float exp_diff = __expf(old_max - new_max);

            float thread_sum = 0.f;
            if (vec_col < vec_cols) {
                const float4 v4 = reinterpret_cast<float4 *>(sS_row_f)[vec_col];
                const float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
                const float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
                const float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
                const float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));
                thread_sum += (e0 + e1) + (e2 + e3);
                if constexpr (kKeepPfp32) {
                    reinterpret_cast<float4 *>(sP_row_f)[vec_col] =
                        make_float4(e0, e1, e2, e3);
                } else {
                    packed_exp0 = __float22half2_rn(make_float2(e0, e1));
                    packed_exp1 = __float22half2_rn(make_float2(e2, e3));
                }
            }

#pragma unroll
            for (int c = tail_start + thread_in_row; c < kBlockN;
                 c += kThreadsPerRow) {
                const float v = (c < valid_k_rows) ? sS_row_f[c] : kNegInf;
                const float e = __expf(fmaxf(v - new_max, -80.0f));
                thread_sum += (c < valid_k_rows) ? e : 0.0f;
                if constexpr (kKeepPfp32) {
                    sP_row_f[c] = (c < valid_k_rows) ? e : 0.0f;
                } else {
                    sP_row_h[c] =
                        (c < valid_k_rows) ? __float2half_rn(e) : __float2half(0.f);
                }
            }

#pragma unroll
            for (int o = kThreadsPerRow / 2; o > 0; o >>= 1) {
                thread_sum += __shfl_down_sync(mask, thread_sum, o, kWarpSize);
            }

            const float row_sum = __shfl_sync(mask, thread_sum, 0, kWarpSize);
            const float old_sum = __shfl_sync(mask, row_sum_reg, 0, kWarpSize);

            if (thread_in_row == 0) {
                row_sum_reg = exp_diff * old_sum + row_sum;
                row_max_reg = new_max;
            }
            if constexpr (!kKeepPfp32) {
                __half2 * sP_half2 = reinterpret_cast<__half2 *>(sP_row_h);
                if (vec_col < vec_cols) {
                    const int base_offset = vec_col * 2;
                    sP_half2[base_offset] = packed_exp0;
                    sP_half2[base_offset + 1] = packed_exp1;
                }
            }

            if (block_n > 0) {
#pragma unroll
                for (int i = 0; i < kAccumsPerThread; ++i) {
                    out_acc[i] *= exp_diff;
                }
            }
        }
        __syncthreads();

        for (int panel_idx = 0; panel_idx < kNumPVPanels; ++panel_idx) {
            const int panel_offset = panel_idx * kPVPanelDim;
            for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
                 kv_tile_start += kBlockN) {
                const int valid_kv_tile_rows =
                    min(kBlockN, valid_k_rows - kv_tile_start);
                onecat_load_kv_panel_fp16(
                    sV, v_cache, smem.page_ids, valid_kv_tile_rows,
                    panel_d_stride_uint4, kv_smem_stride_uint4, tile_page_offset,
                    kv_tile_start, block_size, kv_head_idx, v_block_stride,
                    v_token_stride, v_head_stride, panel_offset, threadIdx.x);
                for (int idx = tid + valid_kv_tile_rows * panel_d_stride_uint4;
                     idx < kBlockN * panel_d_stride_uint4; idx += kThreads) {
                    const int row     = idx / panel_d_stride_uint4;
                    const int vec_col = idx % panel_d_stride_uint4;
                    reinterpret_cast<uint4 *>(sV)[row * kv_smem_stride_uint4 + vec_col] =
                        make_uint4(0, 0, 0, 0);
                }
                __syncthreads();

                if (tid < kGroupSize * kThreadsPerRow) {
                    const int row = tid / kThreadsPerRow;
                    const __half * sP_row = sP + row * kBlockN + kv_tile_start;
                    const float * sP_row_f = sS + row * kBlockN + kv_tile_start;
#pragma unroll
                    for (int token = 0; token < kBlockN; ++token) {
                        if (token >= valid_kv_tile_rows) {
                            break;
                        }
                        float prob;
                        if constexpr (kKeepPfp32) {
                            prob = sP_row_f[token];
                        } else {
                            prob = __half2float(sP_row[token]);
                        }
                        const __half * sV_row = sV + token * kKVStride;
#pragma unroll
                        for (int d_iter = 0; d_iter < (kPVPanelDim / kWarpSize);
                             ++d_iter) {
                            const int local_d = lane_id + d_iter * kWarpSize;
                            const int acc_idx =
                                panel_idx * (kPVPanelDim / kWarpSize) + d_iter;
                            out_acc[acc_idx] =
                                fmaf(prob, __half2float(sV_row[local_d]), out_acc[acc_idx]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    if (tid < kGroupSize * kThreadsPerRow) {
        const int row = tid / kThreadsPerRow;
        const int thread_in_row = tid % kThreadsPerRow;
        if (thread_in_row == 0) {
            smem.row_max[row] = row_max_reg;
            smem.row_sum[row] = row_sum_reg;
        }
    }
    __syncthreads();

    if (tid < kGroupSize * kThreadsPerRow) {
        const int row           = tid / kThreadsPerRow;
        const int thread_in_row = tid % kThreadsPerRow;
        const int head_idx = q_head_base + row;
        const float row_sum = smem.row_sum[row];
        const float inv_row_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
        __half * tmp_out_ptr =
            tmp_out + static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
            static_cast<int64_t>(head_idx) * tmp_out_stride1 +
            static_cast<int64_t>(partition_idx) * tmp_out_stride2;
        const float output_scale = inv_row_sum;
        for (int d = thread_in_row; d < kD; d += kThreadsPerRow) {
            tmp_out_ptr[d] = __float2half(out_acc[d / kWarpSize] * output_scale);
        }
        if (thread_in_row == 0) {
            const int64_t stats_index =
                static_cast<int64_t>(batch_idx) * stats_stride0 +
                static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
            max_logits[stats_index] = smem.row_max[row];
            exp_sums[stats_index] = row_sum;
        }
    }
}

// Ported from flash_attention_decode_reduce_kernel (fixed partition size,
// __half partials) with an FP32 output store for ggml.
template <int PARTITION_SIZE>
__global__ __launch_bounds__(kThreads, 1)
void onecat_reduce_f32_kernel(
        const __half * __restrict__ tmp_out, const float * __restrict__ max_logits,
        const float * __restrict__ exp_sums, float * __restrict__ out,
        const int batch_size, const int max_num_partitions, const int num_heads_q,
        const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
        const int64_t tmp_out_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t out_stride0,
        const int64_t out_stride1, const int n_kv_all) {
    constexpr int D = kD;
    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;

    if (batch_idx >= batch_size || head_idx >= num_heads_q) {
        return;
    }

    const int seq_len = n_kv_all;
    const int num_partitions =
        min(max_num_partitions, (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE);

    if (seq_len <= 0 || num_partitions <= 0) {
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            out[static_cast<int64_t>(batch_idx) * out_stride0 +
                 static_cast<int64_t>(head_idx) * out_stride1 + d] = 0.f;
        }
        return;
    }

    extern __shared__ float shared_mem[];
    float * max_shared   = shared_mem;
    float * weight_shared = shared_mem + max_num_partitions;

    float local_max = -1.0e20f;
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
        const int64_t stats_index =
            static_cast<int64_t>(batch_idx) * stats_stride0 +
            static_cast<int64_t>(head_idx) * stats_stride1 + i;
        const float m = max_logits[stats_index];
        max_shared[i] = m;
        local_max = fmaxf(local_max, m);
    }
    const float global_max = block_reduce_max<kWarps>(local_max);

    float local_sum = 0.f;
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
        const int64_t stats_index =
            static_cast<int64_t>(batch_idx) * stats_stride0 +
            static_cast<int64_t>(head_idx) * stats_stride1 + i;
        const float weight =
            exp_sums[stats_index] * __expf(max_shared[i] - global_max);
        weight_shared[i] = weight;
        local_sum += weight;
    }
    const float global_sum = block_reduce_sum<kWarps>(local_sum);
    const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;
    __syncthreads();

    const int64_t out_base = static_cast<int64_t>(batch_idx) * out_stride0 +
                             static_cast<int64_t>(head_idx) * out_stride1;
    const int64_t tmp_out_base =
        static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
        static_cast<int64_t>(head_idx) * tmp_out_stride1;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < num_partitions; ++i) {
            acc = fmaf(weight_shared[i],
                       static_cast<float>(
                           tmp_out[tmp_out_base +
                                   static_cast<int64_t>(i) * tmp_out_stride2 + d]),
                       acc);
        }
        out[out_base + d] = acc * inv_global_sum;
    }
}

__global__ void onecat_iota_kernel(int * __restrict__ p, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = i;
    }
}

// Q [D, 1, HS, S] (F32) -> contiguous F16 [S, HS, D].
__global__ void onecat_q_f32_to_f16_kernel(
        const float * __restrict__ src, __half * __restrict__ dst, const int64_t s2,
        const int64_t s3, const int heads, const int dim) {
    const int64_t total = static_cast<int64_t>(gridDim.y) * heads * dim;
    const int64_t idx =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) +
        static_cast<int64_t>(blockIdx.y) * (static_cast<int64_t>(gridDim.x) * blockDim.x);
    if (idx >= total) {
        return;
    }
    const int d = idx % dim;
    const int h = (idx / dim) % heads;
    const int s = idx / (static_cast<int64_t>(heads) * dim);
    dst[idx] = __float2half_rn(src[s * s3 + h * s2 + d]);
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

// ---------------------------------------------------------------------------
// host-side adapter
// ---------------------------------------------------------------------------

bool env_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CUDA_FATTN_ONECAT");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

static int select_partition_size(const int64_t n_kv) {
    static const int env_part = [] {
        const char * env = std::getenv("GGML_CUDA_FATTN_ONECAT_PART");
        const int v = env != nullptr ? std::atoi(env) : 0;
        return (v == 256 || v == 512 || v == 1024) ? v : 0;
    }();
    if (env_part != 0) {
        return env_part;
    }
    if (n_kv <= 8192) {
        return 256;
    }
    if (n_kv <= 65536) {
        return 512;
    }
    return 1024;
}

static int num_partitions_for(const int64_t n_kv, const int part_size) {
    const int64_t parts = (n_kv + part_size - 1) / part_size;
    return int(std::min<int64_t>(parts, kMaxPartitions));
}

bool supported(const ggml_tensor * dst) {
    // Local copy of ggml_cuda_fattn_kv_type_supported() from fattn.cu (it is
    // static there and not visible from this translation unit).
    auto kv_type_supported = [](const ggml_type type) {
        switch (type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
                return true;
            default:
                return false;
        }
    };

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q == nullptr || K == nullptr || V == nullptr || sinks != nullptr) {
        return false;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    if (dst->type != GGML_TYPE_F32 || Q->type != GGML_TYPE_F32 ||
            dst->ne[0] != kD || Q->ne[0] != kD || K->ne[0] != kD || V->ne[0] != kD ||
            Q->ne[1] != 1 || // single query token per stream (decode shape only)
            Q->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }

    if (K->ne[1] <= 0 || K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2] ||
            K->ne[3] != V->ne[3] || Q->ne[3] != K->ne[3] ||
            Q->ne[2] % K->ne[2] != 0 ||
            (Q->ne[2] / K->ne[2]) % kGroupSize != 0) {
        return false;
    }

    // The kernel's uint4 KV loads need 16-byte aligned strides and the FP16
    // conversion path needs a supported KV type (same family as the stock
    // MMA kernel).
    if (!kv_type_supported(K->type) || !kv_type_supported(V->type)) {
        return false;
    }
    for (const ggml_tensor * t : {K, V}) {
        if (t->nb[0] != ggml_type_size(t->type)) {
            return false;
        }
        if (t->type == GGML_TYPE_F16) {
            for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
                if (t->nb[i] % 16 != 0) {
                    return false;
                }
            }
        }
    }

    if (mask != nullptr) {
        if (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 ||
                mask->ne[0] != K->ne[1] || mask->ne[1] < Q->ne[1] ||
                mask->nb[0] != sizeof(half)) {
            return false;
        }
    }

    // All partitions must be coverable by the workspace cap.
    const int64_t n_kv = K->ne[1];
    const int part_size = select_partition_size(n_kv);
    if (n_kv > int64_t(num_partitions_for(n_kv, part_size)) * part_size) {
        return false;
    }

    return true;
}

struct extra_data {
    uintptr_t K;
    uintptr_t V;
    uintptr_t q_f16;
    uintptr_t tmp_out;
    uintptr_t max_logits;
    uintptr_t exp_sums;
    uintptr_t end;
    int part_size;
    int max_parts;
};

static extra_data get_extra(const ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool need_f16_K = K->type != GGML_TYPE_F16;
    const bool need_f16_V = V->type != GGML_TYPE_F16;
    const bool V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    const int64_t n_kv   = K->ne[1];
    const int64_t n_head = dst->ne[2]; // heads_q
    const int64_t n_strm = dst->ne[3];

    extra_data e = {};
    e.K = f16_extra.K;
    e.V = f16_extra.V;
    e.part_size = select_partition_size(n_kv);
    e.max_parts = num_partitions_for(n_kv, e.part_size);

    uintptr_t p = GGML_PAD(f16_extra.end, 256);
    e.q_f16 = p;      p = GGML_PAD(p + size_t(n_strm) * n_head * kD * sizeof(__half), 256);
    e.tmp_out = p;    p = GGML_PAD(p + size_t(e.max_parts) * n_strm * n_head * kD * sizeof(__half), 256);
    e.max_logits = p; p += size_t(e.max_parts) * n_strm * n_head * sizeof(float);
    e.exp_sums = p;   p += size_t(e.max_parts) * n_strm * n_head * sizeof(float);
    e.end = p;
    return e;
}

size_t get_alloc_size(const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(supported(dst));
    const extra_data e = get_extra(dst);
    return e.end - (uintptr_t) dst->data;
}

// Persistent per-device iota block table {0,1,...} so the degenerate paging
// adapter needs no per-call host-to-device traffic (CUDA-graph safe: the
// content never changes after initialization). Buffers are never freed
// because an earlier CUDA graph capture may still reference them.
static int * stream_table(const int device, const int min_entries, cudaStream_t stream) {
    struct per_device {
        int * ptr = nullptr;
        int entries = 0;
    };
    static std::mutex mu;
    static std::vector<per_device> tables; // index = ggml device id

    std::lock_guard<std::mutex> lock(mu);
    if (device >= (int) tables.size()) {
        tables.resize(device + 1);
    }
    const int want = std::max(min_entries, 64);
    if (tables[device].entries < want) {
        CUDA_CHECK(cudaMalloc(&tables[device].ptr, want * sizeof(int)));
        onecat_iota_kernel<<<(want + 255) / 256, 256, 0, stream>>>(tables[device].ptr, want);
        CUDA_CHECK(cudaGetLastError());
        tables[device].entries = want;
    }
    return tables[device].ptr;
}

template <int PART_SIZE>
static void launch_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const extra_data & e) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    cudaStream_t stream = ctx.stream();

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(scale));

    const int64_t n_head = Q->ne[2];
    const int64_t n_kv_head = K->ne[2];
    const int64_t n_strm = Q->ne[3];
    const int64_t n_kv = K->ne[1];
    const int64_t gqa = n_head / n_kv_head;
    const int kv_chunks = int(gqa / kGroupSize); // virtual KV heads per real KV head

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1], nb12 = K->nb[2], nb13 = K->nb[3];
    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1], nb22 = V->nb[2], nb23 = V->nb[3];

    const bool V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    if (K->type != GGML_TYPE_F16) {
        half * K_f16 = (half *) e.K;
        GGML_ASSERT(K_f16 != nullptr);
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16, ggml_nelements(K), stream);
            const size_t bs = ggml_blck_size(K->type);
            const size_t ts = ggml_type_size(K->type);
            nb11 = nb11 * bs * sizeof(half) / ts;
            nb12 = nb12 * bs * sizeof(half) / ts;
            nb13 = nb13 * bs * sizeof(half) / ts;
        } else {
            GGML_ASSERT(K->nb[0] == ggml_type_size(K->type));
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ggml_type_size(K->type);
            const int64_t s02 = nb12 / ggml_type_size(K->type);
            const int64_t s03 = nb13 / ggml_type_size(K->type);
            to_fp16(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, stream);
            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16;
    }

    if (V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21 = nb11;
            nb22 = nb12;
            nb23 = nb13;
        } else {
            half * V_f16 = (half *) e.V;
            GGML_ASSERT(V_f16 != nullptr);
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16, ggml_nelements(V), stream);
                const size_t bs = ggml_blck_size(V->type);
                const size_t ts = ggml_type_size(V->type);
                nb21 = nb21 * bs * sizeof(half) / ts;
                nb22 = nb22 * bs * sizeof(half) / ts;
                nb23 = nb23 * bs * sizeof(half) / ts;
            } else {
                GGML_ASSERT(V->nb[0] == ggml_type_size(V->type));
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ggml_type_size(V->type);
                const int64_t s02 = nb22 / ggml_type_size(V->type);
                const int64_t s03 = nb23 / ggml_type_size(V->type);
                to_fp16(V_data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, stream);
                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16;
        }
    }

    // Q: F32 [D, 1, HS, S] -> F16 [S, HS, D].
    {
        const int total = int(n_strm * n_head * kD);
        const int blocks_x = (total + 255) / 256;
        const dim3 blocks(blocks_x, 1, 1);
        onecat_q_f32_to_f16_kernel<<<blocks, 256, 0, stream>>>(
            (const float *) Q->data, (__half *) e.q_f16,
            Q->nb[2] / sizeof(float), Q->nb[3] / sizeof(float),
            int(n_head), kD);
        CUDA_CHECK(cudaGetLastError());
    }

    int * block_table = stream_table(ctx.device, int(n_strm), stream);
    const int max_num_blocks = 1; // one degenerate page per stream

    // Degenerate paging: page s covers the whole per-stream KV window.
    const int block_size = int(GGML_PAD(n_kv, 256));
    const int64_t k_block_stride = nb13 / sizeof(half);
    const int64_t k_token_stride = nb11 / sizeof(half);
    const int64_t k_head_stride  = 0; // real KV head folded into the base pointer
    const int64_t v_block_stride = nb23 / sizeof(half);
    const int64_t v_token_stride = nb21 / sizeof(half);
    const int64_t v_head_stride  = 0;

    const int64_t mask_batch_stride =
        mask != nullptr ? mask->nb[3] / sizeof(half) : 0;
    const half * mask_h = mask != nullptr ? (const half *) mask->data : nullptr;

    static bool smem_attr_set[GGML_CUDA_MAX_DEVICES] = {};
    if (!smem_attr_set[ctx.device]) {
        CUDA_CHECK(cudaFuncSetAttribute(onecat_xqa_partition_kernel<256>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
        CUDA_CHECK(cudaFuncSetAttribute(onecat_xqa_partition_kernel<512>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
        CUDA_CHECK(cudaFuncSetAttribute(onecat_xqa_partition_kernel<1024>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
        smem_attr_set[ctx.device] = true;
    }

    // Workspace strides (elements).
    const int64_t tmp_s0 = n_head * e.max_parts * kD;
    const int64_t tmp_s1 = e.max_parts * kD;
    const int64_t tmp_s2 = kD;
    const int64_t stats_s0 = n_head * e.max_parts;
    const int64_t stats_s1 = e.max_parts;
    const int64_t q_s0 = n_head * kD;
    const int64_t q_s1 = kD;

    const __half * q_base = (const __half *) e.q_f16;
    __half * tmp_base = (__half *) e.tmp_out;
    float * maxl_base = (float *) e.max_logits;
    float * sums_base = (float *) e.exp_sums;

    for (int kv_head = 0; kv_head < n_kv_head; ++kv_head) {
        for (int chunk = 0; chunk < kv_chunks; ++chunk) {
            const int64_t head_off = (kv_head * gqa + chunk * kGroupSize);
            const dim3 blocks(unsigned(n_strm), unsigned(kv_chunks), unsigned(e.max_parts));
            onecat_xqa_partition_kernel<PART_SIZE>
                <<<blocks, kThreads, kSmemBytes, stream>>>(
                    q_base + head_off * q_s1,
                    (const __half *) K_data + kv_head * (nb12 / sizeof(half)),
                    (const __half *) V_data + kv_head * (nb22 / sizeof(half)),
                    tmp_base + head_off * tmp_s1,
                    maxl_base + head_off * stats_s1,
                    sums_base + head_off * stats_s1,
                    block_table, mask_h, int(n_kv), int(mask_batch_stride),
                    int(n_strm), max_num_blocks, e.max_parts,
                    int(n_head), kv_chunks, block_size,
                    q_s0, q_s1,
                    tmp_s0, tmp_s1, tmp_s2, stats_s0, stats_s1,
                    k_block_stride, k_token_stride, k_head_stride,
                    v_block_stride, v_token_stride, v_head_stride,
                    scale);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    const dim3 reduce_blocks(unsigned(n_strm), unsigned(n_head), 1);
    const size_t reduce_smem = 2 * size_t(e.max_parts) * sizeof(float);
    onecat_reduce_f32_kernel<PART_SIZE>
        <<<reduce_blocks, kThreads, reduce_smem, stream>>>(
            tmp_base, maxl_base, sums_base, (float *) dst->data,
            int(n_strm), e.max_parts, int(n_head),
            tmp_s0, tmp_s1, tmp_s2, stats_s0, stats_s1,
            dst->nb[3] / sizeof(float), dst->nb[2] / sizeof(float), int(n_kv));
    CUDA_CHECK(cudaGetLastError());
}

void launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    GGML_ASSERT(supported(dst));
    const extra_data e = get_extra(dst);
    switch (e.part_size) {
        case 256:
            launch_impl<256>(ctx, dst, e);
            break;
        case 512:
            launch_impl<512>(ctx, dst, e);
            break;
        case 1024:
            launch_impl<1024>(ctx, dst, e);
            break;
        default:
            GGML_ABORT("onecat: invalid partition size %d", e.part_size);
    }
}

} // namespace ggml_onecat_v100

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
