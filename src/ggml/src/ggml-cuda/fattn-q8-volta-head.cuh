#pragma once

// Adapted from llama-volta-jusko at d343373dabaee9c726901a45a09aab77cbe80fe; source repository is MIT licensed.

#include "common.cuh"
#include "fattn-common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdlib>
#include <cstring>

namespace ggml_q8v_head {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

__device__ __forceinline__ int qp_get_i() {
    return threadIdx.x & 31;
}

__device__ __forceinline__ int qp_get_j(int l) {
    return l;
}

__device__ __forceinline__ int k_get_i() {
    const int lane = threadIdx.x & 31;
    return (lane / 16) * 4 + lane % 4;
}

__device__ __forceinline__ int k_get_j(int l) {
    return l;
}

__device__ __forceinline__ int d_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return (l & 2) + (lane & ~2);
}

__device__ __forceinline__ int d_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return (lane & 2) + (l & 5);
}

__device__ __forceinline__ int v_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return (l / 2) * 4 + lane % 4;
}

__device__ __forceinline__ int v_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return (lane / 16) * 2 + l % 2;
}

__device__ __forceinline__ void load_qp(half2 (&dst)[4], const half2 * base, int stride) {
    const int row = qp_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        dst[l] = base[row * stride + qp_get_j(l)];
    }
}

__device__ __forceinline__ void load_k(half2 (&dst)[4], const half2 * base, int stride) {
    const int row = k_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        dst[l] = base[row * stride + k_get_j(l)];
    }
}

__device__ __forceinline__ void load_v(half2 (&dst)[4], const half2 * base, int stride) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        dst[l] = base[v_get_i(l) * stride + v_get_j(l)];
    }
}

__device__ __forceinline__ void mma_qk(float (&d)[8], const half2 (&q)[4], const half2 (&k)[4]) {
    const int * A = reinterpret_cast<const int *>(q);
    const int * B = reinterpret_cast<const int *>(k);
    int * D = reinterpret_cast<int *>(d);
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]),
                   "+r"(D[6]), "+r"(D[7])
                 : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]),
                   "+r"(D[6]), "+r"(D[7])
                 : "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]));
}

__device__ __forceinline__ void mma_pv(half2 (&pv)[4], const half2 (&p)[4], const half2 (&v)[4]) {
    const int * P = reinterpret_cast<const int *>(p);
    const int * V = reinterpret_cast<const int *>(v);
    int * PV = reinterpret_cast<int *>(pv);
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
                 : "+r"(PV[0]), "+r"(PV[1]), "+r"(PV[2]), "+r"(PV[3])
                 : "r"(P[0]), "r"(P[1]), "r"(V[0]), "r"(V[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
                 : "+r"(PV[0]), "+r"(PV[1]), "+r"(PV[2]), "+r"(PV[3])
                 : "r"(P[2]), "r"(P[3]), "r"(V[2]), "r"(V[3]));
}

__device__ __forceinline__ void softmax_to_half2(half2 (&p)[4], const float (&d)[8]) {
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 4) {
        p[l0 / 2 + 0] = __floats2half2_rn(d[l0 + 0], d[l0 + 1]);
        p[l0 / 2 + 1] = __floats2half2_rn(d[l0 + 2], d[l0 + 3]);
        const int lane = threadIdx.x & 31;
        const int swap_idx = l0 / 2 + (((lane % 4) / 2) ^ 1);
        p[swap_idx] = __shfl_xor_sync(0xffffffffu, p[swap_idx], 2, 32);
    }
}

#endif

constexpr int D = 256;
constexpr int BC = 16;
constexpr int BR = 32;
constexpr int THREADS = 128;
constexpr int D_SLICE = 64;
constexpr int SMEM_STRIDE = 264;
constexpr float LOG2E = 1.4426950408889634074f;
constexpr int LONG_SPLITS = 160;

enum class variant {
    none,
    qh12_kvh1_t1,
    qh12_kvh1_t4,
    qh12_kvh1_t32,
    qh24_kvh2_t1,
    qh24_kvh2_t4,
    qh24_kvh2_t32,
};

static inline int split_limit() {
    static const int value = []() {
        const char * env = std::getenv("GGML_CUDA_VOLTA_Q8_HEAD_SPLITS");
        if (env == nullptr) {
            return LONG_SPLITS;
        }
        const int parsed = std::atoi(env);
        return parsed >= 1 && parsed <= LONG_SPLITS ? parsed : LONG_SPLITS;
    }();
    return value;
}

static inline int split_count(int n_kv) {
    const int tiles = (n_kv + BC - 1) / BC;
    const int requested = std::min(split_limit(), tiles);
    const int tiles_per_split = (tiles + requested - 1) / requested;
    return (tiles + tiles_per_split - 1) / tiles_per_split;
}

template<int QH, int KVH, int G, int T>
struct extra_data {
    uintptr_t pacc = 0;
    uintptr_t pm = 0;
    uintptr_t pl = 0;
    uintptr_t end = 0;
    int splits = 0;
};

template<int QH, int KVH, int G, int T>
static inline extra_data<QH, KVH, G, T> get_extra(const ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    extra_data<QH, KVH, G, T> e{};
    e.splits = split_count((int) K->ne[1]);
    e.end = (uintptr_t) dst->data + ggml_nbytes(dst);
    e.end = GGML_PAD(e.end, 128);
    e.pacc = e.end;
    e.end += (size_t) e.splits * T * QH * D * sizeof(__nv_bfloat16);
    e.end = GGML_PAD(e.end, 128);
    e.pm = e.end;
    e.end += (size_t) e.splits * T * QH * sizeof(float);
    e.end = GGML_PAD(e.end, 128);
    e.pl = e.end;
    e.end += (size_t) e.splits * T * QH * sizeof(float);
    return e;
}

template<int QH, int T>
__device__ __forceinline__ int64_t pacc_idx(int qh, int d, int tok, int split) {
    return d + (int64_t) D * (qh + (int64_t) QH * (tok + (int64_t) T * split));
}

template<int QH, int T>
__device__ __forceinline__ int64_t pstat_idx(int qh, int tok, int split) {
    return qh + (int64_t) QH * (tok + (int64_t) T * split);
}

template<int G>
__device__ __forceinline__ int row_qh(int row, int kvh) {
    return kvh * G + row % G;
}

template<int G>
__device__ __forceinline__ int row_tok(int row) {
    return row / G;
}

__device__ __forceinline__ int4 deq8(const block_q8_0 & b, int off) {
    const float s = __half2float(b.d);
    half2 h[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h[i] = __floats2half2_rn((float) b.qs[off + 2 * i] * s, (float) b.qs[off + 2 * i + 1] * s);
    }
    return *reinterpret_cast<int4 *>(h);
}

template<int QH, int KVH, int G, int T>
__launch_bounds__(THREADS, 2)
static __global__ void partial_kernel(
        const float * q, const block_q8_0 * K, const block_q8_0 * V, const half * mask,
        int n_kv, int splits, float scale,
        int64_t q_s1, int64_t q_s2, int64_t k_s1, int64_t k_s2, int64_t v_s1, int64_t v_s2,
        int64_t mask_s1, __nv_bfloat16 * pacc, float * pm, float * pl) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
    __shared__ __align__(16) half q_s[BR * SMEM_STRIDE];
    __shared__ __align__(16) half k_s[BC * SMEM_STRIDE];
    __shared__ __align__(16) half v_s[BC * SMEM_STRIDE];
    __shared__ half m_s[T * BC];

    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    // Grid z keeps the shared Q tile at BR=32 for the 48-row G12/T4 case.
    const int row_base = blockIdx.z * BR;
    const int row_total = T * G;
    const int row_count = min(BR, row_total - row_base);
    if (row_count <= 0) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int dim_warp = warp;
    const int logical_tiles = (n_kv + BC - 1) / BC;
    const int tiles_per_split = (logical_tiles + splits - 1) / splits;
    const int first_tile = split * tiles_per_split * BC;
    const int split_end = min(n_kv, first_tile + tiles_per_split * BC);
    if (first_tile >= split_end) {
        return;
    }
    const int key_blocks = (split_end - first_tile + BC - 1) / BC;
    const int tok_begin = row_base / G;
    const int tok_end = (row_base + row_count - 1) / G + 1;

    for (int row = warp; row < BR; row += 4) {
        float vals[8] = {};
        const int global_row = row_base + row;
        if (row < row_count) {
            const int qh = row_qh<G>(global_row, kvh);
            const int tok = row_tok<G>(global_row);
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                vals[p] = q[lane + 32 * p + q_s1 * tok + q_s2 * qh];
            }
        }
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            q_s[row * SMEM_STRIDE + lane + 32 * p] = __float2half(vals[p]);
        }
    }

    float acc[D_SLICE / 8][8] = {};
    float mlo = -CUDART_INF_F;
    float mhi = -CUDART_INF_F;
    float llo = 0.0f;
    float lhi = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * BC;

        for (int chunk = tid; chunk < BC * (D / 8); chunk += THREADS) {
            const int key_l = chunk / (D / 8);
            const int d = (chunk - key_l * (D / 8)) * 8;
            const int key = k0 + key_l;
            half * kd = &k_s[key_l * SMEM_STRIDE + d];
            half * vd = &v_s[key_l * SMEM_STRIDE + d];
            if (key < split_end) {
                const int bo = d / 32;
                const int off = d & 31;
                *reinterpret_cast<int4 *>(kd) = deq8(K[key * k_s1 + kvh * k_s2 + bo], off);
                *reinterpret_cast<int4 *>(vd) = deq8(V[key * v_s1 + kvh * v_s2 + bo], off);
            } else {
                const int4 zero = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4 *>(kd) = zero;
                *reinterpret_cast<int4 *>(vd) = zero;
            }
        }

        for (int i = tok_begin * BC + tid; i < tok_end * BC; i += THREADS) {
            const int tok = i / BC;
            const int col = i % BC;
            const int key = k0 + col;
            m_s[i] = key < n_kv ? mask[key + mask_s1 * tok] : __float2half(-CUDART_INF_F);
        }
        // This first barrier also publishes q_s; the trailing barrier protects K/V before reuse.
        __syncthreads();

#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            const int sub_k0 = k0 + sub * 8;
            float ds[8] = {};
#pragma unroll
            for (int c = 0; c < D / 8; ++c) {
                half2 qf[4];
                half2 kf[4];
                load_qp(qf, reinterpret_cast<const half2 *>(&q_s[c * 8]), SMEM_STRIDE / 2);
                load_k(kf, reinterpret_cast<const half2 *>(&k_s[sub * 8 * SMEM_STRIDE + c * 8]), SMEM_STRIDE / 2);
                mma_qk(ds, qf, kf);
            }

#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int row = d_get_i(l);
                const int col = d_get_j(l);
                const int key = sub_k0 + col;
                if (row < row_count && key < split_end) {
                    const int global_row = row_base + row;
                    const int tok = row_tok<G>(global_row);
                    ds[l] = ds[l] * scale + __half2float(m_s[tok * BC + sub * 8 + col]);
                } else {
                    ds[l] = -CUDART_INF_F;
                }
            }

            float bmlo = fmaxf(fmaxf(ds[0], ds[1]), fmaxf(ds[4], ds[5]));
            float bmhi = fmaxf(fmaxf(ds[2], ds[3]), fmaxf(ds[6], ds[7]));
            bmlo = fmaxf(bmlo, __shfl_xor_sync(0xffffffffu, bmlo, 2));
            bmhi = fmaxf(bmhi, __shfl_xor_sync(0xffffffffu, bmhi, 2));
            const float nmlo = fmaxf(mlo, bmlo);
            const float nmhi = fmaxf(mhi, bmhi);
            const float alo = mlo == -CUDART_INF_F ? 0.0f : exp2f((mlo - nmlo) * LOG2E);
            const float ahi = mhi == -CUDART_INF_F ? 0.0f : exp2f((mhi - nmhi) * LOG2E);
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float nm = (l & 2) ? nmhi : nmlo;
                ds[l] = nm > -CUDART_INF_F && ds[l] > -CUDART_INF_F ? exp2f((ds[l] - nm) * LOG2E) : 0.0f;
            }
            float bllo = ds[0] + ds[1] + ds[4] + ds[5];
            float blhi = ds[2] + ds[3] + ds[6] + ds[7];
            bllo += __shfl_xor_sync(0xffffffffu, bllo, 2);
            blhi += __shfl_xor_sync(0xffffffffu, blhi, 2);
            llo = llo * alo + bllo;
            lhi = lhi * ahi + blhi;
            mlo = nmlo;
            mhi = nmhi;

            const float alpha = lane & 2 ? ahi : alo;
            half2 pp[4];
            softmax_to_half2(pp, ds);
#pragma unroll
            for (int c = 0; c < D_SLICE / 8; ++c) {
                half2 vf[4];
                load_v(vf, reinterpret_cast<const half2 *>(&v_s[sub * 8 * SMEM_STRIDE + dim_warp * D_SLICE + c * 8]), SMEM_STRIDE / 2);
                half2 pv[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
                mma_pv(pv, pp, vf);
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const float2 f = __half22float2(pv[n]);
                    acc[c][2 * n] = acc[c][2 * n] * alpha + f.x;
                    acc[c][2 * n + 1] = acc[c][2 * n + 1] * alpha + f.y;
                }
            }
        }
        __syncthreads();
    }

    const int row = lane;
    const float ownm = lane & 2 ? mhi : mlo;
    const float ownl = lane & 2 ? lhi : llo;
    if (dim_warp == 0 && row < row_count) {
        const int global_row = row_base + row;
        const int qh = row_qh<G>(global_row, kvh);
        const int tok = row_tok<G>(global_row);
        pm[pstat_idx<QH, T>(qh, tok, split)] = ownm;
        pl[pstat_idx<QH, T>(qh, tok, split)] = ownl;
    }
    if (row < row_count) {
        const int global_row = row_base + row;
        const int qh = row_qh<G>(global_row, kvh);
        const int tok = row_tok<G>(global_row);
#pragma unroll
        for (int c = 0; c < D_SLICE / 8; ++c) {
            const int d = dim_warp * D_SLICE + c * 8;
            __nv_bfloat16 out[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                out[i] = __float2bfloat16(acc[c][i]);
            }
            *reinterpret_cast<int4 *>(&pacc[pacc_idx<QH, T>(qh, d, tok, split)]) = *reinterpret_cast<int4 *>(out);
        }
    }
#endif
}

template<int QH, int T>
static __global__ void reduce_kernel(const __nv_bfloat16 * pacc, const float * pm, const float * pl,
        int splits, float * out, int64_t o_s1, int64_t o_s2) {
    const int qh = blockIdx.x;
    const int tok = blockIdx.y;
    const int d = threadIdx.x;
    __shared__ float red[D];

    float local_m = -CUDART_INF_F;
    for (int s = d; s < splits; s += D) {
        local_m = fmaxf(local_m, pm[pstat_idx<QH, T>(qh, tok, s)]);
    }
    red[d] = local_m;
    __syncthreads();
    for (int step = D / 2; step; step >>= 1) {
        if (d < step) {
            red[d] = fmaxf(red[d], red[d + step]);
        }
        __syncthreads();
    }
    const float high_m = red[0];
    if (high_m == -CUDART_INF_F) {
        out[d + o_s1 * qh + o_s2 * tok] = 0.0f;
        return;
    }

    float local_l = 0.0f;
    for (int s = d; s < splits; s += D) {
        const float l = pl[pstat_idx<QH, T>(qh, tok, s)];
        if (l > 0.0f) {
            local_l += l * expf(pm[pstat_idx<QH, T>(qh, tok, s)] - high_m);
        }
    }
    red[d] = local_l;
    __syncthreads();
    for (int step = D / 2; step; step >>= 1) {
        if (d < step) {
            red[d] += red[d + step];
        }
        __syncthreads();
    }
    const float high_l = red[0];

    float num = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float l = pl[pstat_idx<QH, T>(qh, tok, s)];
        if (l > 0.0f) {
            num += __bfloat162float(pacc[pacc_idx<QH, T>(qh, d, tok, s)]) * expf(pm[pstat_idx<QH, T>(qh, tok, s)] - high_m);
        }
    }
    out[d + o_s1 * qh + o_s2 * tok] = high_l > 0.0f ? num / high_l : 0.0f;
}

template<int QH, int KVH, int G, int T>
static inline size_t get_alloc_size_impl(const ggml_tensor * dst) {
    return get_extra<QH, KVH, G, T>(dst).end - (uintptr_t) dst->data;
}

template<int QH, int KVH, int G, int T>
static inline void launch_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];
    const extra_data<QH, KVH, G, T> e = get_extra<QH, KVH, G, T>(dst);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(scale));
    const int64_t q_s1 = Q->nb[1] / sizeof(float);
    const int64_t q_s2 = Q->nb[2] / sizeof(float);
    const int64_t k_s1 = K->nb[1] / sizeof(block_q8_0);
    const int64_t k_s2 = K->nb[2] / sizeof(block_q8_0);
    const int64_t v_s1 = V->nb[1] / sizeof(block_q8_0);
    const int64_t v_s2 = V->nb[2] / sizeof(block_q8_0);
    const int64_t mask_s1 = M->nb[1] / sizeof(half);
    const int64_t o_s1 = dst->nb[1] / sizeof(float);
    const int64_t o_s2 = dst->nb[2] / sizeof(float);
    const cudaStream_t stream = ctx.stream();
    const dim3 partial_blocks(KVH, e.splits, (T * G + BR - 1) / BR);
    const dim3 partial_threads(THREADS, 1, 1);
    const ggml_cuda_kernel_launch_params partial_params(partial_blocks, partial_threads, 0, stream);
    ggml_cuda_kernel_launch(partial_kernel<QH, KVH, G, T>, partial_params,
        (const float *) Q->data, (const block_q8_0 *) K->data, (const block_q8_0 *) V->data,
        (const half *) M->data, (int) K->ne[1], e.splits, scale,
        q_s1, q_s2, k_s1, k_s2, v_s1, v_s2, mask_s1,
        (__nv_bfloat16 *) e.pacc, (float *) e.pm, (float *) e.pl);

    const dim3 reduce_blocks(QH, T, 1);
    const dim3 reduce_threads(D, 1, 1);
    const ggml_cuda_kernel_launch_params reduce_params(reduce_blocks, reduce_threads, 0, stream);
    ggml_cuda_kernel_launch(reduce_kernel<QH, T>, reduce_params,
        (const __nv_bfloat16 *) e.pacc, (const float *) e.pm, (const float *) e.pl,
        e.splits, (float *) dst->data, o_s1, o_s2);
}

static inline variant get_variant(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];

    if (Q == nullptr || K == nullptr || V == nullptr || M == nullptr || dst->src[4] != nullptr || dst->src[5] != nullptr ||
            dst->type != GGML_TYPE_F32 || Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 ||
            V->type != GGML_TYPE_Q8_0 || M->type != GGML_TYPE_F16 || dst->ne[0] != D ||
            dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1] || dst->ne[3] != 1 ||
            Q->ne[0] != D || K->ne[0] != D || V->ne[0] != D ||
            (Q->ne[1] != 1 && Q->ne[1] != 4 && Q->ne[1] != 32) ||
            K->ne[1] <= 0 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || M->ne[3] != 1 ||
            K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2] || M->ne[0] != K->ne[1] ||
            M->ne[1] < Q->ne[1] || M->ne[2] != 1 || dst->nb[0] != sizeof(float) ||
            Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(block_q8_0) || V->nb[0] != sizeof(block_q8_0) ||
            M->nb[0] != sizeof(half)) {
        return variant::none;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return variant::none;
    }

    if (Q->ne[2] == 12 && K->ne[2] == 1 && Q->ne[1] == 1) {
        return variant::qh12_kvh1_t1;
    }
    if (Q->ne[2] == 12 && K->ne[2] == 1 && Q->ne[1] == 4) {
        return variant::qh12_kvh1_t4;
    }
    if (Q->ne[2] == 12 && K->ne[2] == 1 && Q->ne[1] == 32) {
        return variant::qh12_kvh1_t32;
    }
    if (Q->ne[2] == 24 && K->ne[2] == 2 && Q->ne[1] == 1) {
        return variant::qh24_kvh2_t1;
    }
    if (Q->ne[2] == 24 && K->ne[2] == 2 && Q->ne[1] == 4) {
        return variant::qh24_kvh2_t4;
    }
    if (Q->ne[2] == 24 && K->ne[2] == 2 && Q->ne[1] == 32) {
        return variant::qh24_kvh2_t32;
    }
    return variant::none;
}

static inline size_t get_alloc_size(const ggml_tensor * dst) {
    switch (get_variant(dst)) {
        case variant::qh12_kvh1_t1:
            return get_alloc_size_impl<12, 1, 12, 1>(dst);
        case variant::qh12_kvh1_t4:
            return get_alloc_size_impl<12, 1, 12, 4>(dst);
        case variant::qh12_kvh1_t32:
            return get_alloc_size_impl<12, 1, 12, 32>(dst);
        case variant::qh24_kvh2_t1:
            return get_alloc_size_impl<24, 2, 12, 1>(dst);
        case variant::qh24_kvh2_t4:
            return get_alloc_size_impl<24, 2, 12, 4>(dst);
        case variant::qh24_kvh2_t32:
            return get_alloc_size_impl<24, 2, 12, 32>(dst);
        case variant::none:
            break;
    }
    GGML_ABORT("invalid Volta q8 head geometry");
    return 0;
}

static inline void launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (get_variant(dst)) {
        case variant::qh12_kvh1_t1:
            launch_impl<12, 1, 12, 1>(ctx, dst);
            return;
        case variant::qh12_kvh1_t4:
            launch_impl<12, 1, 12, 4>(ctx, dst);
            return;
        case variant::qh12_kvh1_t32:
            launch_impl<12, 1, 12, 32>(ctx, dst);
            return;
        case variant::qh24_kvh2_t1:
            launch_impl<24, 2, 12, 1>(ctx, dst);
            return;
        case variant::qh24_kvh2_t4:
            launch_impl<24, 2, 12, 4>(ctx, dst);
            return;
        case variant::qh24_kvh2_t32:
            launch_impl<24, 2, 12, 32>(ctx, dst);
            return;
        case variant::none:
            break;
    }
    GGML_ABORT("invalid Volta q8 head geometry");
}

} // namespace ggml_q8v_head
