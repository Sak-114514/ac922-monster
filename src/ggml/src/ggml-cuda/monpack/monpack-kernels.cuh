// monpack-kernels.cuh — P3 monpack v2 CUDA device kernels (V100 / sm_70 target)
//
// Scope (see monpack/monpack_format.h for the packing layout, the single authority):
//   1. smem LUT staging (raw-int view for the GEMV dp4a path, fp16-expanded view for the TC path)
//   2. monpack packed-layout addressing helpers (constitution formulas, verbatim)
//   3. per-type packed vec_dot ports — algebra ported 1:1 from vecdotq.cuh (dp4a integer paths);
//      only the weight byte addressing changed from the original GGUF block layout to the packed
//      (d-domain / code-domain) tile layout
//   4. monpack_gemv — M<=8 quantized GEMV, warp/lane<->k mapping and fp32 accumulation order
//      replicated exactly from mmvq.cu (dense mul_mat_vec_q + ids MoE mul_mat_vec_q_moe + the
//      Volta IQ2_S MoE kernel), targeting bit-identical results vs the stock kernels
//   5. monpack_tc_grouped — M>=16 grouped tensor-core kernel on the mma.cuh m8n8k4 primitives
//      (tile<32,4,half2> / tile<8,4,half2,I_MAJOR_MIRRORED> -> tile<32,8,float>),
//      persistent 80 CTA x 256 thread launch, device-side work list built by a prelude kernel,
//      fp32 accumulation, zero host-sync / zero per-call allocation (CUDA graph compatible)
//
// This file adds new code only; it must not require changes to any existing tree file.

#pragma once

#include "../common.cuh"
#include "../vecdotq.cuh"   // get_int_b2/get_int_b4/get_int_from_table_16/unpack_ksigns + VDR_*_MMVQ tables
#include "../mmvq.cuh"      // MMVQ_MAX_BATCH_SIZE
#include "../mma.cuh"       // fork tile library (Volta m8n8k4 primitives)

#include "monpack_format.h"

#include <cstdint>

namespace monpack {

// =================================================================================================
// ---- compile-time type table (must match the monpack_format.h type table; host re-checks) --------
// =================================================================================================

template <ggml_type T> struct mp_type_traits;
template <> struct mp_type_traits<GGML_TYPE_IQ4_NL>  { static constexpr int qk = 32;  static constexpr int cb = 18 - 2;  static constexpr int fam = MONPACK_FAM_A; };
template <> struct mp_type_traits<GGML_TYPE_Q8_0>   { static constexpr int qk = 32;  static constexpr int cb = 34 - 2;  static constexpr int fam = MONPACK_FAM_C; };
template <> struct mp_type_traits<GGML_TYPE_Q2_0>   { static constexpr int qk = 64;  static constexpr int cb = 18 - 2;  static constexpr int fam = MONPACK_FAM_C; };
template <> struct mp_type_traits<GGML_TYPE_IQ2_XXS>{ static constexpr int qk = 256; static constexpr int cb = 66 - 2;  static constexpr int fam = MONPACK_FAM_B; };
template <> struct mp_type_traits<GGML_TYPE_IQ2_XS> { static constexpr int qk = 256; static constexpr int cb = 74 - 2;  static constexpr int fam = MONPACK_FAM_B; };
template <> struct mp_type_traits<GGML_TYPE_IQ2_S>  { static constexpr int qk = 256; static constexpr int cb = 82 - 2;  static constexpr int fam = MONPACK_FAM_B; };
template <> struct mp_type_traits<GGML_TYPE_IQ3_XXS>{ static constexpr int qk = 256; static constexpr int cb = 98 - 2;  static constexpr int fam = MONPACK_FAM_B; };
template <> struct mp_type_traits<GGML_TYPE_IQ3_S>  { static constexpr int qk = 256; static constexpr int cb = 110 - 2; static constexpr int fam = MONPACK_FAM_B; };

static constexpr __host__ __device__ bool mp_type_supported(const ggml_type t) {
    switch (t) {
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_0:
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

// VDR for the MMVQ lane<->k mapping (mmvq.cu get_vdr_mmvq table, restated here because the
// original is a static in mmvq.cu). iq4_nl uses the Q4_0 vdr exactly like the fork does.
static constexpr __host__ __device__ int mp_get_vdr(const ggml_type t) {
    switch (t) {
        case GGML_TYPE_IQ4_NL:   return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:     return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:     return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS:  return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:   return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:    return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS:  return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:    return VDR_IQ3_S_Q8_1_MMVQ;
        default:                 return 1;
    }
}

// =================================================================================================
// ---- packed-layout addressing (monpack_format.h layout_rev=1, verbatim) -------------------------
// =================================================================================================

struct mp_view_t {
    const char * base;      // tensor start inside the blob (== registered tensor_data == blob_base + blob_off)
    uint32_t     n_tiles_n; // ceil(ne_n / 64)
    uint32_t     n_tiles_k; // ne_k / k_tile
    uint32_t     nb;        // k_tile / block_size (blocks per row per tile)
    uint32_t     code_bytes;// block_bytes - 2
    uint64_t     tile_bytes;// nb * 64 * (2 + code_bytes)
};

// constitution:
//   tile_off(e, n0, k0) = blob_off + ((e * n_tiles_n + n0/64) * n_tiles_k + k0/k_tile) * tile_bytes
//   d addr    = tile_off + (row * nb + kb) * 2
//   code addr = tile_off + nb*64*2 + (row * nb + kb) * code_bytes
// with row/kb relative to the tile (row = n % 64, kb = k-block index within the tile).
// `kb_global` below is the k-block index within the whole row ((k0 + kb*block_size)/block_size),
// so the tile k-index is kb_global / nb and the in-tile block is kb_global % nb.
static __device__ __forceinline__ void mp_block(
        const mp_view_t & v, const uint32_t e, const uint32_t row, const uint32_t kb_global,
        const char *& d_out, const char *& code_out) {
    const uint64_t tile_off = ((uint64_t)(e * v.n_tiles_n + (row >> 6)) * v.n_tiles_k + (kb_global / v.nb)) * v.tile_bytes;
    const uint32_t idx      = (row & 63) * v.nb + (kb_global % v.nb);
    d_out   = v.base + tile_off + (uint64_t)idx * 2;
    code_out = v.base + tile_off + (uint64_t)(v.nb * 64 * 2) + (uint64_t)idx * v.code_bytes;
}

static __device__ __forceinline__ float mp_d_to_float(const char * d_ptr) {
    return __half2float(*reinterpret_cast<const ggml_half *>(d_ptr));
}

// =================================================================================================
// ---- GEMV LUT staging: raw integer grids in smem ------------------------------------------------
// ---- 2-word (uint64) grid entries are padded to a 3-word stride (odd stride in 32-bit words is  --
// ---- coprime with the 32 banks => scattered per-lane lookups stay conflict-free); 1-word iq3    --
// ---- grids keep their natural stride-1 layout. Values are byte-identical to the __device__      --
// ---- tables in ggml-common.h, so the dp4a algebra and its results are unchanged.                --
// =================================================================================================

struct mp_gemv_lut_empty { int pad_; __device__ void stage(int, int) {} };

struct mp_gemv_lut_iq2xxs {                     // 256 entries x 3 words
    static constexpr int N = 256, W = 3;
    uint32_t g[N*W];
    __device__ void stage(const int tid, const int nthr) {
        for (int i = tid; i < N; i += nthr) {
            const uint2 e = ((const uint2 *) iq2xxs_grid)[i];
            g[3*i+0] = e.x; g[3*i+1] = e.y; g[3*i+2] = 0;
        }
    }
    __device__ __forceinline__ uint2 get(const int idx) const {
        return make_uint2(g[3*idx+0], g[3*idx+1]);
    }
};

struct mp_gemv_lut_iq2xs {                      // 512 entries x 3 words
    static constexpr int N = 512, W = 3;
    uint32_t g[N*W];
    __device__ void stage(const int tid, const int nthr) {
        for (int i = tid; i < N; i += nthr) {
            const uint2 e = ((const uint2 *) iq2xs_grid)[i];
            g[3*i+0] = e.x; g[3*i+1] = e.y; g[3*i+2] = 0;
        }
    }
    __device__ __forceinline__ uint2 get(const int idx) const {
        return make_uint2(g[3*idx+0], g[3*idx+1]);
    }
};

struct mp_gemv_lut_iq2s {                       // 1024 entries x 3 words
    static constexpr int N = 1024, W = 3;
    uint32_t g[N*W];
    __device__ void stage(const int tid, const int nthr) {
        for (int i = tid; i < N; i += nthr) {
            const uint2 e = ((const uint2 *) iq2s_grid)[i];
            g[3*i+0] = e.x; g[3*i+1] = e.y; g[3*i+2] = 0;
        }
    }
    __device__ __forceinline__ uint2 get(const int idx) const {
        return make_uint2(g[3*idx+0], g[3*idx+1]);
    }
};

struct mp_gemv_lut_iq3xxs {                     // 256 entries x 1 word (natural stride)
    static constexpr int N = 256, W = 1;
    uint32_t g[N];
    __device__ void stage(const int tid, const int nthr) {
        for (int i = tid; i < N; i += nthr) {
            g[i] = iq3xxs_grid[i];
        }
    }
    __device__ __forceinline__ uint32_t get(const int idx) const {
        return g[idx];
    }
};

struct mp_gemv_lut_iq3s {                       // 512 entries x 1 word
    static constexpr int N = 512, W = 1;
    uint32_t g[N];
    __device__ void stage(const int tid, const int nthr) {
        for (int i = tid; i < N; i += nthr) {
            g[i] = iq3s_grid[i];
        }
    }
    __device__ __forceinline__ uint32_t get(const int idx) const {
        return g[idx];
    }
};

template <ggml_type T> struct mp_gemv_lut_type;
template <> struct mp_gemv_lut_type<GGML_TYPE_IQ4_NL>  { using type = mp_gemv_lut_empty; };
template <> struct mp_gemv_lut_type<GGML_TYPE_Q8_0>   { using type = mp_gemv_lut_empty; };
template <> struct mp_gemv_lut_type<GGML_TYPE_Q2_0>   { using type = mp_gemv_lut_empty; };
template <> struct mp_gemv_lut_type<GGML_TYPE_IQ2_XXS>{ using type = mp_gemv_lut_iq2xxs; };
template <> struct mp_gemv_lut_type<GGML_TYPE_IQ2_XS> { using type = mp_gemv_lut_iq2xs; };
template <> struct mp_gemv_lut_type<GGML_TYPE_IQ2_S>  { using type = mp_gemv_lut_iq2s; };
template <> struct mp_gemv_lut_type<GGML_TYPE_IQ3_XXS>{ using type = mp_gemv_lut_iq3xxs; };
template <> struct mp_gemv_lut_type<GGML_TYPE_IQ3_S>  { using type = mp_gemv_lut_iq3s; };

// =================================================================================================
// ---- per-type packed vec_dot ports (GEMV dp4a path) ----------------------------------------------
// ---- Each function mirrors its vecdotq.cuh original line for line; only the weight pointer       --
// ---- derivation changed (mp_block instead of `(const block_x *) vbq + kbx`).                     --
// ---- code-domain field offsets follow the fork block structs minus the leading 2-byte d.         --
// =================================================================================================

// ---- IQ4_NL (source: vecdotq.cuh vec_dot_iq4_nl_q8_1, :1333) ----
// block_iq4_nl: d | uint8_t qs[16]  ->  d-domain 2B, code = qs[16]
static __device__ __forceinline__ float mp_vec_dot_iq4_nl_q8_1(
        const mp_view_t & W, const uint32_t e, const uint32_t row, const uint32_t kbx,
        const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const char * dp; const char * cp;
    mp_block(W, e, row, kbx, dp, cp);

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b2(cp, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = mp_d_to_float(dp) * __low2float(bq8_1->ds);
    return d * sumi;
}

// ---- Q8_0 (source: vecdotq.cuh vec_dot_q8_0_q8_1, :851, + vec_dot_q8_0_q8_1_impl, :246) ----
// block_q8_0: d | int8_t qs[32]  ->  code = qs[32]
static __device__ __forceinline__ float mp_vec_dot_q8_0_q8_1(
        const mp_view_t & W, const uint32_t e, const uint32_t row, const uint32_t kbx,
        const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const char * dp; const char * cp;
    mp_block(W, e, row, kbx, dp, cp);

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_b2(cp, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }

    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

    const float d8_0 = mp_d_to_float(dp);
    const float d8_1 = __low2float(bq8_1->ds);
    return d8_0 * d8_1 * ((float) sumi);
}

// ---- Q2_0, fork QK=64 (source: vecdotq.cuh vec_dot_q2_0_q8_1, :728) ----
// block_q2_0: d | uint8_t qs[16]  ->  code = qs[16]
static __device__ __forceinline__ float mp_vec_dot_q2_0_q8_1(
        const mp_view_t & W, const uint32_t e, const uint32_t row, const uint32_t kbx,
        const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const char * dp; const char * cp;
    mp_block(W, e, row, kbx, dp, cp);

    const float     d2 = mp_d_to_float(dp);
    const int16_t * qs = (const int16_t *) cp + iqs * 4;

    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q  = qs[j];
        const int u  = get_int_b4(bq8_1_chunk->qs, j*2+0);
        const int v  = get_int_b4(bq8_1_chunk->qs, j*2+1);

        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);

        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }

    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

// ---- IQ2_XXS (source: vecdotq.cuh vec_dot_iq2_xxs_q8_1, :1048) ----
// block_iq2_xxs: d | uint16_t qs[32]  ->  code = qs[32] (4B grid indices + 4B signs/scale per 32-weight group)
static __device__ __forceinline__ float mp_vec_dot_iq2_xxs_q8_1(
        const mp_view_t & W, const uint32_t e, const uint32_t row, const uint32_t kbx,
        const mp_gemv_lut_iq2xxs & lut,
        const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const char * dp; const char * cp;
    mp_block(W, e, row, kbx, dp, cp);

    const int q2 = get_int_b2(cp, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(cp, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = lut.get(aux8[k0/2]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
    sumi = sumi * ls / 8;           // (sumi * scale + sumi / 2) / 4
    const float d = mp_d_to_float(dp) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// ---- IQ2_XS (source: vecdotq.cuh vec_dot_iq2_xs_q8_1, :1083) ----
// block_iq2_xs: d | uint16_t qs[32] | uint8_t scales[8]  ->  code: qs[64B], scales at code+64
static __device__ __forceinline__ float mp_vec_dot_iq2_xs_q8_1(
        const mp_view_t & W, const uint32_t e, const uint32_t row, const uint32_t kbx,
        const mp_gemv_lut_iq2xs & lut,
        const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const char * dp; const char * cp;
    mp_block(W, e, row, kbx, dp, cp);

    const int2 q2_packed = make_int2(get_int_b2(cp, iqs + 0), get_int_b2(cp, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = cp[64 + iqs/2] & 0x0F;
    const int ls1 = cp[64 + iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = lut.get(q2[l0/2] & 0x1FF);
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = mp_d_to_float(dp) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// ---- IQ2_S (source: vecdotq.cuh vec_dot_iq2_s_q8_1, :1124) ----
// block_iq2_s: d | uint8_t qs[64] | uint8_t qh[8] | uint8_t scales[8]
//   ->  code: qs[64] (grid bytes 0..31, sign bytes 32..63), qh at code+64, scales at code+72
static __device__ __forceinline__ float mp_vec_dot_iq2_s_q8_1(
        const mp_view_t & W, const uint32_t e, const uint32_t row, const uint32_t kbx,
        const mp_gemv_lut_iq2s & lut,
        const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const char * dp; const char * cp;
    mp_block(W, e, row, kbx, dp, cp);

    const int       qs_packed = get_int_b2(cp, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = cp[64 + iqs/2];

    const int       signs_packed_32 = get_int_b2(cp + 32, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = cp[72 + iqs/2] & 0x0F;
    const int ls1 = cp[72 + iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = lut.get(qs[l0/2] | ((qh << (8-l0)) & 0x300));

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4((int)grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4((int)grid_pos.y ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;

    const float d = mp_d_to_float(dp) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// ---- IQ3_XXS (source: vecdotq.cuh vec_dot_iq3_xxs_q8_1, :1172) ----
// block_iq3_xxs: d | uint8_t qs[96]  ->  code: quant bytes 0..63, sign/scale bytes 64..95
static __device__ __forceinline__ float mp_vec_dot_iq3_xxs_q8_1(
        const mp_view_t & W, const uint32_t e, const uint32_t row, const uint32_t kbx,
        const mp_gemv_lut_iq3xxs & lut,
        const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const char * dp; const char * cp;
    mp_block(W, e, row, kbx, dp, cp);

    const int2 q3_packed = make_int2(get_int_b2(cp, iqs), get_int_b2(cp, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(cp + 64, iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2((int) lut.get(q3[l0 + 0]), (int) lut.get(q3[l0 + 1]));
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = mp_d_to_float(dp) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// ---- IQ3_S (source: vecdotq.cuh vec_dot_iq3_s_q8_1, :1211) ----
// block_iq3_s: d | uint8_t qs[64] | uint8_t qh[8] | uint8_t signs[32] | uint8_t scales[4]
//   ->  code: qs[64], qh at +64, signs at +72, scales at +104
static __device__ __forceinline__ float mp_vec_dot_iq3_s_q8_1(
        const mp_view_t & W, const uint32_t e, const uint32_t row, const uint32_t kbx,
        const mp_gemv_lut_iq3s & lut,
        const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const char * dp; const char * cp;
    mp_block(W, e, row, kbx, dp, cp);

    const int2      qs_packed = make_int2(get_int_b2(cp, iqs + 0), get_int_b2(cp, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = cp[64 + iqs/2];

    const int       signs_packed_32 = get_int_b2(cp + 72, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            (int) lut.get(qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)),
            (int) lut.get(qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)));

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((cp[104 + iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = mp_d_to_float(dp) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

// dispatcher: one call site for the templated GEMV kernels
template <ggml_type type, typename LUT>
static __device__ __forceinline__ float mp_vec_dot(
        const mp_view_t & W, const uint32_t e, const uint32_t row, const uint32_t kbx,
        const LUT & lut,
        const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    if constexpr (type == GGML_TYPE_IQ4_NL) {
        GGML_UNUSED(lut);
        return mp_vec_dot_iq4_nl_q8_1(W, e, row, kbx, bq8_1, iqs);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        GGML_UNUSED(lut);
        return mp_vec_dot_q8_0_q8_1(W, e, row, kbx, bq8_1, iqs);
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        GGML_UNUSED(lut);
        return mp_vec_dot_q2_0_q8_1(W, e, row, kbx, bq8_1, iqs);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return mp_vec_dot_iq2_xxs_q8_1(W, e, row, kbx, lut, bq8_1, iqs);
    } else if constexpr (type == GGML_TYPE_IQ2_XS) {
        return mp_vec_dot_iq2_xs_q8_1(W, e, row, kbx, lut, bq8_1, iqs);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        return mp_vec_dot_iq2_s_q8_1(W, e, row, kbx, lut, bq8_1, iqs);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        return mp_vec_dot_iq3_xxs_q8_1(W, e, row, kbx, lut, bq8_1, iqs);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        return mp_vec_dot_iq3_s_q8_1(W, e, row, kbx, lut, bq8_1, iqs);
    } else {
        NO_DEVICE_CODE;
        return 0.0f;
    }
}

// =================================================================================================
// ---- GEMV kernel 1: dense / ids-single-token (port of mmvq.cu mul_mat_vec_q :649, fusion removed)
// ---- Warp/lane<->k mapping, fp32 accumulation order, smem reduction and epilogue are copied      --
// ---- verbatim; `vx` block addressing is replaced by the packed (e, row, kb) view.                --
// =================================================================================================

// Launch-geometry tables replicated from mmvq.cu (calc_nwarps / calc_rows_per_block,
// MMVQ_PARAMETERS_VOLTA + the GENERIC fallback it defers to). This build targets sm_70 only, so
// table_id == VOLTA and halve_iters is never taken (GB10-only in the fork). The final (nwarps, rpb)
// pair is chosen by the host (mp_choose_gemv_geometry in monpack-kernels.cu, replicating
// should_use_small_k) and passed as template arguments so __launch_bounds__ matches the launch.
static constexpr __host__ __device__ int mp_calc_nwarps_default(const int ncols_dst) {
    switch (ncols_dst) {
        case 1:
        case 2:
        case 3:
        case 4:
            return 4;
        case 5:
        case 6:
        case 7:
        case 8:
            return 2;
        default:
            return 1;
    }
}

// Volta small_k rows_per_block (MMVQ_VOLTA_SK_ROWS=4 / default 2 for ncols>=2, nwarps for ncols==1),
// matching calc_rows_per_block(table_id == VOLTA / GENERIC)
static constexpr __host__ __device__ int mp_calc_rows_per_block(const int ncols_dst, const bool small_k, const int nwarps) {
    if (ncols_dst >= 2 && ncols_dst <= 8) {
        return small_k ? 4 : 2;
    }
    return small_k ? nwarps : 1;
}

template <ggml_type type, int ncols_dst, int c_nwarps, int c_rows_per_block>
__launch_bounds__(c_nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mp_mul_mat_vec_q(
        const mp_view_t W, const uint32_t nchannels_x, const void * vy_ptr, const int32_t * ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = mp_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = mp_get_vdr(type);
    constexpr int nwarps = c_nwarps;
    constexpr int rows_per_cuda_block = c_rows_per_block;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    __shared__ typename mp_gemv_lut_type<type>::type lut;

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    // expert (outermost blob dim) for this block; packed tensors carry it as src0->ne[2] == ne_e
    const uint32_t e = sample_x*nchannels_x + channel_x;

    lut.stage(tid, nwarps*warp_size);
    __syncthreads();

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += mp_vec_dot<type>(
                    W, e, row0 + i, kbx, lut, &y[j*stride_col_y + kby], kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                dst[j*stride_col_dst + i] = tmp[j][i];
            }
        }
    }
}

// =================================================================================================
// ---- GEMV kernel 2: ids MoE multi-token (port of mmvq.cu mul_mat_vec_q_moe :901, fusion removed) --
// =================================================================================================

template <ggml_type type, int c_rows_per_block>
__launch_bounds__(MMVQ_MAX_BATCH_SIZE*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mp_mul_mat_vec_q_moe(
        const mp_view_t W, const void * vy_ptr, const int32_t * ids_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = mp_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = mp_get_vdr(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    __shared__ typename mp_gemv_lut_type<type>::type lut;

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    lut.stage(tid, warp_size);

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const uint32_t e = channel_x; // expert index == src0 channel

    __syncthreads();

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += mp_vec_dot<type>(W, e, row0 + i, kbx, lut, &y[kby], kqs);
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            if (i == threadIdx.x) {
                dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[i];
            }
        }
    }
}

// =================================================================================================
// ---- GEMV kernel 3: Q2_0 SM70 MoE rework — packed replica of kernel_sm70_moe.cuh                 --
// ---- (Bug 2 fix). The deployed baseline engine sets GGML_SM70_MOE_REWORK=1, so for Q2_0 MoE      --
// ---- GEMV with K in {192,256} (the down-projection island slices) and tokens 2..8 the stock      --
// ---- launcher routes to moe_matvec_q2_0_sm70, NOT to mul_mat_vec_q_moe. That kernel decodes      --
// ---- one row per lane and accumulates sequentially (kb outer, 32-element chunk c inner, acc      --
// ---- starting at 0) — a different fp32 summation order than the stock warp reduce, so bit        --
// ---- parity with the baseline requires reproducing it exactly. This port keeps the grid/block    --
// ---- structure, the smem y staging (verbatim window copy) and the per-row accumulation order     --
// ---- 1:1; only the weight bytes come from the packed (d-domain / code-domain) view via           --
// ---- mp_vec_dot_q2_0_q8_1 (whose algebra is line-for-line the fork's vec_dot_q2_0_q8_1).         --
// =================================================================================================

namespace mp_moe_sm70 {

constexpr int WARP_SZ = 32;
constexpr int Q8BLK   = (int) sizeof(block_q8_1);  // 36 = 2*half + 32 qs
constexpr int T_R     = 32;                        // rows per warp (lane -> row decode)
constexpr int NW      = 4;                         // warps per block
constexpr int NT      = NW*WARP_SZ;                // 128 threads
constexpr int ROWS_PER_BLOCK = T_R*NW;             // 128 rows per block

// aligned-window cooperative copy, verbatim from kernel_sm70_moe.cuh moe_stage_window
static __device__ __forceinline__ int stage_window(
        const char * src, const int len, char * dst, const int t, const int nthreads) {
    const uintptr_t a0 = (uintptr_t) src & ~(uintptr_t) 15;
    const int shift = (int)((uintptr_t) src - a0);
    const uintptr_t a1 = ((uintptr_t) src + len + 15) & ~(uintptr_t) 15;
    const int n_u4 = (int)((a1 - a0) >> 4);
    const uint4 * s4 = (const uint4 *) a0;
    uint4 * d4 = (uint4 *) dst;
    for (int i = t; i < n_u4; i += nthreads) d4[i] = s4[i];
    return shift;
}

template <int KB>  // Q2_0 k-blocks per row: 3 (K=192) or 4 (K=256)
__launch_bounds__(NT)
__global__ void mp_moe_matvec_q2_0_sm70(
        const mp_view_t Wv, const void * vy_ptr,
        const int32_t * GGML_CUDA_RESTRICT ids, float * GGML_CUDA_RESTRICT dst,
        const uint32_t ncols_x,             // K (192 or 256)
        const uint32_t nrows_x,             // N rows of the (sliced) tensor
        const uint32_t stride_col_y,        // q8_1 blocks between token rows of y
        const uint32_t stride_col_dst,      // elements between token rows of dst
        const uint32_t stride_channel_y,    // q8_1 blocks between y channels
        const uint32_t stride_channel_dst,  // elements between dst slots
        const uint32_t ids_stride,          // ids elements between token rows
        const uint32_t nchannels_y) {
    // smem layout: y row | per-row outputs (weights are read directly from the packed domains,
    // which are contiguous and coalesced: 2B/entry d domain, 16B/entry code domain)
    extern __shared__ char smem_raw[];
    const int y_bytes = (int)(ncols_x/32) * Q8BLK;
    const int y_pitch = (y_bytes + 31) & ~15;
    char * y_s = smem_raw;
    float * out_s = (float *)(y_s + y_pitch);

    const int tid  = threadIdx.y*WARP_SZ + threadIdx.x;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int ch   = blockIdx.y;   // expert slot (dst channel)
    const int tok  = blockIdx.z;   // token

    // per-(slot,token) expert — the blob's outermost dim
    const int chx = ids[ch + tok*ids_stride];
    const int chy = nchannels_y > 1 ? ch % (int) nchannels_y : 0;

    // ---- block-wide staging: this token's y row (verbatim window copy) ----
    const char * ysrc = (const char *) vy_ptr
        + ((size_t) chy*stride_channel_y + (size_t) tok*stride_col_y) * Q8BLK;
    const int y_shift = stage_window(ysrc, y_bytes, y_s, tid, NT);
    // y staging must complete before any warp decodes (y_s is block-shared) — same barrier
    // placement as the fork kernel
    __syncthreads();
    const block_q8_1 * y0 = (const block_q8_1 *)(y_s + y_shift);

    const int r0 = (int) blockIdx.x*ROWS_PER_BLOCK + warp*T_R;

    // ---- decode: lane -> row, accumulation order copied 1:1 from moe_matvec_q2_0_sm70 ----
    float acc = 0.f;
    if (r0 + lane < (int) nrows_x) {
        const uint32_t row = (uint32_t)(r0 + lane);
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
            const block_q8_1 * yb = y0 + kb*2;  // 2 q8_1 chunks per weight block
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                acc += mp_vec_dot_q2_0_q8_1(Wv, (uint32_t) chx, row, (uint32_t) kb, yb, c);
            }
        }
    }
    out_s[warp*T_R + lane] = acc;  // one warp uniquely owns each row

    __syncthreads();

    // ---- epilogue: dst[ch*stride_channel_dst + tok*stride_col_dst + row] ----
    for (int r = tid; r < ROWS_PER_BLOCK; r += NT) {
        const int row_g = (int) blockIdx.x*ROWS_PER_BLOCK + r;
        if (row_g >= (int) nrows_x) break;
        dst[(size_t) ch*stride_channel_dst + (size_t) tok*stride_col_dst + row_g] = out_s[r];
    }
}

} // namespace mp_moe_sm70

// =================================================================================================
// ---- GEMV kernel 4: Volta IQ2_S MoE special (port of mmvq.cu mul_mat_vec_iq2_s_moe_volta :1133,  --
// ---- gated/q8-out side product dropped — the plain no-fusion calls only). Grid/sign handling     --
// ---- keeps the fork's own reduction order so outputs stay bit-identical to the stock Volta path. --
// =================================================================================================

static __device__ __forceinline__ float mp_vec_dot_iq2_s_group_volta(
        const mp_view_t & W, const uint32_t e, const uint32_t row, const int sb, const int ib,
        const block_q8_1 * GGML_CUDA_RESTRICT yb, const uint2 * GGML_CUDA_RESTRICT grid) {
    const char * dp; const char * cp;
    mp_block(W, e, row, sb, dp, cp);

    const int qs    = get_int_b2(cp, ib);
    const int signs = get_int_b2(cp + 32, ib);
    const int qh    = cp[64 + ib];
    const int sc    = cp[72 + ib];
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint2 g = grid[((qs >> (8*l)) & 0xFF) | ((qh << (8-2*l)) & 0x300)];
        const int s   = (signs >> (8*l)) & 0xFF;
        // spread 4 sign bits to the low bit of each byte; grid bytes are never 0, so (g ^ 0xFF) + 1 = -g
        // never carries into the next byte
        const int sp0 = ((s & 0xF) * 0x00204081) & 0x01010101;
        const int sp1 = ((s >> 4)  * 0x00204081) & 0x01010101;
        const int gl  = (int(g.x) ^ (sp0*0xFF)) + sp0;
        const int gh  = (int(g.y) ^ (sp1*0xFF)) + sp1;
        const int u0  = get_int_b4(yb->qs, 2*l + 0);
        const int u1  = get_int_b4(yb->qs, 2*l + 1);
        if (l < 2) {
            sumi0 = ggml_cuda_dp4a(gl, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(gh, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(gl, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(gh, u1, sumi1);
        }
    }
    const int sumi = (sumi0*(sc & 0x0F) + sumi1*(sc >> 4) + (sumi0 + sumi1)/2)/4;
    return mp_d_to_float(dp) * __low2float(yb->ds) * sumi;
}

#define MP_IQ2S_VOLTA_ROWS   4
#define MP_IQ2S_VOLTA_NWARPS 8
#define MP_IQ2S_VOLTA_MAX_K  8192

__launch_bounds__(MP_IQ2S_VOLTA_NWARPS*WARP_SIZE, 1)
static __global__ void mp_mul_mat_iq2_s_moe_volta(
        const mp_view_t W, const void * vy_ptr, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ids_stride) {
    constexpr int R  = MP_IQ2S_VOLTA_ROWS;
    constexpr int NW = MP_IQ2S_VOLTA_NWARPS;
    __shared__ uint2 grid_s[1024];

    const int tid         = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int channel_dst = blockIdx.y;
    const int token       = blockIdx.z;
    const int nsb         = ncols_x / QK_K;
    const int ng          = nsb * (QK_K/32);

    for (int i = tid; i < 1024; i += NW*WARP_SIZE) {
        grid_s[i] = ((const uint2 *) iq2s_grid)[i];
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[channel_dst + token*ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const int row0 = (blockIdx.x*NW + threadIdx.y)*R;
    if (row0 >= (int) nrows_x) {
        return;
    }

    __syncthreads();

    const block_q8_1 * y = ((const block_q8_1 *) vy_ptr) + channel_y*stride_channel_y + token*stride_col_y;

    float sum[R] = {0.0f};

    const int nrows_tile = min(R, (int) nrows_x - row0);
    const int n = nrows_tile*ng;
    for (int c = threadIdx.x; c < n; c += WARP_SIZE) {
        const int r  = c / ng;
        const int g  = c - r*ng;
        const int sb = g / (QK_K/32);
        const int ib = g % (QK_K/32);
        const float v = mp_vec_dot_iq2_s_group_volta(W, channel_x, row0 + r, sb, ib, y + g, grid_s);
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if (i == r) {
                sum[i] += v;
            }
        }
    }

    ggml_cuda_pdl_lc();

#pragma unroll
    for (int i = 0; i < R; ++i) {
        const float result = warp_reduce_sum<WARP_SIZE>(sum[i]);
        if (threadIdx.x == 0 && i < nrows_tile) {
            dst[channel_dst*stride_channel_dst + token*stride_col_dst + row0 + i] = result;
        }
    }
}

// =================================================================================================
// ---- TC LUT staging: fp16-expanded grids for dequant_block_packed --------------------------------
// ---- grid rows padded to 10 halves (8-value grids) / 6 halves (4-value grids): 5 / 3 32-bit      --
// ---- words row stride, both coprime with 32 banks => per-lane scattered dequant lookups stay    --
// ---- conflict-free. ksigns_iq2xs/kmask_iq2xs staged as bytes (byte-granule scattered accesses   --
// ---- already spread across banks).                                                              --
// =================================================================================================

struct mp_tc_lut_iq2xxs {
    static constexpr int N = 256, V = 8, W = 10; // 8 values padded to 10 halves
    half    g[N][W];
    uint8_t ksigns[128];
    uint8_t kmask[8];
    __device__ void stage(const int tid, const int nthr) {
        for (int i = tid; i < N; i += nthr) {
            const uint64_t e = iq2xxs_grid[i];
            const uint8_t * b = (const uint8_t *) &e;
#pragma unroll
            for (int j = 0; j < V; ++j) g[i][j] = __float2half((float) b[j]);
            g[i][V+0] = (half) 0.f;
            g[i][V+1] = (half) 0.f;
        }
        for (int i = tid; i < 128; i += nthr) ksigns[i] = ksigns_iq2xs[i];
        for (int i = tid; i < 8;   i += nthr) kmask[i]  = kmask_iq2xs[i];
    }
};

struct mp_tc_lut_iq2xs {
    static constexpr int N = 512, V = 8, W = 10;
    half    g[N][W];
    uint8_t ksigns[128];
    uint8_t kmask[8];
    __device__ void stage(const int tid, const int nthr) {
        for (int i = tid; i < N; i += nthr) {
            const uint64_t e = iq2xs_grid[i];
            const uint8_t * b = (const uint8_t *) &e;
#pragma unroll
            for (int j = 0; j < V; ++j) g[i][j] = __float2half((float) b[j]);
            g[i][V+0] = (half) 0.f;
            g[i][V+1] = (half) 0.f;
        }
        for (int i = tid; i < 128; i += nthr) ksigns[i] = ksigns_iq2xs[i];
        for (int i = tid; i < 8;   i += nthr) kmask[i]  = kmask_iq2xs[i];
    }
};

struct mp_tc_lut_iq2s {
    static constexpr int N = 1024, V = 8, W = 10;
    half    g[N][W];                 // 20 KB
    uint8_t kmask[8];                // iq2_s signs live in the code domain
    __device__ void stage(const int tid, const int nthr) {
        for (int i = tid; i < N; i += nthr) {
            const uint64_t e = iq2s_grid[i];
            const uint8_t * b = (const uint8_t *) &e;
#pragma unroll
            for (int j = 0; j < V; ++j) g[i][j] = __float2half((float) b[j]);
            g[i][V+0] = (half) 0.f;
            g[i][V+1] = (half) 0.f;
        }
        for (int i = tid; i < 8; i += nthr) kmask[i] = kmask_iq2xs[i];
    }
};

struct mp_tc_lut_iq3xxs {
    static constexpr int N = 256, V = 4, W = 6;  // 4 values padded to 6 halves
    half    g[N][W];
    uint8_t ksigns[128];
    uint8_t kmask[8];
    __device__ void stage(const int tid, const int nthr) {
        for (int i = tid; i < N; i += nthr) {
            const uint32_t e = iq3xxs_grid[i];
            const uint8_t * b = (const uint8_t *) &e;
#pragma unroll
            for (int j = 0; j < V; ++j) g[i][j] = __float2half((float) b[j]);
            g[i][V+0] = (half) 0.f;
            g[i][V+1] = (half) 0.f;
        }
        for (int i = tid; i < 128; i += nthr) ksigns[i] = ksigns_iq2xs[i];
        for (int i = tid; i < 8;   i += nthr) kmask[i]  = kmask_iq2xs[i];
    }
};

struct mp_tc_lut_iq3s {
    static constexpr int N = 512, V = 4, W = 6;
    half    g[N][W];
    uint8_t kmask[8];                // iq3_s signs live in the code domain
    __device__ void stage(const int tid, const int nthr) {
        for (int i = tid; i < N; i += nthr) {
            const uint32_t e = iq3s_grid[i];
            const uint8_t * b = (const uint8_t *) &e;
#pragma unroll
            for (int j = 0; j < V; ++j) g[i][j] = __float2half((float) b[j]);
            g[i][V+0] = (half) 0.f;
            g[i][V+1] = (half) 0.f;
        }
        for (int i = tid; i < 8; i += nthr) kmask[i] = kmask_iq2xs[i];
    }
};

struct mp_tc_lut_iq4nl {
    static constexpr int N = 16, V = 1, W = 4;   // 16 nibble values, padded row
    half g[N][W];
    __device__ void stage(const int tid, const int nthr) {
        for (int i = tid; i < N; i += nthr) {
            g[i][0] = __float2half((float) kvalues_iq4nl[i]);
            g[i][1] = (half) 0.f;
            g[i][2] = (half) 0.f;
            g[i][3] = (half) 0.f;
        }
    }
};

struct mp_tc_lut_none { int pad_; __device__ void stage(int, int) {} };

template <ggml_type T> struct mp_tc_lut_type;
template <> struct mp_tc_lut_type<GGML_TYPE_IQ4_NL>  { using type = mp_tc_lut_iq4nl; };
template <> struct mp_tc_lut_type<GGML_TYPE_Q8_0>   { using type = mp_tc_lut_none; };
template <> struct mp_tc_lut_type<GGML_TYPE_Q2_0>   { using type = mp_tc_lut_none; };
template <> struct mp_tc_lut_type<GGML_TYPE_IQ2_XXS>{ using type = mp_tc_lut_iq2xxs; };
template <> struct mp_tc_lut_type<GGML_TYPE_IQ2_XS> { using type = mp_tc_lut_iq2xs; };
template <> struct mp_tc_lut_type<GGML_TYPE_IQ2_S>  { using type = mp_tc_lut_iq2s; };
template <> struct mp_tc_lut_type<GGML_TYPE_IQ3_XXS>{ using type = mp_tc_lut_iq3xxs; };
template <> struct mp_tc_lut_type<GGML_TYPE_IQ3_S>  { using type = mp_tc_lut_iq3s; };

// =================================================================================================
// ---- dequant_block_packed<TYPE>: 16 consecutive fp16 weights from one packed block ---------------
// ---- `kb_off` = k offset inside the block (multiple of 16 — guaranteed by the J=16 k stepping    --
// ---- because 16 divides every block_size in the table, so the 16 weights never straddle blocks). --
// ---- Algebra follows the CPU truth in ggml-quants.c dequantize_row_* (see per-type comments).    --
// =================================================================================================

template <ggml_type T> struct mp_type_tag {};   // overload tag: dequant_block_packed(.., mp_type_tag<T>{})

// ---- IQ4_NL (source: ggml-quants.c dequantize_row_iq4_nl :2725) ----
// y[j] = d*kvalues[qs[j]&0xF] for j<16, y[j+16] = d*kvalues[qs[j]>>4]  ->  weight w reads byte (w%16),
// low/high nibble by w/16; with kb_off in {0,16}: byte = w, hi = kb_off>>4.
template <typename LUT>
static __device__ __forceinline__ void dequant_block_packed(
        const char * d_ptr, const char * code, const int kb_off, half * out, const LUT & lut,
        mp_type_tag<GGML_TYPE_IQ4_NL>) {
    const float d = mp_d_to_float(d_ptr);
    const bool hi = kb_off & 16;
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        const int nib = hi ? (code[w] >> 4) : (code[w] & 0xF);
        out[w] = __float2half(d * __half2float(lut.g[nib][0]));
    }
}

// ---- Q8_0 (source: ggml-quants.c dequantize_row_q8_0 :553) ----
// y[j] = qs[j]*d
template <typename LUT>
static __device__ __forceinline__ void dequant_block_packed(
        const char * d_ptr, const char * code, const int kb_off, half * out, const LUT & lut,
        mp_type_tag<GGML_TYPE_Q8_0>) {
    GGML_UNUSED(lut);
    const float d = mp_d_to_float(d_ptr);
    const int8_t * qs = (const int8_t *) (code + kb_off);
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        out[w] = __float2half(d * (float) qs[w]);
    }
}

// ---- Q2_0 (source: ggml-quants.c dequantize_row_q2_0 :439) ----
// q = (qs[j/4] >> 2*(j%4)) & 3; y[j] = (q-1)*d   (00=-1, 01=0, 10=+1, 11=+2)
template <typename LUT>
static __device__ __forceinline__ void dequant_block_packed(
        const char * d_ptr, const char * code, const int kb_off, half * out, const LUT & lut,
        mp_type_tag<GGML_TYPE_Q2_0>) {
    GGML_UNUSED(lut);
    const float d = mp_d_to_float(d_ptr);
    const uint8_t * qs = (const uint8_t *) code;
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        const int g = kb_off + w;
        const uint8_t q = (qs[g >> 2] >> (2 * (g & 3))) & 0x03;
        out[w] = __float2half(((int) q - 1) * d);
    }
}

// ---- IQ2_XXS (source: ggml-quants.c dequantize_row_iq2_xxs :2488) ----
// per 32-weight group ib32: db = d*(0.5 + ls)*0.25 (ls = aux32[1]>>28); grid bytes from aux32[0]'s 4
// low bytes; signs from ksigns_iq2xs[(aux32[1]>>7l)&127]; y = db*grid[j]*(sign&kmask[j] ? -1 : 1)
template <typename LUT>
static __device__ __forceinline__ void dequant_block_packed(
        const char * d_ptr, const char * code, const int kb_off, half * out, const LUT & lut,
        mp_type_tag<GGML_TYPE_IQ2_XXS>) {
    const float d = mp_d_to_float(d_ptr);
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        const int g    = kb_off + w;            // weight index inside the 256-block
        const int ib32 = g >> 5;
        const int l    = (g >> 3) & 3;          // grid-index byte within the group
        const int j    = g & 7;                 // value within the grid entry

        const uint32_t aux32_0 = (uint32_t) get_int_b2(code + 4*ib32 + 0, 0);
        const uint32_t aux32_1 = (uint32_t) get_int_b2(code + 4*ib32 + 4, 0);

        const float db = d * (0.5f + (aux32_1 >> 28)) * 0.25f;
        const uint8_t grid_idx = ((const uint8_t *) &aux32_0)[l];
        const uint8_t signs    = lut.ksigns[(aux32_1 >> 7*l) & 127];

        const float v = db * __half2float(lut.g[grid_idx][j]);
        out[w] = __float2half((signs & lut.kmask[j]) ? -v : v);
    }
}

// ---- IQ2_XS (source: ggml-quants.c dequantize_row_iq2_xs :2516) ----
// db[0/1] = d*(0.5 + scales nibble)*0.25; per uint16 q: grid = iq2xs_grid[q&511],
// signs = ksigns[q>>9]; y = db[l/2]*grid[j]*(sign&kmask[j] ? -1 : 1)
template <typename LUT>
static __device__ __forceinline__ void dequant_block_packed(
        const char * d_ptr, const char * code, const int kb_off, half * out, const LUT & lut,
        mp_type_tag<GGML_TYPE_IQ2_XS>) {
    const float d = mp_d_to_float(d_ptr);
    const uint8_t * c8 = (const uint8_t *) code;
    const uint16_t * c16 = (const uint16_t *) code;
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        const int g = kb_off + w;
        const int u = g >> 3;                    // uint16 index within the 32 qs entries
        const int j = g & 7;
        const int l = u & 3;                     // uint16 index within the 32-weight group

        const uint16_t q16 = c16[u];
        const uint8_t  sc  = c8[64 + (g >> 5)];
        const float db = d * (0.5f + ((((l >> 1) & 1) != 0) ? (sc >> 4) : (sc & 0xF))) * 0.25f;

        const uint8_t signs = lut.ksigns[q16 >> 9];

        const float v = db * __half2float(lut.g[q16 & 511][j]);
        out[w] = __float2half((signs & lut.kmask[j]) ? -v : v);
    }
}

// ---- IQ2_S (source: ggml-quants.c dequantize_row_iq2_s :2543) ----
// db[0/1] = d*(0.5 + scales nibble)*0.25; grid idx = qs[l] | (qh[ib32] << (8-2*l) & 0x300);
// signs = qs[32+l] (code bytes 32..63); y = db*grid[j]*(sign&kmask[j] ? -1 : 1)
template <typename LUT>
static __device__ __forceinline__ void dequant_block_packed(
        const char * d_ptr, const char * code, const int kb_off, half * out, const LUT & lut,
        mp_type_tag<GGML_TYPE_IQ2_S>) {
    const float d = mp_d_to_float(d_ptr);
    const uint8_t * c8 = (const uint8_t *) code;
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        const int g    = kb_off + w;
        const int l    = g >> 3;                 // qs byte within the block (grid idx + sign byte)
        const int j    = g & 7;
        const int ib32 = g >> 5;

        const uint8_t qs_b = c8[l];
        const uint8_t qh_b = c8[64 + ib32];
        const uint8_t sc   = c8[72 + ib32];
        const uint8_t sign = c8[32 + l];

        const float db = d * (0.5f + ((((l >> 1) & 1) != 0) ? (sc >> 4) : (sc & 0xF))) * 0.25f;
        const int grid_idx = qs_b | ((qh_b << (8 - 2*(l & 3))) & 0x300);

        const float v = db * __half2float(lut.g[grid_idx][j]);
        out[w] = __float2half((sign & lut.kmask[j]) ? -v : v);
    }
}

// ---- IQ3_XXS (source: ggml-quants.c dequantize_row_iq3_xxs :2575) ----
// db = d*(0.5 + ls)*0.5 (ls = aux32>>28); per 4-weight pair: grid1/2 = iq3xxs_grid[qs[2l]/qs[2l+1]];
// signs from ksigns[(aux32>>7l)&127]; y[j+0/4] = db*grid[j]*(sign&kmask[j+0/4] ? -1 : 1)
template <typename LUT>
static __device__ __forceinline__ void dequant_block_packed(
        const char * d_ptr, const char * code, const int kb_off, half * out, const LUT & lut,
        mp_type_tag<GGML_TYPE_IQ3_XXS>) {
    const float d = mp_d_to_float(d_ptr);
    const uint8_t * c8 = (const uint8_t *) code;
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        const int g    = kb_off + w;
        const int ib32 = g >> 5;
        const int pair = (g >> 2) & 7;           // 4-weight grid pair within the group (0..7)
        const int l    = pair >> 1;              // pair index within the group (0..3)
        const int ishi = pair & 1;               // grid1 (low 4 weights) vs grid2 (high 4)
        const int j    = g & 3;

        const uint8_t q = c8[8*ib32 + 2*l + ishi];
        const uint32_t aux32 = (uint32_t) get_int_b2(code + 64 + 4*ib32, 0);

        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        const uint8_t signs = lut.ksigns[(aux32 >> 7*l) & 127];

        const float v = db * __half2float(lut.g[q][j]);
        out[w] = __float2half((signs & lut.kmask[j + 4*ishi]) ? -v : v);
    }
}

// ---- IQ3_S (source: ggml-quants.c dequantize_row_iq3_s :2607) ----
// db1/db2 = d*(1 + 2*scales nibble) per two 32-groups; grid idx = qs[2l] | (qh[ib32%2] << (8-2l) & 256)
// (grid2 uses << (7-2l)); signs = signs[4*ib32 + l]; y[j+0/4] = db*grid[j]*(sign&kmask[j+0/4] ? -1 : 1)
template <typename LUT>
static __device__ __forceinline__ void dequant_block_packed(
        const char * d_ptr, const char * code, const int kb_off, half * out, const LUT & lut,
        mp_type_tag<GGML_TYPE_IQ3_S>) {
    const float d = mp_d_to_float(d_ptr);
    const uint8_t * c8 = (const uint8_t *) code;
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        const int g    = kb_off + w;
        const int ib32 = g >> 5;
        const int pair = (g >> 2) & 7;
        const int l    = pair >> 1;
        const int ishi = pair & 1;
        const int j    = g & 3;

        const uint8_t qs_b = c8[8*ib32 + 2*l + ishi];
        const uint8_t qh_b = c8[64 + (ib32 & 1)];
        const uint8_t sc_b = c8[104 + (ib32 >> 1)];
        const uint8_t sign = c8[72 + 4*ib32 + l];

        const float db = d * (1.0f + 2.0f*(((ib32 & 1) != 0) ? (sc_b >> 4) : (sc_b & 0xF)));
        const int grid_idx = qs_b | ((qh_b << ((ishi ? 7 : 8) - 2*l)) & 256);

        const float v = db * __half2float(lut.g[grid_idx][j]);
        out[w] = __float2half((sign & lut.kmask[j + 4*ishi]) ? -v : v);
    }
}

// =================================================================================================
// ---- monpack_tc_grouped: persistent grouped tensor-core kernel -----------------------------------
// ---- Fixed launch: 80 CTA x 256 threads (V100 = 80 SM, 1 CTA/SM). Work items come from a         --
// ---- device-built list (MoE) or are derived arithmetically (dense). Tile: M=16, N=64, J=16.      --
// ---- Per warp: 32x8 output via two m8n8k4-based mma() calls per 8-k step (tile<32,4,half2> A,    --
// ---- tile<8,4,half2,I_MAJOR_MIRRORED> B -> tile<32,8,float> C). A slice (16 rows x 16 k) is      --
// ---- staged once per k step and reused by all 8 warp column slices; B rows 16..31 of the A tile  --
// ---- mirror rows 0..15 (the m8n8k4 A operand needs 32 rows), the duplicate C rows are simply     --
// ---- not written by the epilogue.                                                                --
// =================================================================================================

#define MP_TC_NCTAS    80     // fixed launch, one CTA per V100 SM
#define MP_TC_NTHREADS 256
#define MP_TC_J        16     // k step (halves); 16 divides every block_size, so a step never straddles blocks
#define MP_TC_STRIDE   18     // smem row stride in halves (16 + 2 padding => 9 words, odd => conflict-free)
#define MP_TC_M        16     // M tile
#define MP_TC_N        64     // N tile

struct mp_wl_entry {
    uint32_t e;        // expert
    uint32_t n_tile;   // N-tile index within the expert
    uint32_t m_e;      // live rows in this M block (1..16)
    uint32_t rows_off; // offset into rows[] (or the dense row base when rows == nullptr)
};

// Prelude kernel: builds the (e, n_tile, m_e, rows_off) work list on device from ids.
// Single 256-thread block; zero host interaction. rows[] packs (token<<8 | iex_slot).
__launch_bounds__(256, 1)
static __global__ void mp_tc_build_wl(
        const int32_t * __restrict__ ids, const uint32_t ids_stride, // elements per token row
        const uint32_t n_used, const uint32_t n_tokens, const uint32_t n_experts,
        const uint32_t n_tiles_n,
        uint32_t * __restrict__ cnt,       // [n_experts] scratch
        uint32_t * __restrict__ base,      // [n_experts + 1]
        uint32_t * __restrict__ rows,      // [n_used*n_tokens]
        mp_wl_entry * __restrict__ wl, uint32_t * __restrict__ wl_count) {
    const int tid = threadIdx.x;

    for (uint32_t i = tid; i < n_experts; i += 256) cnt[i] = 0;
    if (tid == 0) *wl_count = 0;
    __syncthreads();

    // Phase 1: count rows per expert. The per-expert atomic rank alone is NOT the
    // global rows[] slot (experts would collide on rows[0..max_count)); rows[] must
    // be laid out per-expert contiguous, so the scatter happens in phase 2 below
    // with slot = base[e] + arrival rank.
    const uint32_t total = n_used*n_tokens;
    for (uint32_t i = tid; i < total; i += 256) {
        const uint32_t t   = i / n_used;
        const uint32_t iex = i % n_used;
        atomicAdd(&cnt[ids[iex + t*ids_stride]], 1u);
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t acc = 0;
        for (uint32_t e = 0; e < n_experts; ++e) {
            base[e] = acc;
            acc    += cnt[e];
        }
        base[n_experts] = acc;
    }
    __syncthreads();

    // Phase 2: re-seed cnt[e] as the per-expert write cursor (base[e], base[e]+1, ...)
    // and scatter rows[] into per-expert contiguous slots.
    for (uint32_t e = tid; e < n_experts; e += 256) cnt[e] = base[e];
    __syncthreads();
    for (uint32_t i = tid; i < total; i += 256) {
        const uint32_t t   = i / n_used;
        const uint32_t iex = i % n_used;
        const uint32_t e   = (uint32_t) ids[iex + t*ids_stride];
        const uint32_t slot = atomicAdd(&cnt[e], 1u);
        rows[slot] = (t << 8) | (iex & 0xFF);
    }
    __syncthreads();

    // cnt[e] now equals base[e+1]; the expert's row count is cnt[e] - base[e].
    for (uint32_t e = tid; e < n_experts; e += 256) {
        const uint32_t c = cnt[e] - base[e];
        for (uint32_t mb = 0; 16*mb < c; ++mb) {
            const uint32_t m_e = (c - 16*mb >= 16) ? 16 : (c - 16*mb);
            for (uint32_t nt = 0; nt < n_tiles_n; ++nt) {
                const uint32_t idx = atomicAdd(wl_count, 1u);
                wl[idx].e        = e;
                wl[idx].n_tile   = nt;
                wl[idx].m_e      = m_e;
                wl[idx].rows_off = base[e] + 16*mb;
            }
        }
    }
}

// Per-thread staging task: task 0..31 -> A smem rows 0..31 (rows 16..31 mirror 0..15),
// task 32..95 -> B smem rows 0..63, tasks >= 96 idle (256 threads cover 96 tasks exactly once per
// k slice). Produces 16 halves into `out` (register staging for the software pipeline).
template <ggml_type type>
static __device__ __forceinline__ void mp_tc_stage_task(
        const int task, const uint32_t k0,
        const mp_view_t & W, const uint32_t e, const uint32_t n_tile, const uint32_t rows_off, const uint32_t m_e,
        const uint32_t * __restrict__ rows,     // nullptr for dense
        const float * __restrict__ src1, const int64_t s_tok, const int64_t s_slot, const uint32_t ne11,
        const typename mp_tc_lut_type<type>::type & lut,
        half * out /*[16]*/) {
    if (task >= 2*MP_TC_M + MP_TC_N) {
        return;
    }
    if (task < 2*MP_TC_M) {
        const uint32_t m = task & (MP_TC_M - 1); // mirrored A rows
        if (m < m_e) {
            uint32_t t, iex;
            if (rows) {
                const uint32_t r = rows[rows_off + m];
                t = r >> 8;
                iex = r & 0xFF;
            } else {
                t = rows_off + m;                // dense: rows_off is the M-block base
                iex = 0;
            }
            const float * a = src1 + (uint64_t)t*s_tok + (uint64_t)(iex % ne11)*s_slot;
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) out[kk] = __float2half(a[k0 + kk]);
        } else {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) out[kk] = (half) 0.f;
        }
    } else {
        const uint32_t r  = task - 2*MP_TC_M;
        constexpr int qk = mp_type_traits<type>::qk;
        const uint32_t kb = k0 / qk;
        const char * dp; const char * cp;
        mp_block(W, e, n_tile*MP_TC_N + r, kb, dp, cp);
        dequant_block_packed(dp, cp, k0 - kb*qk, out, lut, mp_type_tag<type>{});
    }
}

template <ggml_type type>
__launch_bounds__(MP_TC_NTHREADS, 1)
static __global__ void mp_tc_grouped(
        const mp_view_t W,
        const float * __restrict__ src1, const int64_t s_tok, const int64_t s_slot, const uint32_t ne11,
        float * __restrict__ dst, const int64_t dst_ch_stride, const int64_t dst_tok_stride,
        const uint32_t ne_k, const uint32_t ne_n,
        const mp_wl_entry * __restrict__ wl, const uint32_t * __restrict__ rows,
        const uint32_t * __restrict__ wl_count, const uint32_t dense_m) {
    __shared__ half A_s[2][2*MP_TC_M][MP_TC_STRIDE];
    __shared__ half B_s[2][MP_TC_N][MP_TC_STRIDE];
    __shared__ typename mp_tc_lut_type<type>::type lut;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n_off = warp*8;   // 8 warps x 8 columns = 64

    lut.stage(tid, MP_TC_NTHREADS);
    __syncthreads();

    const uint32_t n_items = wl ? *wl_count : (((dense_m + MP_TC_M - 1)/MP_TC_M) * W.n_tiles_n);
    const uint32_t k_steps = ne_k / MP_TC_J;

    for (uint32_t it = blockIdx.x; it < n_items; it += gridDim.x) {
        uint32_t e, n_tile, m_e, rows_off;
        if (wl) {
            const mp_wl_entry w = wl[it];
            e = w.e; n_tile = w.n_tile; m_e = w.m_e; rows_off = w.rows_off;
        } else {
            const uint32_t nt = W.n_tiles_n;
            n_tile  = it % nt;
            const uint32_t mb = it / nt;
            e = 0;
            m_e = (dense_m - MP_TC_M*mb >= MP_TC_M) ? MP_TC_M : (dense_m - MP_TC_M*mb);
            rows_off = MP_TC_M*mb;
        }

        using ggml_cuda_mma::tile;
        using ggml_cuda_mma::DATA_LAYOUT_I_MAJOR_MIRRORED;
        tile<32, 8, float> acc;

        // software pipeline: register-stage the next k slice while the mma consumes the current one
        half regs[16];
        mp_tc_stage_task<type>(tid, 0, W, e, n_tile, rows_off, m_e, rows, src1, s_tok, s_slot, ne11, lut, regs);
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            if (tid < 2*MP_TC_M) {
                A_s[0][tid][kk] = regs[kk];
            } else if (tid < 2*MP_TC_M + MP_TC_N) {
                B_s[0][tid - 2*MP_TC_M][kk] = regs[kk];
            }
        }
        __syncthreads();

        for (uint32_t js = 0; js < k_steps; ++js) {
            const int cur = js & 1;

            if (js + 1 < k_steps) {
                mp_tc_stage_task<type>(tid, (js + 1)*MP_TC_J, W, e, n_tile, rows_off, m_e, rows,
                                       src1, s_tok, s_slot, ne11, lut, regs);
            }


            {
                // ggml_cuda_mma's Volta tiles express the fragment mapping in terms of
                // threadIdx.x (tile<32,4,half2>: i = threadIdx.x, j = l;
                // tile<8,4,half2,I_MAJOR_MIRRORED>: i = (threadIdx.x/16)*4 + threadIdx.x%4,
                // j = l) and load_generic therefore only produces in-bounds, correctly
                // mapped fragments for 32-thread blocks. This kernel runs
                // MP_TC_NTHREADS = 256 threads (8 warps x 8 columns), so the fragments
                // are filled directly with the same library mapping evaluated at the
                // warp-relative lane -- exactly the fragments a 32-thread block would
                // load. Every warp shares the full 32-row A slice (rows 16..31 mirror
                // 0..15 by construction of mp_tc_stage_task) and reads its own 8-row
                // B slice via bs = B_s[cur][n_off].
                tile<32, 4, half2> a0, a1;
                tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> b0, b1;
                const half2 * as = (const half2 *) &A_s[cur][0][0];
                const half2 * bs = (const half2 *) &B_s[cur][n_off][0];
                const int arow = lane;
                const int brow = (lane/16)*4 + (lane & 3);
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    a0.x[l] = as[arow*(MP_TC_STRIDE/2) + l];
                    a1.x[l] = as[arow*(MP_TC_STRIDE/2) + 4 + l];
                    b0.x[l] = bs[brow*(MP_TC_STRIDE/2) + l];
                    b1.x[l] = bs[brow*(MP_TC_STRIDE/2) + 4 + l];
                }
                ggml_cuda_mma::mma(acc, a0, b0);
                ggml_cuda_mma::mma(acc, a1, b1);
            }
            __syncthreads();

            if (js + 1 < k_steps) {
#pragma unroll
                for (int kk = 0; kk < 16; ++kk) {
                    if (tid < 2*MP_TC_M) {
                        A_s[cur ^ 1][tid][kk] = regs[kk];
                    } else if (tid < 2*MP_TC_M + MP_TC_N) {
                        B_s[cur ^ 1][tid - 2*MP_TC_M][kk] = regs[kk];
                    }
                }
            }
            __syncthreads();
        }

        // epilogue: lanes 0..15 hold C rows 0..15 — i = (l&2)+(lane&~2), j = (lane&2)+(l&5)
        // (tile<32,8,float>::get_i/get_j on Volta); mask m_e variable-length rows and ne_n bound.
        if (lane < 16) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const uint32_t i = (l & 2) + (uint32_t)(lane & ~2);
                const uint32_t j = (uint32_t)(lane & 2) + (l & 5);
                if (i >= m_e) {
                    continue;
                }
                const uint32_t gn = n_tile*MP_TC_N + n_off + j;
                if (gn >= ne_n) {
                    continue;
                }
                uint32_t t, iex;
                if (rows) {
                    const uint32_t r = rows[rows_off + i];
                    t = r >> 8;
                    iex = r & 0xFF;
                } else {
                    t = rows_off + i;
                    iex = 0;
                }
                dst[(uint64_t)iex*dst_ch_stride + (uint64_t)t*dst_tok_stride + gn] = acc.x[l];
            }
        }
    }
}

} // namespace monpack
