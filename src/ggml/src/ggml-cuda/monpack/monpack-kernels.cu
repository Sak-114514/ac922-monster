// monpack-kernels.cu — P3 monpack v2 host API (registry + dispatch hooks)
//
// Exports the four entry points documented in monpack/monpack.h:
//   monpack::register_tensor / is_registered / try_mul_mat / try_mul_mat_id
//
// Host launch geometry for the GEMV path replicates mmvq.cu (ggml_cuda_mul_mat_vec_q +
// mul_mat_vec_q_switch_ncols_dst / mul_mat_vec_q_moe_launch, Volta tables) so that the packed-layout
// kernels run with exactly the stock kernels' warp/lane<->k configuration.
// The TC path is fully device-driven (device-built work list, fixed 80x256 persistent launch),
// zero host-sync and zero per-call allocation => CUDA graph capture safe.

#include "monpack-kernels.cuh"
#include "monpack.h"
#include "../quantize.cuh" // quantize_row_q8_1_cuda

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace monpack {

// =================================================================================================
// ---- environment switches (monpack_format.h tail) ------------------------------------------------
// =================================================================================================

static bool mp_env_disabled() {
    static const bool v = [] {
        const char * e = getenv("GGML_CUDA_MONPACK_DISABLE");
        return e != nullptr && atoi(e) == 1;
    }();
    return v;
}

static int mp_env_tc_min_m() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_MONPACK_TC_MIN_M");
        return e != nullptr ? atoi(e) : 16;
    }();
    return v < 1 ? 1 : v;
}

// fork parity: GGML_SM70_DECODE_LINEAR_CHUNKS == "8" reroutes single-token MoE GEMV to the
// multi-token moe kernel (mmvq.cu mul_mat_vec_q_switch_ncols_dst :1482)
static bool mp_sm70_decode_linear_chunks() {
    static const bool v = [] {
        const char * env = getenv("GGML_SM70_DECODE_LINEAR_CHUNKS");
        return env && strcmp(env, "8") == 0;
    }();
    return v;
}

// fork parity: GGML_CUDA_IQ2S_VOLTA=0 disables the special IQ2_S Volta MoE kernel
static bool mp_iq2s_volta_disabled() {
    static const bool v = [] {
        const char * env = getenv("GGML_CUDA_IQ2S_VOLTA");
        return env && atoi(env) == 0;
    }();
    return v;
}

// fork parity (Bug 2): GGML_SM70_MOE_REWORK (default 0; the deployed launch-p3.sh sets 1) routes
// Q2_0 MoE GEMV for K in {192,256} — the down-projection island slices — to kernel_sm70_moe.cuh
// BEFORE the stock moe kernel. That kernel's per-row sequential fp32 accumulation differs from
// the stock warp-reduce order, so bit parity with the baseline requires the packed replica to
// fire under exactly the same conditions.
static bool mp_sm70_moe_rework() {
    static const bool v = [] {
        const char * env = getenv("GGML_SM70_MOE_REWORK");
        return env != nullptr && atoi(env) != 0;
    }();
    return v;
}

static bool mp_device_is_volta() {
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA;
}

// =================================================================================================
// ---- host-side per-type parameters (must match mp_type_traits / monpack_format.h table) ----------
// =================================================================================================

static constexpr int mp_host_block_size(const ggml_type t) {
    switch (t) {
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q8_0:   return 32;
        case GGML_TYPE_Q2_0:   return 64;
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:  return 256;
        default:               return 0;
    }
}

static constexpr uint32_t mp_host_code_bytes(const ggml_type t) {
    switch (t) {
        case GGML_TYPE_IQ4_NL:  return 18 - 2;
        case GGML_TYPE_Q8_0:    return 34 - 2;
        case GGML_TYPE_Q2_0:    return 18 - 2;
        case GGML_TYPE_IQ2_XXS: return 66 - 2;
        case GGML_TYPE_IQ2_XS:  return 74 - 2;
        case GGML_TYPE_IQ2_S:   return 82 - 2;
        case GGML_TYPE_IQ3_XXS: return 98 - 2;
        case GGML_TYPE_IQ3_S:   return 110 - 2;
        default:                return 0;
    }
}

// VDR for host geometry decisions: mp_get_vdr in monpack-kernels.cuh (same table as mmvq.cu).

// =================================================================================================
// ---- grouped-TC device work-list buffers ----------------------------------------------------------
// ---- PER-DEVICE (Bug 1 fix): one independent set per CUDA device. The work list is written and   --
// ---- read by kernels launched on the tensor's own device; a single process-wide buffer would     --
// ---- make device A's kernels touch memory cudaMalloc'ed on device B — with no peer access        --
// ---- between the islands that is an illegal memory access. Each device's set is allocated        --
// ---- lazily at load time (register_tensor, before any graph capture) and grown on demand —       --
// ---- never during an op, so graph capture only ever replays the two kernel launches. Per-device  --
// ---- buffers are naturally graph-replay safe and keep the zero host-sync contract.               --
// ---- Capacity: worst case 512 experts x ceil(M/16) M-blocks x n_tiles items and 64K rows.        --
// =================================================================================================

#define MP_TC_MAX_ROWS (64*1024)   // row-map capacity (token<<8 | iex_slot slots)
#define MP_TC_MAX_EXPERTS 512      // constitution worst case

struct mp_tc_buffers {
    void *        mem      = nullptr;
    mp_wl_entry * wl       = nullptr; // [(MP_TC_MAX_ROWS/16 + 512) * n_tiles]
    uint32_t *    rows     = nullptr; // [MP_TC_MAX_ROWS]
    uint32_t *    cnt      = nullptr; // [MP_TC_MAX_EXPERTS]
    uint32_t *    base     = nullptr; // [MP_TC_MAX_EXPERTS + 1]
    uint32_t *    wl_count = nullptr; // [1]
    size_t        wl_cap   = 0;
};
static mp_tc_buffers g_tc[GGML_CUDA_MAX_DEVICES];
static std::mutex g_tc_mutex;

// Reserve on `device` — the cudaMalloc binds to the CURRENT device, so the caller must have
// selected `device` (ggml_cuda_set_device) before calling.
static bool mp_tc_reserve(const int device, const uint32_t n_tiles_n) {
    static_assert(MP_TC_MAX_ROWS % 16 == 0, "M-block granularity");
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    std::lock_guard<std::mutex> lock(g_tc_mutex);
    mp_tc_buffers & b = g_tc[device];
    const size_t wl_cap = (MP_TC_MAX_ROWS/16 + MP_TC_MAX_EXPERTS) * (size_t) n_tiles_n;
    if (b.mem != nullptr && b.wl_cap >= wl_cap) {
        return true;
    }
    cudaFree(b.mem);
    const size_t bytes = wl_cap * sizeof(mp_wl_entry) + MP_TC_MAX_ROWS * 4 +
                         (MP_TC_MAX_EXPERTS + 1) * 4 + (MP_TC_MAX_EXPERTS + 1) * 4 + 4;
    if (cudaMalloc(&b.mem, bytes) != cudaSuccess) {
        b.mem      = nullptr;
        b.wl       = nullptr;
        b.rows     = nullptr;
        b.cnt      = nullptr;
        b.base     = nullptr;
        b.wl_count = nullptr;
        b.wl_cap   = 0;
        return false;
    }
    char * p = (char *) b.mem;
    b.wl       = (mp_wl_entry *) p;           p += wl_cap * sizeof(mp_wl_entry);
    b.rows     = (uint32_t *) p;              p += MP_TC_MAX_ROWS * 4;
    b.cnt      = (uint32_t *) p;              p += (MP_TC_MAX_EXPERTS + 1) * 4;
    b.base     = (uint32_t *) p;              p += (MP_TC_MAX_EXPERTS + 1) * 4;
    b.wl_count = (uint32_t *) p;
    b.wl_cap   = wl_cap;
    return true;
}

// =================================================================================================
// ---- tensor registry -----------------------------------------------------------------------------
// =================================================================================================

struct mp_meta {
    ggml_type type;
    uint32_t  fam;
    uint32_t  ne_k, ne_n, ne_e, k_tile;
    // derived packed view (monpack_format.h formulas)
    uint32_t  n_tiles_n, n_tiles_k, nb, code_bytes;
    uint64_t  tile_bytes;
};

static std::unordered_map<const void *, mp_meta> g_registry;
static std::mutex g_registry_mutex;

static mp_view_t mp_make_view(const mp_meta & m) {
    mp_view_t v;
    v.base        = nullptr; // filled per call with the tensor data pointer
    v.n_tiles_n   = m.n_tiles_n;
    v.n_tiles_k   = m.n_tiles_k;
    v.nb          = m.nb;
    v.code_bytes  = m.code_bytes;
    v.tile_bytes  = m.tile_bytes;
    return v;
}

void register_tensor(const ggml_tensor * t) {
    if (t == nullptr || t->data == nullptr) {
        return;
    }
    { static int mp_reg_dbg = 10; if (mp_reg_dbg > 0) { fprintf(stderr, "[mp-reg] type=%d ne=[%lld,%lld,%lld] ptr=%p\n", (int)t->type, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], t->data); mp_reg_dbg--; } }
    // family from the type table; k_tile per the monpack_format.h divisibility rule
    const uint32_t k = (uint32_t) t->ne[0];
    const uint32_t k_tile = (k % 256 == 0) ? 256 : (k % 128 == 0) ? 128 : (k % 64 == 0) ? 64 : 0;
    const uint32_t fam =
        (t->type == GGML_TYPE_IQ4_NL) ? MONPACK_FAM_A :
        (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q2_0) ? MONPACK_FAM_C : MONPACK_FAM_B;
    register_tensor(t->data, (uint32_t) t->type, fam,
                    (uint32_t) t->ne[0], (uint32_t) t->ne[1], (uint32_t) t->ne[2], k_tile, /*blob_base=*/ nullptr);
}

void register_tensor(const void * tensor_data, uint32_t ggml_type_, uint32_t family,
                     uint32_t ne_k, uint32_t ne_n, uint32_t ne_e, uint32_t k_tile,
                     const void * blob_base) {
    if (tensor_data == nullptr) {
        return;
    }
    const ggml_type t = (ggml_type) ggml_type_;
    if (!mp_type_supported(t)) {
        GGML_LOG_ERROR("monpack::register_tensor: unsupported ggml_type %u for %p\n", ggml_type_, tensor_data);
        return;
    }
    const int bs = mp_host_block_size(t);
    if (k_tile == 0 || (k_tile % bs) != 0 || ne_k == 0 || ne_n == 0 || ne_e == 0 ||
            (ne_k % k_tile) != 0 || k_tile > 256) {
        GGML_LOG_ERROR("monpack::register_tensor: bad shape k_tile=%u ne=%u/%u/%u for %p\n",
                       k_tile, ne_k, ne_n, ne_e, tensor_data);
        return;
    }
    // family cross-check against the monpack_format.h type table
    const uint32_t fam_expected =
        (t == GGML_TYPE_IQ4_NL) ? MONPACK_FAM_A :
        (t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q2_0) ? MONPACK_FAM_C : MONPACK_FAM_B;
    if (family != fam_expected) {
        GGML_LOG_ERROR("monpack::register_tensor: family %u does not match type %u for %p\n", family, ggml_type_, tensor_data);
        return;
    }
    if (blob_base != nullptr && (const char *) tensor_data < (const char *) blob_base) {
        GGML_LOG_ERROR("monpack::register_tensor: tensor_data %p below blob_base %p\n", tensor_data, blob_base);
        return;
    }

    mp_meta m;
    m.type       = t;
    m.fam        = family;
    m.ne_k       = ne_k;
    m.ne_n       = ne_n;
    m.ne_e       = ne_e;
    m.k_tile     = k_tile;
    m.nb         = k_tile / bs;
    m.code_bytes = mp_host_code_bytes(t);
    m.n_tiles_n  = (ne_n + MONPACK_TILE_N - 1) / MONPACK_TILE_N;
    m.n_tiles_k  = ne_k / k_tile;
    m.tile_bytes = (uint64_t) m.nb * MONPACK_TILE_N * (2 + m.code_bytes);

    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        g_registry[tensor_data] = m;
    }

    // grouped-TC work-list buffers: pre-allocate at load time (before any graph capture), one
    // independent set per device. The registry cannot know which device this tensor lands on
    // (island/meta-buffer members get their final buffer after registration), so reserve for
    // every device; the per-device cost is a few MB and the buffers are device-local by
    // construction (Bug 1: the old single process-wide buffer caused cross-device illegal
    // memory accesses on this 6-GPU, dual-island rig).
    {
        const int prev_device = ggml_cuda_get_device();
        const int n_devices   = ggml_cuda_info().device_count;
        bool all_ok = true;
        for (int d = 0; d < n_devices && d < GGML_CUDA_MAX_DEVICES; ++d) {
            ggml_cuda_set_device(d);
            all_ok = mp_tc_reserve(d, m.n_tiles_n) && all_ok;
        }
        ggml_cuda_set_device(prev_device);
        if (!all_ok) {
            GGML_LOG_ERROR("monpack::register_tensor: TC work-list buffer allocation failed for %p "
                           "(TC MoE falls back to chunked GEMV on the affected device)\n", tensor_data);
        }
    }
}

bool is_registered(const void * tensor_data) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    return tensor_data != nullptr && g_registry.count(tensor_data) != 0;
}

static bool mp_lookup(const void * p, mp_meta & m) {
    if (p == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    const auto it = g_registry.find(p);
    if (it == g_registry.end()) {
        return false;
    }
    m = it->second;
    return true;
}

// =================================================================================================
// ---- GEMV host geometry (replicates mmvq.cu should_use_small_k + Volta tables for sm_70) ----------
// =================================================================================================

#define MP_VOLTA_SK_NWARPS 2   // mmvq.cu MMVQ_VOLTA_SK_NWARPS
#define MP_VOLTA_SK_ROWS   4   // mmvq.cu MMVQ_VOLTA_SK_ROWS

template <ggml_type type>
static bool mp_should_use_small_k(const int ncols_dst, const int ncols_x) {
    // mmvq.cu should_use_small_k (Volta: not turing_plus, not pascal_older, not RDNA => only the
    // ncols_dst == 1 iq_slow_other exclusion applies)
    const int qk  = mp_type_traits<type>::qk;
    const int qi  = ggml_cuda_type_traits<type>::qi;
    const int vdr = mp_get_vdr(type);
    const int blocks_per_row_x      = ncols_x / qk;
    const int blocks_per_iter_1warp = vdr * WARP_SIZE / qi;
    const int nwarps                = mp_calc_nwarps_default(ncols_dst); // default (non-SK) shape

    bool use = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

    if (ncols_dst == 1) {
        switch (type) { // iq_slow_other list from mmvq.cu (types present in the monpack table)
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
                return false;
            default:
                break;
        }
    }
    return use;
}

struct mp_geo {
    int nwarps;
    int rpb;
};

template <ggml_type type>
static mp_geo mp_choose_geo(const int ncols_dst, const int ncols_x) {
    const bool sk = mp_should_use_small_k<type>(ncols_dst, ncols_x);
    const int nw = (ncols_dst >= 2 && ncols_dst <= 8 && sk) ? (int) MP_VOLTA_SK_NWARPS
                                                            : mp_calc_nwarps_default(ncols_dst);
    return {nw, mp_calc_rows_per_block(ncols_dst, sk, nw)};
}

// =================================================================================================
// ---- GEMV entry: q8_1 quantization + cache (verbatim port of ggml_cuda_mul_mat_vec_q :1732) ------
// =================================================================================================

template <ggml_type type>
static void mp_moe_launch(const mp_meta & meta, const mp_view_t & view, const void * src1_q8_1,
        const int32_t * ids_d, float * dst_d,
        const int64_t ne10, const int64_t s11, const int64_t s12,
        const int64_t ne11, const int64_t ne12,
        const int64_t nrows_x, const int64_t nchannels_dst,
        const int64_t s1, const int64_t s2,
        const int64_t ids_stride, cudaStream_t stream);

template <ggml_type type>
static void mp_dense_launch(const mp_meta & meta, const mp_view_t & view, const void * src1_q8_1,
        const int32_t * ids_d, float * dst_d,
        const int64_t ne10, const int64_t ne10_padded,
        const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t nrows_x, const int64_t nchannels_dst,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const int64_t ids_stride, cudaStream_t stream) {
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s12 = ne11 * s11;
    const int64_t s13 = ne12 * s12;

    const uint3 nchannels_y_fd   = ids_d ? init_fastdiv_values(ne11) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids_d ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / meta.ne_e);
    const uint3 sample_ratio_fd  = init_fastdiv_values(ne13); // nsamples_dst / nsamples_x, nsamples_x == 1

    // fork parity (mmvq.cu :1482): has_ids && (tokens > 1 || (sm70 decode-chunks)) -> MoE kernel
    const bool is_moe = ids_d && (ne12 > 1 || (mp_sm70_decode_linear_chunks() && ne13 == 1));

    if (is_moe) {
        mp_moe_launch<type>(meta, view, src1_q8_1, ids_d, dst_d,
                ne10, s11, s12, ne11, ne12, nrows_x, nchannels_dst, s1, s2, ids_stride, stream);
        return;
    }

    const int64_t ncols_dst_d = ids_d ? 1 : ne11;
    const mp_geo geo = mp_choose_geo<type>((int) ncols_dst_d, (int) ne10);

    const auto run = [&](auto ncols_tag, auto nw_tag, auto rpb_tag) {
        constexpr int c_ncols = decltype(ncols_tag)::value;
        constexpr int c_nw    = decltype(nw_tag)::value;
        constexpr int c_rpb   = decltype(rpb_tag)::value;
        const int64_t nblocks = (nrows_x + c_rpb - 1) / c_rpb;
        const dim3 block_nums((unsigned) nblocks, (unsigned) nchannels_dst, (unsigned) ne13);
        const dim3 block_dims(WARP_SIZE, c_nw, 1);
        const auto lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(mp_mul_mat_vec_q<type, c_ncols, c_nw, c_rpb>, lp,
            view, meta.ne_e, src1_q8_1, ids_d, dst_d,
            (uint32_t) ne10, nchannels_y_fd,
            (uint32_t) s11,                            // stride_col_y      (dense)
            (uint32_t) (ids_d ? s2 : s1),              // stride_col_dst
            channel_ratio_fd,
            (uint32_t) (ids_d ? s11 : s12),            // stride_channel_y
            (uint32_t) (ids_d ? s1 : s2),              // stride_channel_dst
            sample_ratio_fd,
            (uint32_t) s13, (uint32_t) s3, (uint32_t) ids_stride);
    };

    const auto dispatch_geo = [&](auto ncols_tag) {
        constexpr int c_ncols = decltype(ncols_tag)::value;
        if (c_ncols == 1) {
            if (geo.rpb == 4) { run(ncols_tag, std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{}); }
            else              { run(ncols_tag, std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{}); }
        } else if (c_ncols <= 4) {
            if (geo.nwarps == 2) { run(ncols_tag, std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{}); }
            else                 { run(ncols_tag, std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{}); }
        } else {
            if (geo.rpb == 4) { run(ncols_tag, std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{}); }
            else              { run(ncols_tag, std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{}); }
        }
    };

    switch (ncols_dst_d) {
        case 1: dispatch_geo(std::integral_constant<int, 1>{}); break;
        case 2: dispatch_geo(std::integral_constant<int, 2>{}); break;
        case 3: dispatch_geo(std::integral_constant<int, 3>{}); break;
        case 4: dispatch_geo(std::integral_constant<int, 4>{}); break;
        case 5: dispatch_geo(std::integral_constant<int, 5>{}); break;
        case 6: dispatch_geo(std::integral_constant<int, 6>{}); break;
        case 7: dispatch_geo(std::integral_constant<int, 7>{}); break;
        case 8: dispatch_geo(std::integral_constant<int, 8>{}); break;
        default:
            GGML_ABORT("monpack: fatal error");
    }
}

template <ggml_type type>
static void mp_moe_launch(const mp_meta & meta, const mp_view_t & view, const void * src1_q8_1,
        const int32_t * ids_d, float * dst_d,
        const int64_t ne10, const int64_t s11, const int64_t s12,
        const int64_t ne11, const int64_t ne12,
        const int64_t nrows_x, const int64_t nchannels_dst,
        const int64_t s1, const int64_t s2,
        const int64_t ids_stride, cudaStream_t stream) {
    GGML_UNUSED(meta);

    const uint3 nchannels_y_fd = init_fastdiv_values(ne11);

    // fork parity (mmvq.cu mul_mat_vec_q_moe_launch :134): with GGML_SM70_MOE_REWORK=1 (the
    // deployed engine default) Q2_0 MoE GEMV for K in {192,256} takes the sm70 rework kernel
    // whose accumulation order defines the baseline's numerics — replicate it bit for bit.
    if constexpr (type == GGML_TYPE_Q2_0) {
        if (mp_sm70_moe_rework() && mp_device_is_volta() &&
                (ne10 == 192 || ne10 == 256) && ne12 >= 2 && ne12 <= 8 && nrows_x > 0) {
            const int     kb   = (int) (ne10 / 64); // 3 for K=192, 4 for K=256
            const int     y_pitch = (((int) (ne10/32) * (int) sizeof(block_q8_1)) + 31) & ~15;
            const size_t  smem = (size_t) y_pitch + mp_moe_sm70::ROWS_PER_BLOCK * 4;
            const dim3 block_nums((unsigned) ((nrows_x + mp_moe_sm70::ROWS_PER_BLOCK - 1) / mp_moe_sm70::ROWS_PER_BLOCK),
                                  (unsigned) nchannels_dst, (unsigned) ne12);
            const dim3 block_dims(mp_moe_sm70::WARP_SZ, mp_moe_sm70::NW);
            const auto lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, smem, stream);
            if (kb == 4) {
                ggml_cuda_kernel_launch(mp_moe_sm70::mp_moe_matvec_q2_0_sm70<4>, lp,
                    view, src1_q8_1, ids_d, dst_d,
                    (uint32_t) ne10, (uint32_t) nrows_x,
                    (uint32_t) s12, (uint32_t) s2,      // stride_col_y,      stride_col_dst
                    (uint32_t) s11, (uint32_t) s1,      // stride_channel_y,  stride_channel_dst
                    (uint32_t) ids_stride, (uint32_t) ne11);
            } else {
                ggml_cuda_kernel_launch(mp_moe_sm70::mp_moe_matvec_q2_0_sm70<3>, lp,
                    view, src1_q8_1, ids_d, dst_d,
                    (uint32_t) ne10, (uint32_t) nrows_x,
                    (uint32_t) s12, (uint32_t) s2,
                    (uint32_t) s11, (uint32_t) s1,
                    (uint32_t) ids_stride, (uint32_t) ne11);
            }
            return;
        }
    }

    // fork parity: the Volta IQ2_S MoE special kernel is the default IQ2_S path
    // (mmvq.cu mul_mat_vec_q_moe_launch :1333) for no-fusion calls of supported K.
    if constexpr (type == GGML_TYPE_IQ2_S) {
        const bool no_fusion = true;
        if (!mp_iq2s_volta_disabled() && no_fusion && ne10 % QK_K == 0 && ne10 <= MP_IQ2S_VOLTA_MAX_K) {
            constexpr int rows_per_cuda_block = MP_IQ2S_VOLTA_ROWS * MP_IQ2S_VOLTA_NWARPS;
            const dim3 block_nums((unsigned) ((nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block),
                                  (unsigned) nchannels_dst, (unsigned) ne12);
            const dim3 block_dims(WARP_SIZE, MP_IQ2S_VOLTA_NWARPS);
            const auto lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
            ggml_cuda_kernel_launch(mp_mul_mat_iq2_s_moe_volta, lp,
                view, src1_q8_1, ids_d, dst_d,
                (uint32_t) ne10, nchannels_y_fd, (uint32_t) nrows_x,
                (uint32_t) s12, (uint32_t) s2,
                (uint32_t) s11, (uint32_t) s1,
                (uint32_t) ids_stride);
            return;
        }
    }

    // 2 rows per block; IQ4_NL uses 8 on Volta (mmvq.cu mul_mat_vec_q_moe_launch :1265)
    constexpr int c_rpb = (type == GGML_TYPE_IQ4_NL) ? 8 : 2;
    const int64_t nblocks = (nrows_x + c_rpb - 1) / c_rpb;
    const dim3 block_nums((unsigned) nblocks, (unsigned) nchannels_dst);
    const dim3 block_dims(WARP_SIZE, (unsigned) ne12);
    const auto lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    ggml_cuda_kernel_launch(mp_mul_mat_vec_q_moe<type, c_rpb>, lp,
        view, src1_q8_1, ids_d, dst_d,
        (uint32_t) ne10, nchannels_y_fd, (uint32_t) nrows_x,
        (uint32_t) s12, (uint32_t) s2,
        (uint32_t) s11, (uint32_t) s1,
        (uint32_t) ne12, (uint32_t) ids_stride);
}

static void mp_gemv_switch_type(const mp_meta & meta, const mp_view_t & view, const void * src1_q8_1,
        const int32_t * ids_d, float * dst_d,
        const int64_t ne10, const int64_t ne10_padded,
        const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t nrows_x, const int64_t nchannels_dst,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const int64_t ids_stride, cudaStream_t stream) {
    switch (meta.type) {
        case GGML_TYPE_IQ4_NL:
            mp_dense_launch<GGML_TYPE_IQ4_NL>(meta, view, src1_q8_1, ids_d, dst_d, ne10, ne10_padded, ne11, ne12, ne13, nrows_x, nchannels_dst, s1, s2, s3, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mp_dense_launch<GGML_TYPE_Q8_0>(meta, view, src1_q8_1, ids_d, dst_d, ne10, ne10_padded, ne11, ne12, ne13, nrows_x, nchannels_dst, s1, s2, s3, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mp_dense_launch<GGML_TYPE_Q2_0>(meta, view, src1_q8_1, ids_d, dst_d, ne10, ne10_padded, ne11, ne12, ne13, nrows_x, nchannels_dst, s1, s2, s3, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mp_dense_launch<GGML_TYPE_IQ2_XXS>(meta, view, src1_q8_1, ids_d, dst_d, ne10, ne10_padded, ne11, ne12, ne13, nrows_x, nchannels_dst, s1, s2, s3, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mp_dense_launch<GGML_TYPE_IQ2_XS>(meta, view, src1_q8_1, ids_d, dst_d, ne10, ne10_padded, ne11, ne12, ne13, nrows_x, nchannels_dst, s1, s2, s3, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mp_dense_launch<GGML_TYPE_IQ2_S>(meta, view, src1_q8_1, ids_d, dst_d, ne10, ne10_padded, ne11, ne12, ne13, nrows_x, nchannels_dst, s1, s2, s3, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mp_dense_launch<GGML_TYPE_IQ3_XXS>(meta, view, src1_q8_1, ids_d, dst_d, ne10, ne10_padded, ne11, ne12, ne13, nrows_x, nchannels_dst, s1, s2, s3, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mp_dense_launch<GGML_TYPE_IQ3_S>(meta, view, src1_q8_1, ids_d, dst_d, ne10, ne10_padded, ne11, ne12, ne13, nrows_x, nchannels_dst, s1, s2, s3, ids_stride, stream);
            break;
        default:
            GGML_ABORT("monpack: fatal error");
    }
}

// The GEMV entry mirroring ggml_cuda_mul_mat_vec_q (mmvq.cu :1732), fusion removed.
static bool mp_run_gemv(ggml_backend_cuda_context & ctx, const mp_meta & meta,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type == GGML_TYPE_I32);

    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(!ids || ids->nb[0] == sizeof(int32_t));

    const float   * src1_d = (const float *)   src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *) ids->data : nullptr;
    float         *  dst_d = (float *)         dst->data;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_1_size   = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;

    // reuse the quantized input of an earlier GEMV that read the same activation (GGML_CUDA_NO_Q8_CACHE=1: off)
    static const bool q8_cache_on = getenv("GGML_CUDA_NO_Q8_CACHE") == nullptr;
    auto * qc = ctx.q8_1_cache;
    const bool q8_cacheable = q8_cache_on && q8_1_size <= (1 << 20);
    int q8_hit = -1;
    for (int c = 0; c < 2 && q8_cacheable; ++c) {
        if (qc[c].matches(src1)) {
            q8_hit = c;
        }
    }
    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
    char * src1_q8_1 = nullptr;
    if (q8_hit >= 0) {
        src1_q8_1 = (char *) qc[q8_hit].buf;
        if (q8_hit == 0) {
            std::swap(qc[0], qc[1]); // entry 1 is the most recently used
        }
    } else {
        if (q8_cacheable) {
            std::swap(qc[0], qc[1]); // reuse the least recently used entry, it becomes entry 1
            if (qc[1].buf == nullptr) {
                qc[1].cap = 1 << 20;
                CUDA_CHECK(cudaMalloc(&qc[1].buf, qc[1].cap));
            }
            src1_q8_1 = (char *) qc[1].buf;
        } else {
            src1_q8_1 = src1_q8_1_pool.alloc(q8_1_size);
        }
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        const int64_t s13 = src1->nb[3] / sizeof(float);
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        if (q8_cacheable) {
            qc[1].set(src1, nullptr);
        }
    }

    const int64_t s1  = dst->nb[1] / sizeof(float);
    const int64_t s2  = dst->nb[2] / sizeof(float);
    const int64_t s3  = dst->nb[3] / sizeof(float);

    const int64_t nchannels_dst = ids ? ne1 : ne2;
    const int64_t ids_stride = ids ? ids->nb[1] / sizeof(int32_t) : 0;

    mp_view_t view = mp_make_view(meta);
    view.base = (const char *) src0->data;

    mp_gemv_switch_type(meta, view, src1_q8_1, ids_d, dst_d,
        ne10, ne10_padded, ne11, ne12, ne13,
        ne01, nchannels_dst, s1, s2, s3, ids_stride, stream);
    return true;
}

// =================================================================================================
// ---- grouped TC entry -----------------------------------------------------------------------------
// =================================================================================================

template <ggml_type type>
static void mp_tc_launch(const mp_meta & meta, const mp_view_t & view,
        const float * src1_d, const int64_t s_tok, const int64_t s_slot, const uint32_t ne11,
        float * dst_d, const int64_t dst_ch_stride, const int64_t dst_tok_stride,
        const mp_wl_entry * wl, const uint32_t * rows, const uint32_t * wl_count, const uint32_t dense_m,
        cudaStream_t stream) {
    const dim3 block_nums(MP_TC_NCTAS);
    const dim3 block_dims(MP_TC_NTHREADS);
    const auto lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    ggml_cuda_kernel_launch(mp_tc_grouped<type>, lp,
        view, src1_d, s_tok, s_slot, ne11, dst_d, dst_ch_stride, dst_tok_stride,
        meta.ne_k, meta.ne_n, wl, rows, wl_count, dense_m);
}

static void mp_tc_switch_type(const mp_meta & meta, const mp_view_t & view,
        const float * src1_d, const int64_t s_tok, const int64_t s_slot, const uint32_t ne11,
        float * dst_d, const int64_t dst_ch_stride, const int64_t dst_tok_stride,
        const mp_wl_entry * wl, const uint32_t * rows, const uint32_t * wl_count, const uint32_t dense_m,
        cudaStream_t stream) {
    switch (meta.type) {
        case GGML_TYPE_IQ4_NL:  mp_tc_launch<GGML_TYPE_IQ4_NL >(meta, view, src1_d, s_tok, s_slot, ne11, dst_d, dst_ch_stride, dst_tok_stride, wl, rows, wl_count, dense_m, stream); break;
        case GGML_TYPE_Q8_0:    mp_tc_launch<GGML_TYPE_Q8_0  >(meta, view, src1_d, s_tok, s_slot, ne11, dst_d, dst_ch_stride, dst_tok_stride, wl, rows, wl_count, dense_m, stream); break;
        case GGML_TYPE_Q2_0:    mp_tc_launch<GGML_TYPE_Q2_0  >(meta, view, src1_d, s_tok, s_slot, ne11, dst_d, dst_ch_stride, dst_tok_stride, wl, rows, wl_count, dense_m, stream); break;
        case GGML_TYPE_IQ2_XXS: mp_tc_launch<GGML_TYPE_IQ2_XXS>(meta, view, src1_d, s_tok, s_slot, ne11, dst_d, dst_ch_stride, dst_tok_stride, wl, rows, wl_count, dense_m, stream); break;
        case GGML_TYPE_IQ2_XS:  mp_tc_launch<GGML_TYPE_IQ2_XS >(meta, view, src1_d, s_tok, s_slot, ne11, dst_d, dst_ch_stride, dst_tok_stride, wl, rows, wl_count, dense_m, stream); break;
        case GGML_TYPE_IQ2_S:   mp_tc_launch<GGML_TYPE_IQ2_S >(meta, view, src1_d, s_tok, s_slot, ne11, dst_d, dst_ch_stride, dst_tok_stride, wl, rows, wl_count, dense_m, stream); break;
        case GGML_TYPE_IQ3_XXS: mp_tc_launch<GGML_TYPE_IQ3_XXS>(meta, view, src1_d, s_tok, s_slot, ne11, dst_d, dst_ch_stride, dst_tok_stride, wl, rows, wl_count, dense_m, stream); break;
        case GGML_TYPE_IQ3_S:   mp_tc_launch<GGML_TYPE_IQ3_S >(meta, view, src1_d, s_tok, s_slot, ne11, dst_d, dst_ch_stride, dst_tok_stride, wl, rows, wl_count, dense_m, stream); break;
        default:
            GGML_ABORT("monpack: fatal error");
    }
}

// dense (ids == nullptr): rows are implicit, work items derived arithmetically — no buffers needed
static bool mp_run_tc_dense(ggml_backend_cuda_context & ctx, const mp_meta & meta,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const uint32_t m_tokens) {
    cudaStream_t stream = ctx.stream();
    mp_view_t view = mp_make_view(meta);
    view.base = (const char *) src0->data;

    const int64_t s_tok = src1->nb[1] / sizeof(float);
    const int64_t dst_tok = dst->nb[1] / sizeof(float);

    mp_tc_switch_type(meta, view, (const float *) src1->data, s_tok, /*s_slot=*/ 0, /*ne11=*/ 1,
        (float *) dst->data, /*dst_ch=*/ 0, dst_tok,
        nullptr, nullptr, nullptr, m_tokens, stream);
    return true;
}

// MoE: device-side work list built by the prelude kernel, then the persistent grouped kernel.
static bool mp_run_tc_moe(ggml_backend_cuda_context & ctx, const mp_meta & meta,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const uint32_t n_used   = (uint32_t) ids->ne[0];
    const uint32_t n_tokens = (uint32_t) ids->ne[1];
    const uint64_t m_total  = (uint64_t) dst->ne[1] * dst->ne[2];
    if (meta.ne_e > 512 || m_total > MP_TC_MAX_ROWS || src1->ne[3] != 1 || n_tokens != (uint32_t) dst->ne[2] ||
            n_used != (uint32_t) dst->ne[1] || (uint64_t) n_used*n_tokens != m_total) {
        return false;
    }

    mp_view_t view = mp_make_view(meta);
    view.base = (const char *) src0->data;

    const int64_t s_tok  = src1->nb[2] / sizeof(float);
    const int64_t s_slot = src1->nb[1] / sizeof(float);
    const uint32_t ne11  = (uint32_t) src1->ne[1];
    const int64_t dst_ch  = dst->nb[1] / sizeof(float);
    const int64_t dst_tok = dst->nb[2] / sizeof(float);
    const uint32_t ids_stride = (uint32_t) (ids->nb[1] / sizeof(int32_t));

    // per-device work-list set (Bug 1): this op's kernels run on ctx.device and must only
    // touch buffers allocated on that device
    mp_tc_buffers & tc = g_tc[ctx.device];
    if (tc.mem == nullptr || tc.wl == nullptr) {
        // register-time reservation missed this device (allocation failure): degrade to the
        // chunked GEMV route (still reads the packed layout correctly) instead of touching a
        // foreign-device buffer
        return false;
    }

    {
        const dim3 block_nums(1);
        const dim3 block_dims(256);
        const auto lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(mp_tc_build_wl, lp,
            (const int32_t *) ids->data, ids_stride, n_used, n_tokens, meta.ne_e, meta.n_tiles_n,
            tc.cnt, tc.base, tc.rows, tc.wl, tc.wl_count);
    }

    mp_tc_switch_type(meta, view, (const float *) src1->data, s_tok, s_slot, ne11,
        (float *) dst->data, dst_ch, dst_tok,
        tc.wl, tc.rows, tc.wl_count, /*dense_m=*/ 0, stream);
    return true;
}

// =================================================================================================
// ---- dispatch hooks -------------------------------------------------------------------------------
// =================================================================================================

static bool mp_src0_matches(const ggml_tensor * src0, const mp_meta & m) {
    return src0->ne[0] == m.ne_k && src0->ne[1] == m.ne_n &&
           src0->ne[2] == m.ne_e && src0->ne[3] == 1 &&
           ggml_is_contiguous(src0);
}

bool try_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * src0) {
    if (mp_env_disabled()) {
        return false;
    }
    mp_meta m;
    if (!mp_lookup(src0->data, m)) {
        return false;
    }
    const ggml_tensor * src1 = dst->src[1];
    if (src1 == nullptr || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            src1->ne[0] != m.ne_k || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            !mp_src0_matches(src0, m) || (m.ne_k % mp_host_block_size(m.type)) != 0 ||
            src1->ne[0] % QK8_1 != 0 || dst->ne[3] != src1->ne[3]) {
        return false;
    }

    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t m_total = ne11 * ne12 * ne13;
    const int tc_min = mp_env_tc_min_m();

    if (m_total <= MMVQ_MAX_BATCH_SIZE) {
        return mp_run_gemv(ctx, m, src0, src1, /*ids=*/ nullptr, dst);
    }
    if (m_total >= tc_min) {
        if (ne12 == 1 && ne13 == 1 && dst->ne[2] == 1 && dst->ne[3] == 1 && m_total <= MP_TC_MAX_ROWS) {
            return mp_run_tc_dense(ctx, m, src0, src1, dst, (uint32_t) m_total);
        }
        return false;
    }
    // 8 < M_total < tc_min: chunk the columns onto the GEMV (same device as the fork's
    // ggml_sm70_decode_linear_chunks tail handling)
    if (ne12 == 1 && ne13 == 1 && dst->ne[2] == 1 && dst->ne[3] == 1) {
        for (int64_t first = 0; first < ne11; first += MMVQ_MAX_BATCH_SIZE) {
            ggml_tensor input  = *src1;
            ggml_tensor output = *dst;
            const int64_t count = std::min<int64_t>(MMVQ_MAX_BATCH_SIZE, ne11 - first);
            input.ne[1] = output.ne[1] = count;
            input.data = (char *) src1->data + first * src1->nb[1];
            output.data = (char *) dst->data + first * dst->nb[1];
            output.src[0] = const_cast<ggml_tensor *>(src0);
            output.src[1] = &input;
            if (!mp_run_gemv(ctx, m, src0, &input, /*ids=*/ nullptr, &output)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool try_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    if (mp_env_disabled()) {
        return false;
    }
    mp_meta m;
    if (!mp_lookup(src0->data, m)) {
        return false;
    }
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    if (src1 == nullptr || ids == nullptr || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
            src1->ne[0] != m.ne_k || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            !mp_src0_matches(src0, m) || (m.ne_k % mp_host_block_size(m.type)) != 0 ||
            src1->ne[0] % QK8_1 != 0 || ids->ne[1] != dst->ne[2] || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            src1->ne[3] != 1 || dst->ne[3] != 1) {
        return false;
    }

    const int64_t ne2 = dst->ne[2];       // tokens
    const int64_t m_total = dst->ne[1] * ne2;
    const int tc_min = mp_env_tc_min_m();

    if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
        return mp_run_gemv(ctx, m, src0, src1, ids, dst);
    }
    if (m_total >= tc_min) {
        return mp_run_tc_moe(ctx, m, src0, src1, ids, dst);
    }
    // 8 < M_total < tc_min: chunk the tokens onto the MoE GEMV (fork decode-chunks style)
    if (src1->ne[3] == 1 && dst->ne[3] == 1) {
        for (int64_t first = 0; first < ne2; first += MMVQ_MAX_BATCH_SIZE) {
            ggml_tensor input   = *src1;
            ggml_tensor selected = *ids;
            ggml_tensor output  = *dst;
            const int64_t count = std::min<int64_t>(MMVQ_MAX_BATCH_SIZE, ne2 - first);
            input.ne[2] = output.ne[2] = selected.ne[1] = count;
            input.data   = (char *) src1->data + first * src1->nb[2];
            selected.data = (char *) ids->data + first * ids->nb[1];
            output.data  = (char *) dst->data + first * dst->nb[2];
            output.src[0] = const_cast<ggml_tensor *>(src0);
            output.src[1] = &input;
            output.src[2] = &selected;
            if (!mp_run_gemv(ctx, m, src0, &input, &selected, &output)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// ---- no-ctx stubs matching the originally inserted call-site shapes ------------------------------
// They must never compute: without the backend cuda context the correct stream (and with it CUDA
// graph capture membership, the pool and the q8_1 cache) is unrecoverable. Abort loudly instead of
// falling through to a stock kernel that would misread the packed bytes.
bool try_mul_mat(ggml_tensor * dst, const ggml_tensor * src0) {
    GGML_UNUSED(dst);
    GGML_UNUSED(src0);
    GGML_ABORT("monpack::try_mul_mat needs the backend cuda context — call "
               "monpack::try_mul_mat(ctx, dst, src0) from ggml_cuda_mul_mat (see monpack/monpack.h)");
    return false; // unreachable; silences -Wsuggest-attribute=noreturn
}

bool try_mul_mat_id(ggml_tensor * dst) {
    GGML_UNUSED(dst);
    GGML_ABORT("monpack::try_mul_mat_id needs the backend cuda context — call "
               "monpack::try_mul_mat_id(ctx, dst) from ggml_cuda_mul_mat_id (see monpack/monpack.h)");
    return false; // unreachable; silences -Wsuggest-attribute=noreturn
}

} // namespace monpack
