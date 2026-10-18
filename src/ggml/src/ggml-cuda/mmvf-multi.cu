#include "mmvf-multi.cuh"

// Several f32/bf16 GEMVs that read the same activations (1..4 columns, K <= 2560) in one launch: the router and the
// shared-expert gate, the GDN alpha and beta projections, the indexer q and k projections. Weights are read with
// 16-byte (f32) or 8-byte (bf16) loads. Each segment keeps
// the epilogue of its node: none, sigmoid, or softplus(x + pre[row]) * mul[row] (see ggml_cuda_mm_fusion_args_host).

#define MMVF_MULTI_NW  8 // warps per block

struct mmvf_multi_args {
    int n_seg;
    int row_end[MMVF_MULTI_MAX_SEG];  // cumulative rows
    const char * w[MMVF_MULTI_MAX_SEG];
    int64_t      w_row[MMVF_MULTI_MAX_SEG]; // bytes per weight row
    int          bf16[MMVF_MULTI_MAX_SEG];
    float      * dst[MMVF_MULTI_MAX_SEG];
    int64_t      dst_col[MMVF_MULTI_MAX_SEG]; // floats between output columns
    int          post_op[MMVF_MULTI_MAX_SEG];
    const float * pre[MMVF_MULTI_MAX_SEG];
    const float * mul[MMVF_MULTI_MAX_SEG];
};

// weights of 4 consecutive columns k..k+3 of a row
static __device__ __forceinline__ float4 mmvf_multi_w4(const char * row, const bool bf16, const int k) {
    if (bf16) {
        const uint2 u = *(const uint2 *) (row + (size_t) k*2);
        return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u),
                           __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xFFFF0000u));
    }
    return *(const float4 *) (row + (size_t) k*4);
}

template <int n, int off>
static __device__ __forceinline__ void mmvf_multi_reduce_steps(float * v, const int lane) {
    const bool hi = lane & off;
#pragma unroll
    for (int i = 0; i < n; ++i) {
        const float send = hi ? v[i] : v[i + n];
        const float keep = hi ? v[i + n] : v[i];
        v[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, off, WARP_SIZE);
    }
    if constexpr (n > 1) {
        mmvf_multi_reduce_steps<n/2, off/2>(v, lane);
    }
}

// block: MMVF_MULTI_RPB rows (of any segments); its warps split K, every lane issues all of its weight and activation
// loads at once (activations reused for the block's rows), then one transpose-reduce per warp and a shared-memory
// sum over the warps
#define MMVF_MULTI_LMAX 3 // float4 steps per lane: K <= MMVF_MULTI_NW*WARP_SIZE*4*LMAX = 3072
template <int T, int MMVF_MULTI_RPB>
static __global__ void __launch_bounds__(MMVF_MULTI_NW*WARP_SIZE) mmvf_multi_kernel(
        const float * __restrict__ x, const int64_t x_col, const int K, const mmvf_multi_args a) {
    constexpr int NV = MMVF_MULTI_RPB*T <= 2 ? 2 : MMVF_MULTI_RPB*T <= 4 ? 4 : MMVF_MULTI_RPB*T <= 8 ? 8 : 16;
    __shared__ float part[MMVF_MULTI_NW][NV];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int K4   = K/4;
    const int n_rows = a.row_end[a.n_seg - 1];
    const int slice  = (K4 + MMVF_MULTI_NW - 1) / MMVF_MULTI_NW; // float4 per warp
    const int k_beg  = warp*slice;
    const int k_end  = min(K4, k_beg + slice);

    int seg[MMVF_MULTI_RPB];
    int lrow[MMVF_MULTI_RPB];
    const char * wrow[MMVF_MULTI_RPB];
    bool bf[MMVF_MULTI_RPB];
    bool ok[MMVF_MULTI_RPB];
#pragma unroll
    for (int r = 0; r < MMVF_MULTI_RPB; ++r) {
        const int row = blockIdx.x*MMVF_MULTI_RPB + r;
        ok[r] = row < n_rows;
        int s = 0;
        while (s + 1 < a.n_seg && row >= a.row_end[s]) {
            ++s;
        }
        seg[r]  = s;
        lrow[r] = row - (s > 0 ? a.row_end[s - 1] : 0);
        wrow[r] = a.w[s] + (size_t) (ok[r] ? lrow[r] : 0)*a.w_row[s];
        bf[r]   = a.bf16[s];
    }

    float4 wv[MMVF_MULTI_RPB][MMVF_MULTI_LMAX];
    float4 xv[T][MMVF_MULTI_LMAX];
#pragma unroll
    for (int l = 0; l < MMVF_MULTI_LMAX; ++l) {
        const int k = k_beg + lane + l*WARP_SIZE;
        const bool in = k < k_end;
#pragma unroll
        for (int t = 0; t < T; ++t) {
            xv[t][l] = in ? *(const float4 *) (x + (size_t) t*x_col + (size_t) 4*k) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int r = 0; r < MMVF_MULTI_RPB; ++r) {
            wv[r][l] = in && ok[r] ? mmvf_multi_w4(wrow[r], bf[r], 4*k) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    float acc[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        acc[i] = 0.0f;
    }
#pragma unroll
    for (int l = 0; l < MMVF_MULTI_LMAX; ++l) {
#pragma unroll
        for (int r = 0; r < MMVF_MULTI_RPB; ++r) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[r*T + t] += wv[r][l].x*xv[t][l].x + wv[r][l].y*xv[t][l].y + wv[r][l].z*xv[t][l].z + wv[r][l].w*xv[t][l].w;
            }
        }
    }

    mmvf_multi_reduce_steps<NV/2, WARP_SIZE/2>(acc, lane);
    float v = acc[0];
#pragma unroll
    for (int off = WARP_SIZE/(2*NV); off > 0; off /= 2) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, off, WARP_SIZE);
    }
    if (lane % (WARP_SIZE/NV) == 0) {
        part[warp][lane / (WARP_SIZE/NV)] = v;
    }
    __syncthreads();
    if (threadIdx.x < MMVF_MULTI_RPB*T) {
        const int r = threadIdx.x / T;
        const int t = threadIdx.x % T;
        if (ok[r]) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < MMVF_MULTI_NW; ++w) {
                sum += part[w][threadIdx.x];
            }
            const int s  = seg[r];
            const int lr = lrow[r];
            if (a.post_op[s] == 1) {
                sum = 1.0f / (1.0f + expf(-sum));
            } else if (a.post_op[s] == 2) {
                sum = sum + a.pre[s][lr];
                sum = (sum > 20.0f) ? sum : logf(1.0f + expf(sum));
                sum = sum * a.mul[s][lr];
            }
            a.dst[s][(size_t) t*a.dst_col[s] + lr] = sum;
        }
    }
}

bool ggml_cuda_mmvf_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const ggml_cuda_mmvf_multi_seg * segs,
        const int n_seg) {
    const int64_t K = src1->ne[0];
    const int     T = (int) src1->ne[1];
    if (n_seg < 1 || n_seg > MMVF_MULTI_MAX_SEG || T < 1 || T > 4 || K % 4 != 0 || K > MMVF_MULTI_NW*WARP_SIZE*4*MMVF_MULTI_LMAX ||
            src1->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 ||
            (uintptr_t) src1->data % 16 != 0 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    mmvf_multi_args a = {};
    a.n_seg = n_seg;
    int rows = 0;
    for (int s = 0; s < n_seg; ++s) {
        const ggml_tensor * w = segs[s].w;
        const ggml_tensor * d = segs[s].dst;
        const bool bf16 = w->type == GGML_TYPE_BF16;
        if ((w->type != GGML_TYPE_F32 && !bf16) || w->ne[0] != K || w->nb[0] != ggml_type_size(w->type) ||
                w->ne[2] != 1 || w->ne[3] != 1 || w->nb[1] % (bf16 ? 8 : 16) != 0 || (uintptr_t) w->data % 16 != 0 ||
                d->type != GGML_TYPE_F32 || d->ne[0] != w->ne[1] || d->nb[0] != sizeof(float)) {
            return false;
        }
        rows += (int) w->ne[1];
        a.row_end[s] = rows;
        a.w[s]       = (const char *) w->data;
        a.w_row[s]   = w->nb[1];
        a.bf16[s]    = bf16;
        a.dst[s]     = (float *) segs[s].out;
        a.dst_col[s] = d->nb[1] / sizeof(float);
        a.post_op[s] = segs[s].post_op;
        a.pre[s]     = segs[s].pre ? (const float *) segs[s].pre->data : nullptr;
        a.mul[s]     = segs[s].mul ? (const float *) segs[s].mul->data : nullptr;
    }
    // few rows (alpha + beta): one row per block, so enough SMs take part; wide GEMVs: 4 rows share the activations
    const int rpb = rows < 256 ? 1 : 4;
    const dim3 grid((rows + rpb - 1) / rpb);
    const dim3 block(MMVF_MULTI_NW*WARP_SIZE);
    const size_t smem = 0;
    const float * x = (const float *) src1->data;
    const int64_t x_col = src1->nb[1] / sizeof(float);
    cudaStream_t stream = ctx.stream();
    const auto launch = [&](auto t_tag) {
        constexpr int TT = decltype(t_tag)::value;
        if (rpb == 1) {
            mmvf_multi_kernel<TT, 1><<<grid, block, smem, stream>>>(x, x_col, (int) K, a);
        } else {
            mmvf_multi_kernel<TT, 4><<<grid, block, smem, stream>>>(x, x_col, (int) K, a);
        }
    };
    switch (T) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
