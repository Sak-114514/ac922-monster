#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// ssm_conv (+ bias + silu) that also writes RMS_NORM + SCALE of up to two head-aligned channel ranges (GDN q and k),
// one head (128 channels) per block. The norms are rounded as rms_norm_scale_f32: the per-row sum of squares goes
// through the same lane mapping and block reduction (the extra zero warps of the 256-thread version add exact zeros).
struct ssm_conv_norm_args {
    int     n;
    int64_t off[2];   // first channel
    int64_t nch[2];   // channels (a multiple of 128)
    float * out[2];   // [128, nch/128, n_t, n_s] contiguous
    float   eps[2];
    float   s[2];
};

template <bool apply_silu, size_t d_conv>
static __global__ void ssm_conv_norm_f32(const float * src0, const float * src1, const float * bias,
                                         const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                         float * dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                         const int64_t n_t, const ssm_conv_norm_args na) {
    constexpr int split_d_inner = 128;
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    // which normalized range this block's head belongs to, if any
    const int64_t c0 = (int64_t) bidy * split_d_inner;
    int k = -1;
    for (int q = 0; q < na.n; ++q) {
        if (c0 >= na.off[q] && c0 < na.off[q] + na.nch[q]) {
            k = q;
        }
    }
    __shared__ float s_sum[32];

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        const float y = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        y_block[i * stride_y + tid] = y;

        if (k >= 0) {
            float tmp = 0.0f;
            tmp += y * y;
            tmp = block_reduce<block_reduce_method::SUM, split_d_inner>(tmp, s_sum);
            __syncthreads(); // s_sum is reused by the next token
            const float mean  = tmp / split_d_inner;
            const float scale = rsqrtf(mean + na.eps[k]);
            const int64_t n_heads = na.nch[k] / split_d_inner;
            const int64_t head    = (c0 - na.off[k]) / split_d_inner;
            const float yn = scale * y;
            na.out[k][((bidx * n_t + i) * n_heads + head) * split_d_inner + tid] = yn * na.s[k];
        }
    }
}

bool ggml_cuda_op_ssm_conv_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node,
                                ggml_tensor * silu_dst, ggml_tensor * const * rms, ggml_tensor * const * scale, int n) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * bias = bias_add_node ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;
    const ggml_tensor * out  = silu_dst ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t n_t = out->ne[1];
    const int64_t n_s = out->ne[2];
    if (nc != 4 || n_t > 32 || nr % 128 != 0 || out->ne[0] != nr || src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) ||
            src0->nb[1] != src0->ne[0] * sizeof(float) || out->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32 ||
            out->nb[0] != sizeof(float) || (bias && (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) || ggml_nelements(bias) != nr))) {
        return false;
    }

    ssm_conv_norm_args na = {};
    na.n = n;
    for (int q = 0; q < n; ++q) {
        const ggml_tensor * v = rms[q]->src[0]; // [128, heads, n_t, n_s] window of out
        float sb = 0.0f;
        memcpy(&na.eps[q], rms[q]->op_params, sizeof(float));
        memcpy(&na.s[q],   (const float *) scale[q]->op_params + 0, sizeof(float));
        memcpy(&sb,        (const float *) scale[q]->op_params + 1, sizeof(float));
        if (v->view_src != out || v->ne[0] != 128 || v->nb[0] != sizeof(float) || v->nb[1] != 128*sizeof(float) ||
                v->ne[2] != n_t || v->nb[2] != out->nb[1] || v->ne[3] != n_s || (n_s > 1 && v->nb[3] != out->nb[2]) ||
                v->view_offs % (128*sizeof(float)) != 0 || sb != 0.0f || !ggml_is_contiguous(scale[q]) ||
                scale[q]->type != GGML_TYPE_F32 || !ggml_are_same_shape(scale[q], v)) {
            return false;
        }
        // the norms are written while other blocks still read the conv inputs: their memory may have been reused
        const auto overlaps = [](const ggml_tensor * x, const ggml_tensor * y) {
            const char * xb = (const char *) x->data;
            const char * yb = (const char *) y->data;
            return xb < yb + ggml_nbytes(y) && yb < xb + ggml_nbytes(x);
        };
        if (overlaps(scale[q], src0) || overlaps(scale[q], src1) || (bias && overlaps(scale[q], bias)) ||
                overlaps(scale[q], out)) {
            return false;
        }
        na.off[q] = v->view_offs / sizeof(float);
        na.nch[q] = 128 * v->ne[1];
        na.out[q] = (float *) scale[q]->data;
        if (na.off[q] + na.nch[q] > nr) {
            return false;
        }
    }

    const dim3 blocks(n_s, nr / 128, 1);
    if (silu_dst) {
        ssm_conv_norm_f32<true, 4><<<blocks, 128, 0, ctx.stream()>>>((const float *) src0->data, (const float *) src1->data,
            bias ? (const float *) bias->data : nullptr, src0->nb[1], src0->nb[2], src1->nb[1], (float *) out->data,
            out->nb[0], out->nb[1], out->nb[2], n_t, na);
    } else {
        ssm_conv_norm_f32<false, 4><<<blocks, 128, 0, ctx.stream()>>>((const float *) src0->data, (const float *) src1->data,
            bias ? (const float *) bias->data : nullptr, src0->nb[1], src0->nb[2], src1->nb[1], (float *) out->data,
            out->nb[0], out->nb[1], out->nb[2], n_t, na);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
