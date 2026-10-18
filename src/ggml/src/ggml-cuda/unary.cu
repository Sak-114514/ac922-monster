#include "unary.cuh"
#include "convert.cuh"

static __device__ __forceinline__ float op_abs(float x) {
    return fabsf(x);
}

static __device__ __forceinline__ float op_sgn(float x) {
    return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f)));
}

static __device__ __forceinline__ float op_neg(float x) {
    return -x;
}

static __device__ __forceinline__ float op_step(float x) {
    return x > 0.0f;
}

static __device__ __forceinline__ float op_gelu(float x) {
    return ggml_cuda_op_gelu_single(x);
}

static __device__ __forceinline__ float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;

    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}

static __device__ __forceinline__ float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;

    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __device__ __forceinline__ float op_silu(float x) {
    return ggml_cuda_op_silu_single(x);
}

static __device__ __forceinline__ float op_tanh(float x) {
    return tanhf(x);
}

static __device__ __forceinline__ float op_relu(float x) {
    return fmaxf(x, 0);
}

static __device__ __forceinline__ float op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_exp(float x) {
    return expf(x);
}

static __device__ __forceinline__ float op_sqr(float x) {
    return x * x;
}

static __device__ __forceinline__ float op_relu_sqr(float x) {
    const float r = fmaxf(x, 0.0f);
    return r * r;
}

static __device__ __forceinline__ float op_sqrt(float x) {
    return sqrtf(x);
}

static __device__ __forceinline__ float op_sin(float x) {
    return sinf(x);
}

static __device__ __forceinline__ float op_cos(float x) {
    return cosf(x);
}

static __device__ __forceinline__ float op_log(float x) {
    return logf(x);
}

static __device__ __forceinline__ float op_expm1(float x) {
    return expm1f(x);
}

static __device__ __forceinline__ float op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static __device__ __forceinline__ float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static __device__ __forceinline__ float op_floor(float x) {
    return floorf(x);
}

static __device__ __forceinline__ float op_ceil(float x) {
    return ceilf(x);
}

static __device__ __forceinline__ float op_round(float x) {
    return round(x);
}

static __device__ __forceinline__ float op_trunc(float x) {
    return trunc(x);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = (T)op((float)x[i]);
}

template <float (*op)(float), typename T>
static void unary_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_op_kernel<op, T>, launch_params, x, dst, k);
}

template <float (*op)(float)>
void ggml_cuda_op_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_abs>(ctx, dst);
}

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sgn>(ctx, dst);
}

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_neg>(ctx, dst);
}

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_step>(ctx, dst);
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu>(ctx, dst);
}

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_quick>(ctx, dst);
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_silu>(ctx, dst);
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_tanh>(ctx, dst);
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_relu>(ctx, dst);
}

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sigmoid>(ctx, dst);
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardsigmoid>(ctx, dst);
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardswish>(ctx, dst);
}

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_exp>(ctx, dst);
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqr>(ctx, dst);
}

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqrt>(ctx, dst);
}

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sin>(ctx, dst);
}

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_cos>(ctx, dst);
}

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_log>(ctx, dst);
}

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_elu>(ctx, dst);
}

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_floor>(ctx, dst);
}

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_ceil>(ctx, dst);
}

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_round>(ctx, dst);
}

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_trunc>(ctx, dst);
}

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_expm1>(ctx, dst);
}

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_softplus>(ctx, dst);
}
/* gated ops */

template <float (*op)(float), typename T>
static __global__ void unary_gated_op_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ggml_cuda_pdl_sync();
    dst[i] = (T)(op((float)x[j0]) * (float)g[j1]);
}

// unary_gated_op_kernel that also writes its output quantized to q8_1 exactly as quantize_q8_1 does, laid out for a GEMV
// that reads the output as rows of ne10 (padded to ne10_padded with zero quants). One warp per quant block.
template <float (*op)(float)>
static __global__ void unary_gated_q8_kernel(const float * x, const float * g, float * dst, const int64_t n, const int64_t o0,
        const int64_t o1, block_q8_1 * y, const int64_t ne10, const int64_t ne10_padded) {
    const int64_t c = (int64_t) blockDim.x*blockIdx.x + threadIdx.x; // column in the GEMV's row
    const int64_t r = blockIdx.y;
    if (c >= ne10_padded) {
        return;
    }
    float xi = 0.0f;
    if (c < ne10) {
        const int64_t i  = r*ne10 + c;
        const int64_t j0 = (i / n) * o0 + (i % n);
        const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
        xi = op(x[j0]) * g[j1];
        dst[i] = xi;
    }
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const int64_t i_cont = r*ne10_padded + c;
    y[i_cont / QK8_1].qs[i_cont % QK8_1] = q;
    if (i_cont % QK8_1 == 0) {
        y[i_cont / QK8_1].ds = make_half2(d, sum);
    }
}

// if a quantized GEMV reads dst next, write dst and its q8_1 form in one kernel; false if not applicable
template <float (*op)(float)>
static bool unary_gated_q8_try(ggml_backend_cuda_context & ctx, const ggml_tensor * dst_node, const float * x, const float * g,
        float * dst, const int64_t n, const int64_t o0, const int64_t o1) {
    static const bool disabled = getenv("GGML_CUDA_NO_Q8_PRODUCE") != nullptr || getenv("GGML_CUDA_NO_Q8_CACHE") != nullptr;
    if (disabled) {
        return false;
    }
    const ggml_tensor * key = ctx.q8_1_consumer(dst_node);
    if (key == nullptr) {
        return false;
    }
    const int64_t ne10        = key->ne[0];
    const int64_t nrows       = ggml_nrows(key);
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    void * y = ctx.q8_1_cache_produce(key, dst_node, nrows*ne10_padded/QK8_1*sizeof(block_q8_1));
    if (y == nullptr) {
        return false;
    }
    const dim3 blocks((ne10_padded + 255) / 256, nrows, 1);
    unary_gated_q8_kernel<op><<<blocks, 256, 0, ctx.stream()>>>(x, g, dst, n, o0, o1, (block_q8_1 *) y, ne10, ne10_padded);
    return true;
}

template <float (*op)(float), typename T>
static void unary_gated_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_gated_op_kernel<op, T>, launch_params, x, g, dst, k, n, o0, o1);
}

template <float (*op)(float)>
void ggml_cuda_op_unary_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ((const int32_t *) dst->op_params)[1];

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (half *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        if (!unary_gated_q8_try<op>(ctx, dst, src0_p, src1_p, (float *) dst_d, nc, src0_o / sizeof(float), src1_o / sizeof(float))) {
            unary_gated_cuda<op>(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), stream);
        }
    }
}

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_relu>(ctx, dst);
}

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu>(ctx, dst);
}

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_silu>(ctx, dst);
}

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_quick>(ctx, dst);
}

// swiglu_oai

template <typename T>
static __global__ void swiglu_oai_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float alpha, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    float xi = x[j0];
    float gi = g[j1];

    dst[i] = ggml_cuda_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
static void swiglu_oai_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float alpha, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_oai_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(x, g, dst, k, n, o0, o1, alpha, limit);
}

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    float * src0_p = (float *) src0_d;
    float * src1_p = (float *) src1_d;

    if (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_cuda(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), alpha, limit, stream);
}

// swiglu_clamp

template <typename T>
static __global__ void swiglu_clamp_kernel(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    dst[i] = (T) ggml_cuda_op_swiglu_clamp_single((float) gate[j0], (float) up[j1], limit);
}

template <typename T>
static void swiglu_clamp_cuda(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_clamp_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(gate, up, dst, k, n, o0, o1, limit);
}

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float limit = ggml_get_op_params_f32(dst, 3);

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (half *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), limit, stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (float *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), limit, stream);
    }
}

/* CUDA kernel + launcher for xIELU */

template <typename T>
static __global__ void xielu_kernel(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float xi = ggml_cuda_cast<float>(x[i]);

    const float gate_pos = (xi > 0.0f);
    const float y_pos = alpha_p * xi * xi + beta * xi;
    const float min_v_eps = fminf(xi, eps);
    const float y_neg = (expm1f(min_v_eps) - xi) * alpha_n + beta * xi;
    const float out = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;

    dst[i] = ggml_cuda_cast<T>(out);
}

template <typename T>
static void xielu_cuda(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_XIELU_BLOCK_SIZE) / CUDA_XIELU_BLOCK_SIZE;
    xielu_kernel<<<num_blocks, CUDA_XIELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, alpha_n, alpha_p, beta, eps);
}

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta    = ggml_get_op_params_f32(dst, 3);
    const float eps     = ggml_get_op_params_f32(dst, 4);

    if (src0->type == GGML_TYPE_F16) {
        xielu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else {
        xielu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    }
}



/* silu_back */

static __device__ __forceinline__ float op_silu_back(float grad, float x) {
    const float s = 1.0f / (1.0f + expf(-x));
    return grad * s * (1.0f + x * (1.0f - s));
}

template <class T>
static __global__ void silu_back_kernel(const T * grad, const T * xf, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_silu_back((float)grad[i], (float)xf[i]);
}

template <class T>
static void silu_back_cuda(const T * grad, const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back_kernel<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_back_cuda((const half *)src0_d, (const half *)src1_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_back_cuda((const float*)src0_d, (const float*)src1_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

/* leaky relu */

static __device__ __forceinline__ float op_leaky_relu(float x, const float negative_slope) {
    return fmaxf(x, 0) + fminf(x, 0.0f) * negative_slope;
}

template <class T>
static __global__ void leaky_relu_kernel(const T * x, T * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_leaky_relu((float)x[i], negative_slope);
}

template <class T>
static void leaky_relu_cuda(const T * x, T * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_kernel<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        leaky_relu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), negative_slope, stream);
    } else {
        leaky_relu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), negative_slope, stream);
    }
}

/* fused unary + mul */

template <float (*op)(float)>
static void ggml_cuda_op_unary_mul_impl(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    // unary_node: UNARY op applied to unary_node->src[0]
    // mul_node:   MUL(a, b) where one of a/b is unary_node
    // Output goes to mul_node->data

    const ggml_tensor * unary_src = unary_node->src[0];  // input to the unary op
    const ggml_tensor * other_src = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(unary_src));
    GGML_ASSERT(unary_src->nb[0] == ggml_element_size(unary_src));
    GGML_ASSERT(ggml_is_contiguous_1(other_src));
    GGML_ASSERT(other_src->nb[0] == ggml_element_size(other_src));
    GGML_ASSERT(ggml_are_same_shape(unary_src, other_src));

    GGML_ASSERT(unary_src->type == GGML_TYPE_F32 || unary_src->type == GGML_TYPE_F16);
    GGML_ASSERT(unary_src->type == other_src->type);
    GGML_ASSERT(unary_src->type == mul_node->type);

    cudaStream_t stream = ctx.stream();

    const int64_t k  = ggml_nelements(mul_node);
    const int64_t nc = unary_src->ne[0];
    const int64_t unary_stride = unary_src->nb[1];
    const int64_t other_stride = other_src->nb[1];

    if (unary_src->type == GGML_TYPE_F16) {
        unary_gated_cuda<op>((const half *) unary_src->data, (const half *) other_src->data,
                             (half *) mul_node->data, k, nc,
                             unary_stride / sizeof(half), other_stride / sizeof(half), stream);
    } else {
        if (!unary_gated_q8_try<op>(ctx, mul_node, (const float *) unary_src->data, (const float *) other_src->data,
                                    (float *) mul_node->data, nc, unary_stride / sizeof(float), other_stride / sizeof(float))) {
            unary_gated_cuda<op>((const float *) unary_src->data, (const float *) other_src->data,
                                 (float *) mul_node->data, k, nc,
                                 unary_stride / sizeof(float), other_stride / sizeof(float), stream);
        }
    }
}

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SILU:
            ggml_cuda_op_unary_mul_impl<op_silu>(ctx, unary_node, mul_node);
            break;
        case GGML_UNARY_OP_SIGMOID:
            ggml_cuda_op_unary_mul_impl<op_sigmoid>(ctx, unary_node, mul_node);
            break;
        case GGML_UNARY_OP_SOFTPLUS:
            ggml_cuda_op_unary_mul_impl<op_softplus>(ctx, unary_node, mul_node);
            break;
        default:
            GGML_ABORT("Unsupported unary op for fused unary+mul");
    }
}

/* fused short elementwise chains around one unary op (decode graphs are launch-bound):
 *   SCALE -> UNARY                 u(x*s + b)
 *   ADD   -> UNARY -> MUL          u(x + c) * d
 *   UNARY -> MUL   -> ADD          u(g) * a + c
 * operands are contiguous F32 and broadcast against the chain's output shape */

// operand index modes against the output's flat index i (ne0 = output row length):
//   0: same shape (i)   1: one row broadcast over rows (i % ne0)   2: one value per row (i / ne0)
//   3: one value for everything
struct elem_chain_operand {
    const float * data = nullptr;
    int mode = 0;
};

struct elem_chain_args {
    elem_chain_operand u;      // unary input (before scale/bias/pre_add)
    elem_chain_operand pre;    // added before the unary
    elem_chain_operand mul;    // multiplied after the unary
    elem_chain_operand post;   // added last
    float scale = 1.0f;
    float bias  = 0.0f;
};

static __device__ __forceinline__ float elem_chain_load(const elem_chain_operand & o, const int i, const int i0, const int i1) {
    switch (o.mode) {
        case 0:  return o.data[i];
        case 1:  return o.data[i0];
        case 2:  return o.data[i1];
        default: return o.data[0];
    }
}

template <float (*op)(float), bool has_scale, bool has_pre, bool has_mul, bool has_post>
static __global__ void elem_chain_f32(const elem_chain_args args, float * dst, const int n, const uint3 ne0_fd) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
    const int i1 = dm.x;
    const int i0 = dm.y;

    float v = elem_chain_load(args.u, i, i0, i1);
    if constexpr (has_scale) {
        v = args.scale * v + args.bias;
    }
    if constexpr (has_pre) {
        v = v + elem_chain_load(args.pre, i, i0, i1);
    }
    v = op(v);
    if constexpr (has_mul) {
        v = v * elem_chain_load(args.mul, i, i0, i1);
    }
    if constexpr (has_post) {
        v = v + elem_chain_load(args.post, i, i0, i1);
    }
    dst[i] = v;
}

static int elem_chain_mode(const ggml_tensor * t, const ggml_tensor * dst) {
    if (ggml_are_same_shape(t, dst)) {
        return 0;
    }
    if (ggml_nelements(t) == 1) {
        return 3;
    }
    if (t->ne[0] == dst->ne[0] && ggml_nelements(t) == t->ne[0]) {
        return 1;
    }
    if (t->ne[0] == 1 && ggml_nelements(t) == ggml_nrows(dst)) {
        // one value per output row: rows must enumerate in the same order
        for (int d = 1; d < 4; ++d) {
            if (t->ne[d] != dst->ne[d]) {
                return -1;
            }
        }
        return 2;
    }
    return -1;
}

static bool elem_chain_operand_ok(const ggml_tensor * t, const ggml_tensor * dst) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && elem_chain_mode(t, dst) >= 0;
}

static elem_chain_operand elem_chain_operand_of(const ggml_tensor * t, const ggml_tensor * dst) {
    elem_chain_operand o;
    o.data = (const float *) t->data;
    o.mode = elem_chain_mode(t, dst);
    return o;
}

static bool elem_chain_unary_supported(const ggml_tensor * unary) {
    switch (ggml_get_unary_op(unary)) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_SOFTPLUS:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_elem_chain_supported(const ggml_tensor * const * nodes, const int n_nodes) {
    const ggml_tensor * out = nodes[n_nodes - 1];
    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || ggml_nelements(out) >= INT32_MAX) {
        return false;
    }
    for (int k = 0; k < n_nodes; ++k) {
        const ggml_tensor * t = nodes[k];
        if (t->op == GGML_OP_UNARY && !elem_chain_unary_supported(t)) {
            return false;
        }
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
        for (int j = 0; j < 2; ++j) {
            const ggml_tensor * s = t->src[j];
            if (s && (k == 0 || s != nodes[k - 1]) && !elem_chain_operand_ok(s, out)) {
                return false;
            }
        }
        // every intermediate result has the output's shape, except a broadcast unary input feeding MUL
        if (k > 0 && k < n_nodes - 1 && !ggml_are_same_shape(t, out)) {
            return false;
        }
    }
    return true;
}

template <float (*op)(float)>
static void elem_chain_launch(const elem_chain_args & args, float * dst, const int n, const uint3 ne0_fd, const bool has_scale,
        const bool has_pre, const bool has_mul, const bool has_post, cudaStream_t stream) {
    const int nb = (n + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    if (has_scale && !has_pre && !has_mul && !has_post) {
        elem_chain_f32<op, true, false, false, false><<<nb, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(args, dst, n, ne0_fd);
    } else if (!has_scale && has_pre && has_mul && !has_post) {
        elem_chain_f32<op, false, true, true, false><<<nb, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(args, dst, n, ne0_fd);
    } else if (!has_scale && !has_pre && has_mul && has_post) {
        elem_chain_f32<op, false, false, true, true><<<nb, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(args, dst, n, ne0_fd);
    } else {
        GGML_ABORT("unsupported elementwise chain");
    }
}

// nodes: SCALE,UNARY | ADD,UNARY,MUL | UNARY,MUL,ADD
void ggml_cuda_op_elem_chain(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, const int n_nodes) {
    const ggml_tensor * out = nodes[n_nodes - 1];
    elem_chain_args args;
    bool has_scale = false, has_pre = false, has_mul = false, has_post = false;
    const ggml_tensor * unary = nullptr;

    const auto other = [](const ggml_tensor * node, const ggml_tensor * prev) {
        return node->src[0] == prev ? node->src[1] : node->src[0];
    };

    if (nodes[0]->op == GGML_OP_SCALE) {
        unary = nodes[1];
        args.u = elem_chain_operand_of(nodes[0]->src[0], out);
        memcpy(&args.scale, (const float *) nodes[0]->op_params + 0, sizeof(float));
        memcpy(&args.bias,  (const float *) nodes[0]->op_params + 1, sizeof(float));
        has_scale = true;
    } else if (nodes[0]->op == GGML_OP_ADD) {
        unary = nodes[1];
        args.u   = elem_chain_operand_of(nodes[0]->src[0], out);
        args.pre = elem_chain_operand_of(nodes[0]->src[1], out);
        args.mul = elem_chain_operand_of(other(nodes[2], nodes[1]), out);
        has_pre = has_mul = true;
    } else {
        unary = nodes[0];
        args.u    = elem_chain_operand_of(nodes[0]->src[0], out);
        args.mul  = elem_chain_operand_of(other(nodes[1], nodes[0]), out);
        args.post = elem_chain_operand_of(other(nodes[2], nodes[1]), out);
        has_mul = has_post = true;
    }

    const int n = (int) ggml_nelements(out);
    const uint3 ne0_fd = init_fastdiv_values((uint32_t) out->ne[0]);
    float * dst = (float *) out->data;
    cudaStream_t stream = ctx.stream();
    switch (ggml_get_unary_op(unary)) {
        case GGML_UNARY_OP_SILU:     elem_chain_launch<op_silu>    (args, dst, n, ne0_fd, has_scale, has_pre, has_mul, has_post, stream); break;
        case GGML_UNARY_OP_SIGMOID:  elem_chain_launch<op_sigmoid> (args, dst, n, ne0_fd, has_scale, has_pre, has_mul, has_post, stream); break;
        case GGML_UNARY_OP_SOFTPLUS: elem_chain_launch<op_softplus>(args, dst, n, ne0_fd, has_scale, has_pre, has_mul, has_post, stream); break;
        default: GGML_ABORT("unsupported unary op in elementwise chain");
    }
}

/* fused relu + sqr */

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node) {
    const ggml_tensor * src = relu_node->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(src->type == sqr_node->type);

    const int k = ggml_nelements(src);
    if (src->type == GGML_TYPE_F16) {
        unary_cuda<op_relu_sqr>((const half *)src->data, (half *)sqr_node->data, k, stream);
    } else {
        unary_cuda<op_relu_sqr>((const float *)src->data, (float *)sqr_node->data, k, stream);
    }
}
