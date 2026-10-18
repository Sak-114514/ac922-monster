#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

// max and first index of the max of one row; the result is valid in thread 0
static __device__ void argmax_row_f32(const float * __restrict__ rowx, const int64_t ncols, float & maxval_out, int & argmax_out) {
    float maxval = -FLT_MAX;
    int   argmax = -1;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        // ties go to the lower index, like a sequential scan (col < 0: that lane saw no value)
        if (val > maxval || (val == maxval && col >= 0 && (argmax < 0 || col < argmax))) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval || (val == maxval && col >= 0 && (argmax < 0 || col < argmax))) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    maxval_out = maxval;
    argmax_out = argmax;
}

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;
    float maxval;
    int   argmax;
    argmax_row_f32(x + row * ncols, ncols, maxval, argmax);
    if (threadIdx.x == 0) {
        dst[row] = argmax;
    }
}

// GGML_OP_ARGMAX_SPLIT: this device holds columns [offset, offset + ncols) of the rows. Step 1: many blocks per row
// each find the max of ARGMAX_SPLIT_CHUNK columns. Step 2: per row, write (max, global index) into slot rank of a row
// of 2*n_dev floats. The other slots get 0 if the rows are summed across devices next (partial), else -inf.
#define ARGMAX_SPLIT_CHUNK 4096

static __global__ void argmax_split_part_f32(const float * __restrict__ x, float * __restrict__ pv, int * __restrict__ pi,
        const int64_t ncols, const int nchunks) {
    const int64_t row   = blockIdx.y;
    const int64_t c0    = (int64_t) blockIdx.x*ARGMAX_SPLIT_CHUNK;
    const int64_t n     = min((int64_t) ARGMAX_SPLIT_CHUNK, ncols - c0);
    float maxval;
    int   argmax;
    argmax_row_f32(x + row*ncols + c0, n, maxval, argmax);
    if (threadIdx.x == 0) {
        pv[row*nchunks + blockIdx.x] = maxval;
        pi[row*nchunks + blockIdx.x] = argmax < 0 ? -1 : (int) c0 + argmax;
    }
}

static __global__ void argmax_split_final_f32(const float * __restrict__ pv, const int * __restrict__ pi, float * __restrict__ dst,
        const int nchunks, const int n_dev, const int rank, const int offset, const int partial) {
    const int64_t row = blockIdx.x;
    float best = -FLT_MAX;
    int   bi   = -1;
    for (int c = threadIdx.x; c < nchunks; c += WARP_SIZE) { // chunks in order: ties keep the lower index
        const float v = pv[row*nchunks + c];
        const int   i = pi[row*nchunks + c];
        if (i >= 0 && (v > best || bi < 0)) {
            best = v;
            bi   = i;
        }
    }
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        const float v = __shfl_xor_sync(0xFFFFFFFF, best, off, WARP_SIZE);
        const int   i = __shfl_xor_sync(0xFFFFFFFF, bi,   off, WARP_SIZE);
        if (i >= 0 && (bi < 0 || v > best || (v == best && i < bi))) {
            best = v;
            bi   = i;
        }
    }
    if (threadIdx.x == 0) {
        float * d = dst + row * 2 * n_dev;
        for (int s = 0; s < n_dev; ++s) {
            d[2*s + 0] = s == rank ? best                           : (partial ? 0.0f : -INFINITY);
            d[2*s + 1] = s == rank ? (float) (offset + max(bi, 0)) : 0.0f;
        }
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}

void ggml_cuda_argmax_split(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));

    const int n_dev   = ggml_get_op_params_i32(dst, 0);
    const int rank    = ggml_get_op_params_i32(dst, 1);
    const int offset  = ggml_get_op_params_i32(dst, 2);
    const int partial = ggml_get_op_params_i32(dst, 3);
    GGML_ASSERT(dst->ne[0] == 2*n_dev && dst->ne[1] == src0->ne[1]);

    const int64_t ne00    = src0->ne[0];
    const int64_t nrows   = ggml_nrows(src0);
    const int     nchunks = (int) ((ne00 + ARGMAX_SPLIT_CHUNK - 1) / ARGMAX_SPLIT_CHUNK);

    ggml_cuda_pool_alloc<float> pv(ctx.pool(), nrows*nchunks);
    ggml_cuda_pool_alloc<int>   pi(ctx.pool(), nrows*nchunks);
    argmax_split_part_f32<<<dim3(nchunks, nrows), 256, 0, ctx.stream()>>>((const float *) src0->data, pv.get(), pi.get(), ne00, nchunks);
    argmax_split_final_f32<<<nrows, WARP_SIZE, 0, ctx.stream()>>>(pv.get(), pi.get(), (float *) dst->data, nchunks, n_dev, rank, offset, partial);
}
