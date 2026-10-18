#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(CUB_TOP_K_AVAILABLE)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

// Count contiguous column ranges so tie selection and output order do not depend on CTA arrival.
template<int BLOCK_SIZE>
static __global__ void top_k_stable_count(
        const float * src, const top_k_radix_state * states, int * counts,
        int ncols, int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int part = blockIdx.x % blocks_per_row;
    const int span = (ncols + blocks_per_row - 1) / blocks_per_row;
    const int end = min(ncols, (part + 1) * span);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const uint32_t threshold = states[row].prefix;
    int greater = 0, equal = 0;
    for (int col = part * span + threadIdx.x; col < end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(src[(size_t) row * ncols + col]);
        greater += key > threshold;
        equal += key == threshold;
    }
    for (int delta = 16; delta > 0; delta /= 2) {
        greater += __shfl_down_sync(0xffffffff, greater, delta);
        equal += __shfl_down_sync(0xffffffff, equal, delta);
    }
    __shared__ int warp_g[BLOCK_SIZE / 32], warp_e[BLOCK_SIZE / 32];
    if (lane == 0) { warp_g[warp] = greater; warp_e[warp] = equal; }
    __syncthreads();
    if (threadIdx.x == 0) {
        greater = equal = 0;
        for (int w = 0; w < BLOCK_SIZE / 32; ++w) { greater += warp_g[w]; equal += warp_e[w]; }
        counts[2 * blockIdx.x] = greater;
        counts[2 * blockIdx.x + 1] = equal;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_stable_gather(
        const float * src, int * dst, const top_k_radix_state * states, const int * counts,
        int ncols, int k, int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int part = blockIdx.x % blocks_per_row;
    const int span = (ncols + blocks_per_row - 1) / blocks_per_row;
    const int end = min(ncols, (part + 1) * span);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const uint32_t threshold = states[row].prefix;
    const int rank = states[row].rank;
    __shared__ int base_g, base_e, warp_g[BLOCK_SIZE / 32], warp_e[BLOCK_SIZE / 32];
    if (threadIdx.x == 0) {
        base_g = base_e = 0;
        for (int p = 0; p < part; ++p) {
            base_g += counts[2 * (row * blocks_per_row + p)];
            base_e += counts[2 * (row * blocks_per_row + p) + 1];
        }
    }
    __syncthreads();
    for (int tile = part * span; tile < end; tile += BLOCK_SIZE) {
        const int col = tile + threadIdx.x;
        const uint32_t key = col < end ? top_k_float_to_ordered(src[(size_t) row * ncols + col]) : 0;
        const bool greater = col < end && key > threshold;
        const bool equal = col < end && key == threshold;
        const unsigned gm = __ballot_sync(0xffffffff, greater);
        const unsigned em = __ballot_sync(0xffffffff, equal);
        if (lane == 0) { warp_g[warp] = __popc(gm); warp_e[warp] = __popc(em); }
        __syncthreads();
        int gp = base_g + __popc(gm & ((1u << lane) - 1));
        int ep = base_e + __popc(em & ((1u << lane) - 1));
        for (int w = 0; w < warp; ++w) { gp += warp_g[w]; ep += warp_e[w]; }
        if (greater) dst[(size_t) row * k + gp] = col;
        if (equal && ep < rank) dst[(size_t) row * k + k - rank + ep] = col;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < BLOCK_SIZE / 32; ++w) { base_g += warp_g[w]; base_e += warp_e[w]; }
        }
        __syncthreads();
    }
}

// One launch per radix digit: every block histograms its slice of the row, and the last block of
// the row to finish sums the slices and picks the digit (what top_k_radix_select does), so the select
// needs no launch of its own. The first pass starts from the initial state and the last pass clears
// the gather counters, which removes the init and reset launches too: 5 launches instead of 11,
// same selection. Row counters come from a zeroed per-device buffer and are reset after use.
template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram_select(
        const float * __restrict__ src,
        top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        unsigned int * __restrict__ counters,
        int ncols,
        int blocks_per_row,
        int shift,
        int k,
        bool first_pass,
        bool last_pass) {
    constexpr int NBINS = 1 << RADIX_BITS;
    static_assert(NBINS == BLOCK_SIZE, "one bin per thread");

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];
    __shared__ bool is_last;

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state0 = first_pass ? top_k_radix_state{0, 0, k, 0, 0} : states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state0.prefix_mask) == state0.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t row_offset = (size_t) row * blocks_per_row * NBINS;
    block_histograms[row_offset + (size_t) row_block * NBINS + tid] = histogram[tid];
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        is_last = atomicAdd(&counters[row], 1u) == (unsigned int) blocks_per_row - 1;
    }
    __syncthreads();
    if (!is_last) {
        return;
    }

    int count = 0;
    for (int rb = 0; rb < blocks_per_row; ++rb) {
        count += __ldcg(&block_histograms[row_offset + (size_t) rb * NBINS + tid]);
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = state0;
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        if (last_pass) {
            state.greater_count = 0;
            state.equal_count = 0;
        }
        states[row] = state;
        counters[row] = 0;
    }
}

static unsigned int * top_k_radix_counters(const int nrows, cudaStream_t stream) {
    constexpr int n_counters = 1 << 14;
    static unsigned int * buf[GGML_CUDA_MAX_DEVICES] = {};
    static int next[GGML_CUDA_MAX_DEVICES] = {};
    const int dev = ggml_cuda_get_device();
    GGML_ASSERT(nrows <= n_counters);
    if (buf[dev] == nullptr) {
        CUDA_CHECK(cudaMalloc(&buf[dev], n_counters*sizeof(unsigned int)));
        // stream-ordered, so a first use inside graph capture stays legal
        CUDA_CHECK(cudaMemsetAsync(buf[dev], 0, n_counters*sizeof(unsigned int), stream));
    }
    if (next[dev] + nrows > n_counters) {
        next[dev] = 0;
    }
    unsigned int * p = buf[dev] + next[dev];
    next[dev] += nrows;
    return p;
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    const dim3 row_grid(blocks_per_row * nrows);

    static const bool stable = [] { const char * s = getenv("GGML_CUDA_TOPK_STABLE"); return s && atoi(s) != 0; }();
    auto gather = [&] {
        if (stable) {
            top_k_stable_count<BLOCK_SIZE><<<row_grid, BLOCK_SIZE, 0, stream>>>(src, states, histograms, ncols, blocks_per_row);
            top_k_stable_gather<BLOCK_SIZE><<<row_grid, BLOCK_SIZE, 0, stream>>>(src, dst, states, histograms, ncols, k, blocks_per_row);
        } else {
            top_k_radix_gather<BLOCK_SIZE><<<row_grid, BLOCK_SIZE, 0, stream>>>(src, dst, states, ncols, k, blocks_per_row);
        }
    };
    static const bool fused = !getenv("GGML_CUDA_TOPK_UNFUSED");
    if (fused) {
        unsigned int * counters = top_k_radix_counters(nrows, stream);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram_select<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, counters, ncols, blocks_per_row, shift, k,
                    shift == 32 - RADIX_BITS, shift == 0);
        }
        gather();
        return;
    }

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    gather();
}

#endif // !defined(CUB_TOP_K_AVAILABLE)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // long rows: radix select reads each row a few times instead of fully sorting it, which
    // matters for top-k over tens of thousands of scores per row (block-sparse attention).
    // like DeviceTopK above, the k indices come back unordered
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
