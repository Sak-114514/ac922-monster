#include "allreduce-p2p.cuh"

// Fixed grid: every call uses all blocks, so the per-block call counters stay equal and all blocks of a call
// use the same half of the double-buffered staging area.
#define P2P_AR_BLOCKS  8
#define P2P_AR_THREADS 512
// one flag per 64-byte line
#define P2P_AR_FLAG_STRIDE 16
#define P2P_AR_MAX_RANKS 8

// Layout on every rank:
//   stage: [2 halves][n ranks][cap floats]  slot s of a half is written by rank s
//   flags: [n ranks][P2P_AR_BLOCKS][P2P_AR_FLAG_STRIDE]  row s is written by rank s
// A call pushes the local partial into slot `rank` of every peer's stage, raises one flag per peer, waits for
// the flags of all peers and sums all slots in rank order, so every rank gets bit-identical results.
// Double buffering is enough: a rank can only start call k+2 on a half after all peers passed call k+1,
// which they do only after they finished reading that half in call k.

// Large tensors (>= GGML_CUDA_P2P_AR_RING_MIN_BYTES bytes, default 4 MiB) instead run the segmented kernel
// k_allreduce_p2p_seg below: a reduce-by-owner phase plus an allgather phase over 1/n chunks, which cuts the
// per-rank egress from (n-1)*V to 2*(n-1)/n*V (~2V). It shares `stage`, the double-buffer halves and the
// per-block epoch counters with the one-shot kernel, but has its OWN flag array (flags_seg) so the two flag
// value schemes never alias: the one-shot kernel always writes cell 0 of a line, while the segmented kernel
// uses cell `round` (2n <= P2P_AR_FLAG_STRIDE) and every cell holds the same per-call epoch value. Both
// kernels bump the epoch exactly once per call, so the call k+2-vs-call k double-buffer invariant above holds
// even when a graph mixes one-shot and segmented AllReduce nodes.

// op params of an ALLREDUCE node; cap is included so a regrown staging area changes the node and the
// captured CUDA graph is not reused with stale pointers
struct p2p_ar_op_params {
    ggml_cuda_p2p_ar * ar;
    int64_t            cap;
    int32_t            rank;
    int32_t            zero; // the local partial was not computed (zero-sized slice): add 0 instead
};
static_assert(sizeof(p2p_ar_op_params) <= GGML_MAX_OP_PARAMS, "p2p_ar_op_params too large");

struct p2p_ar_args {
    float    * stage[P2P_AR_MAX_RANKS]; // stage of each rank (peer memory except [rank])
    unsigned * flags[P2P_AR_MAX_RANKS]; // flags of each rank (peer memory except [rank])
    unsigned * epoch;                   // local call counter per block
    int64_t    cap;                     // floats per slot
    int        rank;
    int        n;
    int        zero;
    int64_t    max_spins; // GGML_CUDA_P2P_AR_DEBUG: report a stuck call and return instead of trapping
    int        id;        // GGML_CUDA_P2P_AR_DEBUG: index of the node in its graph
};

struct ggml_cuda_p2p_ar {
    int        n = 0;
    int        devices[P2P_AR_MAX_RANKS];
    float    * stage[P2P_AR_MAX_RANKS] = {};
    unsigned * flags[P2P_AR_MAX_RANKS] = {};
    unsigned * flags_seg[P2P_AR_MAX_RANKS] = {}; // separate flag cells for k_allreduce_p2p_seg, see above
    unsigned * epoch[P2P_AR_MAX_RANKS] = {};
    int64_t    cap = 0;
};

static __global__ void k_allreduce_p2p(float * data, const int64_t n_el, const p2p_ar_args p) {
    const int b = blockIdx.x;

    __shared__ unsigned ep;
    if (threadIdx.x == 0) {
        ep = p.epoch[b] + 1;
        p.epoch[b] = ep;
    }
    __syncthreads();

    const int64_t per  = (n_el + gridDim.x - 1) / gridDim.x;
    const int64_t i0   = b * per;
    const int64_t i1   = min(n_el, i0 + per);
    const int64_t half = (int64_t) (ep & 1) * p.n * p.cap;

    const int64_t my_slot = half + p.rank * p.cap;
    for (int64_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        const float v = p.zero ? 0.0f : data[i];
        for (int r = 0; r < p.n; ++r) {
            if (r != p.rank) {
                p.stage[r][my_slot + i] = v;
            }
        }
    }
    __syncthreads();

    // thread r signals peer r and waits for it
    const int r = threadIdx.x;
    if (r < p.n && r != p.rank) {
        __threadfence_system();
        *(volatile unsigned *) &p.flags[r][(p.rank * P2P_AR_BLOCKS + b) * P2P_AR_FLAG_STRIDE] = ep;
        const volatile unsigned * mine = &p.flags[p.rank][(r * P2P_AR_BLOCKS + b) * P2P_AR_FLAG_STRIDE];
        int64_t spins = 0;
        while (*mine < ep) {
            if (++spins > p.max_spins) {
                if (p.max_spins == (int64_t(1) << 28)) {
                    __trap(); // peer never arrived: fail loudly instead of hanging
                }
                printf("p2p allreduce stuck: rank %d block %d call %u node %d waits for rank %d at %u\n", p.rank, b, ep, p.id, r, *mine);
                break;
            }
        }
    }
    __syncthreads();

    const float * mine = p.stage[p.rank] + half;
    for (int64_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < p.n; ++s) {
            const float v = s != p.rank ? __ldcv(mine + s * p.cap + i) : p.zero ? 0.0f : data[i];
            acc = s == 0 ? v : acc + v;
        }
        data[i] = acc;
    }
}

// Segmented (ring-style) AllReduce for large tensors. The tensor is cut into n contiguous segments, one per
// rank, and reduced in 2n rounds:
//   phase A, rounds s = 0..n-1 (reduce by owner): every rank r != s pushes segment s of its partial into
//     rank s's staging slot r; rank s then sums its own segment in rank order and stores the sum in data.
//   phase B, rounds n+s (allgather): rank s pushes its reduced segment s into every peer's staging slot s;
//     every other rank copies it over its local segment s.
// Per-rank egress drops from (n-1)*V (one-shot, all-pairs) to 2*(n-1)/n*V, about 2V.
//
// Bit-consistency argument: in the one-shot kernel every element is summed as
//   acc = 0; for r = 0..n-1: v_r = (r == rank) ? (zero ? 0 : data[i]) : stage_slot_r[i]; acc = r == 0 ? v_r : acc + v_r
// i.e. exactly the additions v_0 + v_1 + ... + v_{n-1} left to right, where v_r is rank r's partial (or the
// 0 that a non-computed slice pushes). Here the element's segment is owned by rank s, and phase A round s
// performs the identical sequence on the owner: for r = 0..n-1, v_r read from slot r for r != s and from the
// local data (or 0 when zero) for r == s, accumulated with acc = r == 0 ? v_r : acc + v_r. Phase B then only
// COPIES the owner's bytes to the other ranks, without any arithmetic. Every rank therefore ends up with a
// result bit-identical to the one-shot kernel: the same additions, in the same order, executed segment-wise.
//
// Synchronization reuses the stage/flag double buffering of the one-shot kernel: the half is still (ep & 1)
// and the shared per-block epoch bumps exactly once per call, so mixing one-shot and segmented nodes on one
// rank keeps the "call k+2 cannot overtake a peer still reading call k" invariant. Flags live in the separate
// flags_seg array: round `round` uses flag cell `round` (2n <= P2P_AR_FLAG_STRIDE, n <= P2P_AR_MAX_RANKS) of
// the same line the one-shot kernel would use, and the cell again holds the epoch of the call, so per-cell
// values stay monotonic exactly like the one-shot flags. Every round is a full barrier: every block of every
// rank raises its flag for all peers and waits for all peers before the round's read phase.
static __global__ void k_allreduce_p2p_seg(float * data, const int64_t n_el, const p2p_ar_args p) {
    const int b = blockIdx.x;

    __shared__ unsigned ep;
    if (threadIdx.x == 0) {
        ep = p.epoch[b] + 1;
        p.epoch[b] = ep;
    }
    __syncthreads();

    const int     n    = p.n;
    const int64_t half = (int64_t) (ep & 1) * n * p.cap;
    const int64_t step = (int64_t) gridDim.x * blockDim.x; // all blocks cooperate on one segment per round

    for (int round = 0; round < 2 * n; ++round) {
        const int     s    = round < n ? round : round - n; // rank that owns the segment of this round
        const int64_t s_lo = n_el * s / n;
        const int64_t s_hi = n_el * (s + 1) / n;

        if (round < n) {
            // phase A: push segment s of the local partial into the owner's stage, our slot
            if (s != p.rank) {
                float * dst = p.stage[s] + half + p.rank * p.cap;
                for (int64_t i = s_lo + (int64_t) b * blockDim.x + threadIdx.x; i < s_hi; i += step) {
                    dst[i] = p.zero ? 0.0f : data[i];
                }
            }
        } else {
            // phase B: the owner broadcasts its reduced segment into every peer's stage, slot s
            if (s == p.rank) {
                for (int q = 0; q < n; ++q) {
                    if (q == p.rank) {
                        continue;
                    }
                    float * dst = p.stage[q] + half + s * p.cap;
                    for (int64_t i = s_lo + (int64_t) b * blockDim.x + threadIdx.x; i < s_hi; i += step) {
                        dst[i] = data[i];
                    }
                }
            }
        }
        __syncthreads();

        // thread r signals peer r and waits for it (same pattern as the one-shot kernel, cell = round)
        const int r = threadIdx.x;
        if (r < n && r != p.rank) {
            __threadfence_system();
            *(volatile unsigned *) &p.flags[r][(p.rank * P2P_AR_BLOCKS + b) * P2P_AR_FLAG_STRIDE + round] = ep;
            const volatile unsigned * mine = &p.flags[p.rank][(r * P2P_AR_BLOCKS + b) * P2P_AR_FLAG_STRIDE + round];
            int64_t spins = 0;
            while (*mine < ep) {
                if (++spins > p.max_spins) {
                    if (p.max_spins == (int64_t(1) << 28)) {
                        __trap(); // peer never arrived: fail loudly instead of hanging
                    }
                    printf("p2p allreduce (seg) stuck: rank %d block %d call %u round %d node %d waits for rank %d at %u\n", p.rank, b, ep, round, p.id, r, *mine);
                    break;
                }
            }
        }
        __syncthreads();

        if (round < n) {
            if (s == p.rank) {
                // the owner sums its segment in rank order: the same adds in the same order as one-shot
                const float * mine = p.stage[p.rank] + half;
                for (int64_t i = s_lo + (int64_t) b * blockDim.x + threadIdx.x; i < s_hi; i += step) {
                    float acc = 0.0f;
                    for (int q = 0; q < n; ++q) {
                        const float v = q != p.rank ? __ldcv(mine + q * p.cap + i) : p.zero ? 0.0f : data[i];
                        acc = q == 0 ? v : acc + v;
                    }
                    data[i] = acc;
                }
            }
        } else {
            if (s != p.rank) {
                // copy the owner's reduced segment over the local one: pure copy, no arithmetic
                const float * mine = p.stage[p.rank] + half + s * p.cap;
                for (int64_t i = s_lo + (int64_t) b * blockDim.x + threadIdx.x; i < s_hi; i += step) {
                    data[i] = __ldcv(mine + i);
                }
            }
        }
        __syncthreads();
    }
}

// AllReduce tensors of at least this many bytes run the segmented kernel instead of one-shot all-pairs.
// Bucketing is per launch and depends only on the tensor size, which is identical on every rank, so every
// rank picks the same algorithm for a given node. Set this very large (e.g. 1152921504606846976 = 1<<60) to
// keep every tensor on the one-shot path - a one-env rollback of the whole change. Parsed once per process.
static int64_t p2p_ar_ring_min_bytes() {
    static const int64_t v = [] {
        const char * e = getenv("GGML_CUDA_P2P_AR_RING_MIN_BYTES");
        const int64_t parsed = e && *e ? (int64_t) atoll(e) : (int64_t) 4 << 20; // default 4 MiB
        return parsed < 0 ? (int64_t) 1 << 60 : parsed;
    }();
    return v;
}

void ggml_cuda_op_allreduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    GGML_ASSERT(dst->data == dst->src[0]->data);

    p2p_ar_op_params op;
    memcpy(&op, dst->op_params, sizeof(op));
    const ggml_cuda_p2p_ar * ar = op.ar;
    const int64_t n_el = ggml_nelements(dst);
    GGML_ASSERT(n_el <= ar->cap);

    p2p_ar_args p = {};
    for (int r = 0; r < ar->n; ++r) {
        p.stage[r] = ar->stage[r];
        p.flags[r] = ar->flags[r];
    }
    p.epoch = ar->epoch[op.rank];
    p.cap   = ar->cap;
    p.rank  = op.rank;
    p.n     = ar->n;
    p.zero  = op.zero;
    static const bool debug = getenv("GGML_CUDA_P2P_AR_DEBUG") != nullptr;
    p.max_spins = debug ? int64_t(1) << 24 : int64_t(1) << 28;
    p.id        = atoi(dst->name + strlen("allreduce_"));
    if (debug) {
        GGML_LOG_WARN("p2p allreduce launch: rank %d %s n=%lld zero=%d stream=%p\n", op.rank, dst->name, (long long) n_el, op.zero, (void *) ctx.stream());
    }

    // size bucketing: small (decode-sized) tensors stay on the unchanged one-shot path; large (prefill)
    // tensors take the segmented kernel, whose egress is ~2V instead of (n-1)*V per rank
    if (n_el * (int64_t) sizeof(float) >= p2p_ar_ring_min_bytes()) {
        p2p_ar_args ps = p;
        for (int r = 0; r < ar->n; ++r) {
            ps.flags[r] = ar->flags_seg[r]; // segmented rounds use their own flag cells, never one-shot's
        }
        if (debug) {
            GGML_LOG_WARN("p2p allreduce launch (seg): rank %d %s n=%lld zero=%d stream=%p\n", op.rank, dst->name, (long long) n_el, op.zero, (void *) ctx.stream());
        }
        k_allreduce_p2p_seg<<<P2P_AR_BLOCKS, P2P_AR_THREADS, 0, ctx.stream()>>>((float *) dst->data, n_el, ps);
        return;
    }

    k_allreduce_p2p<<<P2P_AR_BLOCKS, P2P_AR_THREADS, 0, ctx.stream()>>>((float *) dst->data, n_el, p);
}

ggml_cuda_p2p_ar * ggml_cuda_p2p_ar_init(const int * devices, size_t n_devices) {
    if (n_devices < 2 || n_devices > P2P_AR_MAX_RANKS || getenv("GGML_CUDA_P2P_AR_DISABLE") != nullptr) {
        return nullptr;
    }
    // every pair needs peer access: each rank writes into every peer
    for (size_t i = 0; i < n_devices; ++i) {
        for (size_t j = 0; j < n_devices; ++j) {
            if (i == j) {
                continue;
            }
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, devices[i], devices[j]));
            if (!can) {
                GGML_LOG_INFO("%s: no peer access from device %d to %d, in-graph AllReduce off\n", __func__, devices[i], devices[j]);
                return nullptr;
            }
        }
    }
    const int n = (int) n_devices;
    ggml_cuda_p2p_ar * ar = new ggml_cuda_p2p_ar;
    ar->n = n;
    for (int i = 0; i < n; ++i) {
        ar->devices[i] = devices[i];
        ggml_cuda_set_device(devices[i]);
        for (int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const cudaError_t err = cudaDeviceEnablePeerAccess(devices[j], 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                CUDA_CHECK(err);
            }
        }
        (void) cudaGetLastError();
        // load the kernels now: a lazy load while a peer spins in the AllReduce can deadlock
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p_seg));
        const size_t n_flags = (size_t) n * P2P_AR_BLOCKS * P2P_AR_FLAG_STRIDE;
        CUDA_CHECK(cudaMalloc(&ar->flags[i], n_flags * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&ar->flags_seg[i], n_flags * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&ar->epoch[i], P2P_AR_BLOCKS * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(ar->flags[i], 0, n_flags * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(ar->flags_seg[i], 0, n_flags * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(ar->epoch[i], 0, P2P_AR_BLOCKS * sizeof(unsigned)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    GGML_LOG_INFO("%s: in-graph P2P AllReduce on %d devices\n", __func__, n);
    return ar;
}

void ggml_cuda_p2p_ar_free(ggml_cuda_p2p_ar * ar) {
    if (ar == nullptr) {
        return;
    }
    for (int i = 0; i < ar->n; ++i) {
        ggml_cuda_set_device(ar->devices[i]);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(ar->stage[i]));
        CUDA_CHECK(cudaFree(ar->flags[i]));
        CUDA_CHECK(cudaFree(ar->flags_seg[i]));
        CUDA_CHECK(cudaFree(ar->epoch[i]));
    }
    delete ar;
}

bool ggml_cuda_p2p_ar_reserve(ggml_cuda_p2p_ar * ar, size_t nbytes) {
    const int64_t cap = (int64_t) ((nbytes + sizeof(float) - 1) / sizeof(float));
    if (cap <= ar->cap) {
        return true;
    }
    // growing moves the staging buffers: no call may be in flight
    for (int i = 0; i < ar->n; ++i) {
        ggml_cuda_set_device(ar->devices[i]);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (int i = 0; i < ar->n; ++i) {
        ggml_cuda_set_device(ar->devices[i]);
        CUDA_CHECK(cudaFree(ar->stage[i]));
        CUDA_CHECK(cudaMalloc(&ar->stage[i], 2 * (size_t) ar->n * cap * sizeof(float)));
    }
    ar->cap = cap;
    return true;
}

void ggml_cuda_p2p_ar_set_params(ggml_cuda_p2p_ar * ar, size_t rank, ggml_tensor * node) {
    GGML_ASSERT(rank < (size_t) ar->n);
    p2p_ar_op_params op = {};
    op.ar   = ar;
    op.cap  = ar->cap;
    op.rank = (int32_t) rank;
    op.zero = (node->src[0]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
    memset(node->op_params, 0, sizeof(node->op_params));
    memcpy(node->op_params, &op, sizeof(op));
}
