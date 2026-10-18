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
        // load the kernel now: a lazy load while a peer spins in the AllReduce can deadlock
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k_allreduce_p2p));
        const size_t n_flags = (size_t) n * P2P_AR_BLOCKS * P2P_AR_FLAG_STRIDE;
        CUDA_CHECK(cudaMalloc(&ar->flags[i], n_flags * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&ar->epoch[i], P2P_AR_BLOCKS * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(ar->flags[i], 0, n_flags * sizeof(unsigned)));
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
