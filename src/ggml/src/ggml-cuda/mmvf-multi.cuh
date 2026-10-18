#include "common.cuh"

#define MMVF_MULTI_MAX_SEG 4

// one GEMV of a combined launch: weight w (F32 or BF16, [K, rows]), the node dst whose layout the output follows,
// out: where the output goes (dst->data, or the result of the fused activation), and the activation:
// 0 none, 1 sigmoid, 2 softplus(x + pre[row]) * mul[row]
struct ggml_cuda_mmvf_multi_seg {
    const ggml_tensor * w;
    const ggml_tensor * dst;
    void *              out;
    int                 post_op;
    const ggml_tensor * pre;
    const ggml_tensor * mul;
};

// all segments read the same activations src1 (F32, [K, 1..4], K*T*4 <= 40 KB); false (nothing launched) if the
// shapes do not fit
bool ggml_cuda_mmvf_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const ggml_cuda_mmvf_multi_seg * segs,
        int n_seg);
