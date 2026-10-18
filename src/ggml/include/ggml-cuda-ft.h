#pragma once
#include "ggml.h"

// Private, versioned interface obtained through the CUDA backend registry.
struct ggml_cuda_ft_layer {
    struct ggml_tensor * gate;
    struct ggml_tensor * up;
    struct ggml_tensor * down;
    struct ggml_tensor * gate_scale;
    struct ggml_tensor * up_scale;
    struct ggml_tensor * down_scale;
};
struct ggml_cuda_ft_api {
    void * (*create)(int device, int slots, int threads, int top_k,
                     const struct ggml_cuda_ft_layer * layers, int n_layers);
    void (*destroy)(void * owner);
    struct ggml_tensor * (*build)(void * owner, struct ggml_context * ctx, int layer,
                                  struct ggml_tensor * x, struct ggml_tensor * ids,
                                  struct ggml_tensor * weights);
    // v2: one BF16 draft layer, quantized at load into resident NVFP4 banks.
    void * (*create_draft)(int device, int top_k, int layer, const struct ggml_cuda_ft_layer * weights);
    // v3: run independent GPU work between routed expert submission and completion.
    struct ggml_tensor * (*begin)(struct ggml_context * ctx, struct ggml_tensor * moe);
    struct ggml_tensor * (*finish)(struct ggml_context * ctx, struct ggml_tensor * start, struct ggml_tensor * dependency);
};

struct ggml_cuda_ft_dense_api {
    void * (*create)(int device, struct ggml_tensor * const * weights, int count);
    void (*destroy)(void * owner);
    struct ggml_tensor * (*build)(void * owner, struct ggml_context * ctx,
                                  struct ggml_tensor * weight, struct ggml_tensor * x);
    void * (*clone)(void * owner);
    // v3: four-stream sigmoid gating and ordered mean reduction.
    struct ggml_tensor * (*build_hc)(void * owner, struct ggml_context * ctx,
                                    struct ggml_tensor * x, struct ggml_tensor * logits, int width);
    // v4: four-stream injection and residual addition.
    struct ggml_tensor * (*build_hc_combine)(void * owner, struct ggml_context * ctx,
        struct ggml_tensor * residual, struct ggml_tensor * block, struct ggml_tensor * inject);
};
