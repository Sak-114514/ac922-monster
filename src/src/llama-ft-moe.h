#pragma once
#include "ggml-cuda-ft.h"
struct llama_model;
class llama_ft_moe {
public:
    llama_ft_moe(const llama_model & model, int slots, int threads);
    llama_ft_moe(const llama_model & model, const llama_ft_moe * target, bool native_draft, bool dense_draft);
    bool has_dense() const { return dense_owner != nullptr; }
    ~llama_ft_moe();
    llama_ft_moe(const llama_ft_moe &) = delete;
    llama_ft_moe & operator=(const llama_ft_moe &) = delete;
    ggml_tensor * build(ggml_context * ctx, int layer, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * weights) const;
    ggml_tensor * begin(ggml_context * ctx, ggml_tensor * moe) const;
    ggml_tensor * finish(ggml_context * ctx, ggml_tensor * start, ggml_tensor * dependency) const;
    ggml_tensor * build_dense(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) const;
    ggml_tensor * build_hc(ggml_context * ctx, ggml_tensor * x, ggml_tensor * logits, int width) const;
    ggml_tensor * build_hc_combine(ggml_context * ctx, ggml_tensor * residual, ggml_tensor * block, ggml_tensor * inject) const;
private:
    int device = 0;
    const ggml_cuda_ft_dense_api * dense_api = nullptr;
    void * dense_owner = nullptr;
    const ggml_cuda_ft_api * api = nullptr;
    void * owner = nullptr;
};
