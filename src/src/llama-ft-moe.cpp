#include "llama-ft-moe.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include <stdexcept>
#include <cstdlib>
#include <cstring>

llama_ft_moe::llama_ft_moe(const llama_model & model, int slots, int threads) {
    if (model.arch != LLM_ARCH_QWEN4EXP || model.devices.size() != 1) {
        throw std::runtime_error("FreeToken native MoE requires Qwen4Exp on one CUDA device");
    }
    auto dev = model.devices[0].dev;
    auto reg = ggml_backend_dev_backend_reg(dev);
    api = static_cast<const ggml_cuda_ft_api *>(ggml_backend_reg_get_proc_address(reg,"ggml_cuda_ft_api_v3"));
    if (!api) { throw std::runtime_error("CUDA backend has no FreeToken native MoE API"); }
    device = 0;
    for (; device < int(ggml_backend_reg_dev_count(reg)); ++device) {
        if (ggml_backend_reg_dev_get(reg,device) == dev) { break; }
    }
    std::vector<ggml_cuda_ft_layer> layers;
    for (const auto & l : model.layers) {
        layers.push_back({l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps,
                          l.ffn_gate_exps_s,l.ffn_up_exps_s,l.ffn_down_exps_s});
    }
    const char * dense = std::getenv("GGML_FT_DENSE_FP8");
    if (dense && std::strcmp(dense,"1") == 0) {
        dense_api = static_cast<const ggml_cuda_ft_dense_api *>(ggml_backend_reg_get_proc_address(reg,"ggml_cuda_ft_dense_api_v4"));
        if (!dense_api) { throw std::runtime_error("CUDA backend has no FreeToken FP8 dense API"); }
    }
    owner = api->create(device,slots,threads,model.hparams.n_expert_used(),layers.data(),layers.size());
    try {
        if (dense_api) {
            std::vector<ggml_tensor *> weights;
            for (const auto & entry : model.tensors_by_name) {
                const auto & name = entry.first;
                if (name.find("embd") == std::string::npos && name.find("conv") == std::string::npos &&
                    name.find("norm") == std::string::npos) { weights.push_back(entry.second); }
            }
            dense_owner = dense_api->create(device,weights.data(),weights.size());
        }
    } catch (...) { api->destroy(owner); owner = nullptr; throw; }
}
llama_ft_moe::llama_ft_moe(const llama_model & model, const llama_ft_moe * target, bool native_draft, bool dense_draft) {
    device = target->device;
    api = target->api;
    if (native_draft) {
        if (model.arch != LLM_ARCH_QWEN4EXP || model.hparams.n_layer_nextn != 1 || !api || !api->create_draft) {
            throw std::runtime_error("FreeToken NVFP4 MTP requires one Qwen4Exp draft layer and the native CUDA v2 API");
        }
        const int il = model.hparams.n_layer();
        const auto & l = model.layers.at(il);
        const ggml_cuda_ft_layer layer{l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps,
                                      l.ffn_gate_exps_s,l.ffn_up_exps_s,l.ffn_down_exps_s};
        owner = api->create_draft(device,model.hparams.n_expert_used(),il,&layer);
    }
    try {
        if (dense_draft && target->dense_owner) {
            dense_api = target->dense_api;
            dense_owner = dense_api->clone(target->dense_owner);
        }
    } catch (...) { if (owner) { api->destroy(owner); owner = nullptr; } throw; }
}
llama_ft_moe::~llama_ft_moe() { if (dense_owner) { dense_api->destroy(dense_owner); } if (owner) { api->destroy(owner); } }
ggml_tensor * llama_ft_moe::build(ggml_context * ctx, int layer, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * weights) const {
    return owner ? api->build(owner,ctx,layer,x,ids,weights) : nullptr;
}

ggml_tensor * llama_ft_moe::build_dense(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) const {
    return dense_owner ? dense_api->build(dense_owner,ctx,w,x) : nullptr;
}

ggml_tensor * llama_ft_moe::begin(ggml_context * ctx, ggml_tensor * moe) const {
    const char * enabled = std::getenv("GGML_FT_SHARED_OVERLAP");
    return owner && dense_owner && enabled && std::strcmp(enabled,"1") == 0 ? api->begin(ctx,moe) : nullptr;
}

ggml_tensor * llama_ft_moe::finish(ggml_context * ctx, ggml_tensor * start, ggml_tensor * dependency) const {
    return api->finish(ctx,start,dependency);
}

ggml_tensor * llama_ft_moe::build_hc(ggml_context * ctx, ggml_tensor * x, ggml_tensor * logits, int width) const {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_FT_HC_FUSION");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled && dense_owner ? dense_api->build_hc(dense_owner, ctx, x, logits, width) : nullptr;
}

ggml_tensor * llama_ft_moe::build_hc_combine(ggml_context * ctx, ggml_tensor * residual, ggml_tensor * block, ggml_tensor * inject) const {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_FT_HC_COMBINE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled && dense_owner ? dense_api->build_hc_combine(dense_owner, ctx, residual, block, inject) : nullptr;
}
