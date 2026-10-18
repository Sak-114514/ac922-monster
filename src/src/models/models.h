#pragma once

#include "llama-model.h"
#include "llama-graph.h"
#include "llama-model-loader.h"

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>
#include <map>

class llama_memory_hybrid_idx_context;

// ref: https://github.com/ggml-org/llama.cpp/pull/28068
static inline ggml_tensor * build_gdn_l2_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    const float n = x->ne[0];

    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps/n), 1.0f/sqrtf(n));
}

//
// base classes
//

struct llm_build_mamba_base : public llm_graph_context {
    llm_build_mamba_base(const llm_graph_params & params);

    virtual ~llm_build_mamba_base() = default;

    ggml_tensor * build_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
    ggml_tensor * build_mamba2_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il) const;

};

struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                int           il);

    // use the ggml_gated_delta_net fused operator (K=1; state has shape [S_v, S_v, H_v, n_seqs])
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_fused(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // choose one of two implementations above based on the number of tokens
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // read conv state from cache, concat with qkv_mixed, write back (single slot or per-token)
    // qkv_mixed: (qkv_dim, n_seq_tokens, n_seqs); returns conv_input: (kernel_size + n_seq_tokens - 1, channels, n_seqs)
    ggml_tensor * build_conv_state(
            llm_graph_input_rs * inp,
            ggml_tensor *        conv_states_all,
            ggml_tensor *        qkv_mixed,
            int64_t              conv_kernel_size,
            int64_t              conv_channels,
            int                  il);

    // run delta-net attention and write the new recurrent state(s) back to ssm_states_all
    // s: (head_v_dim, head_v_dim, num_v_heads, n_seqs); returns output: (head_v_dim, num_v_heads, n_seq_tokens, n_seqs)
    ggml_tensor * build_recurrent_attn(
            llm_graph_input_rs * inp,
            ggml_tensor *        ssm_states_all,
            ggml_tensor *        q,
            ggml_tensor *        k,
            ggml_tensor *        v,
            ggml_tensor *        g,
            ggml_tensor *        b,
            ggml_tensor *        s,
            int                  il);
};

struct llm_build_rwkv6_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv6_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv6_base() = default;

    ggml_tensor * build_rwkv6_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;

    ggml_tensor * build_rwkv6_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

// Base class for RWKV7-related models
struct llm_build_rwkv7_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv7_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv7_base() = default;

    // RWKV7-specific graph building methods
    ggml_tensor * build_rwkv7_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;
    ggml_tensor * build_rwkv7_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       ggml_tensor *&       first_layer_value,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

//
// models
//





























































// Quant-only stub for mmproj GGUFs
// none of these are ever called, they only exist to satisfy the llama_model_base interface



























































































































































































































































































































































struct msa_params {
    int blk;
    int topk_blocks;
    int local;
};















struct llama_model_qwen4exp : public llama_model_base {
    llama_model_qwen4exp(const struct llama_model_params & params) : llama_model_base(params) {}

    class llm_graph_input_qsa;
    class llm_graph_input_qsa_blk;
    class llm_graph_input_qsa_mblk;

    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);
    protected:
        // tag-dispatched ctor for graph_mtp: binds the members without building the trunk
        struct no_build_t {};
        graph(const llama_model & model, const llm_graph_params & params, no_build_t) :
            llm_build_delta_net_base(params), model(model) {}

        // HC replaces every layer norm: residual is [n_embd, hc, n_tokens]
        ggml_tensor * build_hc_mix(
                    ggml_tensor * x,
                    ggml_tensor * w_norm,
                    ggml_tensor * w_down,
                    ggml_tensor * w_up,
                    ggml_tensor * w_inject,
                    ggml_tensor ** inject,
                            int   il);

        ggml_tensor * build_hc_combine(
                    ggml_tensor * residual,
                    ggml_tensor * block_out,
                    ggml_tensor * inject,
                            int   il);

        ggml_tensor * build_layer_attn(
              llm_graph_input_attn_kv * inp_attn,
  const llama_memory_hybrid_idx_context * mctx_hyb,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il,
                           bool   kv_only = false);

        // dense self-attention restricted to the cells that top_k names
        ggml_tensor * build_attn_qsa(
        llm_graph_input_attn_kv * inp,
                    ggml_tensor * q_cur,
                    ggml_tensor * k_cur,
                    ggml_tensor * v_cur,
                    ggml_tensor * top_k,
                          float   kq_scale,
                            int   il);

        // the QSA cache layout inputs do not depend on the layer, only on its compress ratio,
        // so the layers sharing a ratio share one input set
        std::map<uint32_t, llm_graph_input_qsa *> qsa_inps;

        // pooled block-key cache (see llama_memory_hybrid_idx::get_mem_blk): inputs per ratio,
        // the incremental update, and the fixed-cost decode selection + attention over whole blocks
        std::map<uint32_t, llm_graph_input_qsa_blk *> qsa_blk_inps;

        llm_graph_input_qsa_blk * get_qsa_blk_input(
  const llama_memory_hybrid_idx_context * mctx_hyb,
                       uint32_t   ratio,
                           bool   fast);

        ggml_tensor * build_qsa_blk_update(
  const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_qsa_blk * inp,
                            int * sections,
                            int   il);

        ggml_tensor * build_qsa_blk_select(
  const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_qsa_blk * inp,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        ggml_tensor * build_attn_qsa_blk(
        llm_graph_input_attn_kv * inp,
        llm_graph_input_qsa_blk * blk_inp,
                    ggml_tensor * q_cur,
                    ggml_tensor * k_cur,
                    ggml_tensor * v_cur,
                    ggml_tensor * top_blk,
                          float   kq_scale,
                            int   il);

        // per-sequence pooled block-key cache (LLAMA_QSA_POOLED_MULTI=1): the same roles as
        // the three above, with each sequence selecting over its own rows of the pooled cache
        // and gathering its own member cells, so sequences may share one cells array
        std::map<uint32_t, llm_graph_input_qsa_mblk *> qsa_mblk_inps;

        llm_graph_input_qsa_mblk * get_qsa_mblk_input(
  const llama_memory_hybrid_idx_context * mctx_hyb,
                       uint32_t   ratio,
                        int64_t   n_sel);

        ggml_tensor * build_qsa_mblk_update(
  const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_qsa_mblk * inp,
                            int * sections,
                            int   il);

        ggml_tensor * build_qsa_mblk_select(
  const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_qsa_mblk * inp,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                    int * sections,
                    int   il);

        ggml_tensor * build_attn_qsa_mblk(
        llm_graph_input_attn_kv * inp,
        llm_graph_input_qsa_mblk * blk_inp,
                    ggml_tensor * q_cur,
                    ggml_tensor * k_cur,
                    ggml_tensor * v_cur,
                    ggml_tensor * sel_cells,
                          float   kq_scale,
                            int   il);

        // device-bound owner for the fused CUDA hyper-connection kernels, chosen by build_hc_mix
        // from the layer's weights and reused by the build_hc_combine that closes the sublayer
        void * hc_owner = nullptr;

        // QSA: token indices this layer's queries may attend to, or nullptr for dense
        ggml_tensor * build_qsa_top_k(
  const llama_memory_hybrid_idx_context * mctx_hyb,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                    ggml_tensor * kq_mask,
                            int * sections,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // build_rs writes the state tensor in place, so one gather per cache tensor is reused
        std::map<ggml_tensor *, ggml_tensor *> rs_rows;

        // one conv history per cache tensor: delta-net and PLE each have their own
        ggml_tensor * build_conv_state_at(
             llm_graph_input_rs * inp,
                    ggml_tensor * conv_states_all,
                    ggml_tensor * x,
                        int64_t   state_cols,
                        int64_t   channels,
                            int   il);

        ggml_tensor * build_inp_ple(
  const llama_memory_hybrid_idx_context * mctx_hyb);

        ggml_tensor * build_ple(
             llm_graph_input_rs * inp,
                    ggml_tensor * emb,
                    ggml_tensor * hidden,
                            int   il);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const llama_model & model;
    };

    // LLM_GRAPH_TYPE_DECODER_MTP draft head: one HC-wrapped dense-attention + MoE block
    struct graph_mtp : public graph {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};



















