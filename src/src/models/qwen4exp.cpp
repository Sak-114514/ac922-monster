#include "models.h"
#include "llama-impl.h"
#include "llama-ft-moe.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"
#include "qwen4exp-mtp-prefill-policy.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <cinttypes>

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    if (hparams.n_layer_nextn > 1 || hparams.n_layer_nextn >= hparams.n_layer_all) {
        throw std::runtime_error("qwen4exp requires at most one MTP block after the trunk");
    }
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it
    // the gammas load as [n_embd, hc] so the grouped norm multiplies them without a graph reshape
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, TENSOR_ALLOW_RESHAPE);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, 0);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
    }

    // MTP tensors sit in the trailing blocks; skip them entirely unless a draft head was asked for
    const int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        auto & layer = layers[il];

        // the MTP block is structurally a trunk block: is_recr()/is_ple() are both false past
        // the trunk, so it takes the full-attention + MoE path below with no special casing
        const int flags = il < n_layer ? 0 : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (il < n_layer) {
            continue;
        }

        // NextN/MTP head. enorm/hnorm gate the two inputs; eh_proj is the checkpoint's
        // fc_embedding and fc_hidden fused side by side, so one matmul over
        // concat(e, h) computes fc_embedding@e + fc_hidden@h.
        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);

        // the head's own output mixer, mirroring the trunk's hc_head_*: it collapses the
        // hc streams and stands in for the output norm, of which qwen4exp has none
        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags);
        layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags);

        // qwen4exp sets mtp_use_dedicated_embeddings=false, so these are absent and the
        // head falls back to the trunk's embedding table and LM head
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// LLAMA_QWEN4EXP_HC_FUSION=1: run the hyper-connection mix and combine as the fused CUDA kernels
// of the FreeToken dense API even without the native MoE path, which needs a single device. Each
// CUDA device gets one weightless owner; the kernels only need to know which device they run on.
static const ggml_cuda_ft_dense_api * qwen4exp_hc_api = nullptr;

static void * qwen4exp_hc_owner(const ggml_tensor * w) {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_QWEN4EXP_HC_FUSION");
        return env && atoi(env) != 0;
    }();
    if (!enabled || w == nullptr || w->buffer == nullptr) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(w->buffer));
    if (dev == nullptr) {
        return nullptr;
    }
    // called for every mix of every graph build: look the device up before anything that may
    // query the driver (the CUDA device type getter calls cudaGetDeviceProperties)
    static std::mutex mutex;
    static std::map<ggml_backend_dev_t, void *> owners;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = owners.find(dev);
    if (it != owners.end()) {
        return it->second;
    }
    void * owner = nullptr;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const auto * api = ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ? nullptr :
        static_cast<const ggml_cuda_ft_dense_api *>(ggml_backend_reg_get_proc_address(reg, "ggml_cuda_ft_dense_api_v4"));
    if (api != nullptr) {
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
            if (ggml_backend_reg_dev_get(reg, i) == dev) {
                owner = api->create((int) i, nullptr, 0);
                qwen4exp_hc_api = api;
                break;
            }
        }
    }
    owners.emplace(dev, owner);
    return owner;
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // LLAMA_HC_FUSED=2: the whole mix as two kernels (ggml_hc_mix_down/up), for up to 8 tokens. Slower than the
    //     int8 GEMV path for now (f32 activations, few blocks), so only 1 (fused combine) is on by default.
    static const bool hc_fused_mix = [] { const char * e = getenv("LLAMA_HC_FUSED"); return e && atoi(e) >= 2; }();
    if (hc_fused_mix && !ft_moe && loras->empty() && nt <= 8 && ggml_is_contiguous(x) && n_embd % 512 == 0 &&
            w_down->type == GGML_TYPE_Q8_0 && w_up->type == GGML_TYPE_Q8_0 && w_norm->type == GGML_TYPE_F32 &&
            (!w_inject || w_inject->type == GGML_TYPE_F32)) {
        ggml_tensor * down = ggml_hc_mix_down(ctx0, x, w_norm, w_down, inject ? w_inject : nullptr, hparams.f_norm_rms_eps);
        cb(down, "hc_down", il);
        ggml_tensor * mixed = ggml_hc_mix_up(ctx0, x, w_norm, w_up, down);
        cb(mixed, "hc_mixed", il);
        if (inject) {
            *inject = down; // build_hc_combine reads the inject rows of this layout
        }
        return mixed;
    }

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = build_lora_mm(w_up, lo);
    cb(gate, "hc_gate", il);

    ggml_tensor * mixed = ft_moe && !cparams.warmup && loras->empty()
        ? ft_moe->build_hc(ctx0, xn, gate, n_embd) : nullptr;
    hc_owner = !ft_moe && loras->empty() ? qwen4exp_hc_owner(w_down) : nullptr;
    if (!mixed && hc_owner) {
        mixed = qwen4exp_hc_api->build_hc(hc_owner, ctx0, xn, gate, n_embd);
    }
    if (!mixed && cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
    } else if (!mixed) {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // LLAMA_HC_FUSED=1: the combine as one kernel
    static const bool hc_fused_combine = [] { const char * e = getenv("LLAMA_HC_FUSED"); return e && atoi(e) >= 1; }();
    if (hc_fused_combine && il >= 0 && nt <= 8 && ggml_is_contiguous(residual) && ggml_is_contiguous(block_out) &&
            ggml_is_contiguous(inject) && inject->type == GGML_TYPE_F32) {
        ggml_tensor * cur = ggml_hc_combine(ctx0, block_out, residual, inject);
        cb(cur, "hc_combine", il);
        return cur;
    }
    if (inject->ne[0] != hc) {
        // inject is the ggml_hc_mix_down layout: [lr, inject (hc), 1/rms (hc)] per token
        if (ggml_is_contiguous(residual) && ggml_is_contiguous(block_out) && ggml_is_contiguous(inject)) {
            ggml_tensor * cur = ggml_hc_combine(ctx0, block_out, residual, inject);
            cb(cur, "hc_combine", il);
            return cur;
        }
        const int64_t lr = inject->ne[0] - 2*hc;
        inject = ggml_cont(ctx0, ggml_view_2d(ctx0, inject, hc, inject->ne[1], inject->nb[1], lr*ggml_element_size(inject)));
    }

    ggml_tensor * cur = ft_moe && !cparams.warmup && loras->empty()
        ? ft_moe->build_hc_combine(ctx0, residual, block_out, inject) : nullptr;
    if (!cur && hc_owner) {
        cur = qwen4exp_hc_api->build_hc_combine(hc_owner, ctx0, residual, block_out, inject);
    }
    if (!cur) {
        // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
        ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
        w = ggml_scale(ctx0, w, 2.0f);

        if (cparams.fused_dsv4_hc_post && il >= 0) {
            // identity comb: every stream adds the same block output, scaled by its own weight
            cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
            res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
        } else {
            w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

            ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
            b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

            cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
        }
    }
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    // tensor split: a view of the CPU embedding can not be split, use a real copy
    ggml_tensor * inp_3d = model.split_mode() == LLAMA_SPLIT_MODE_TENSOR
            ? ggml_cont_3d(ctx0, inpL, n_embd, 1, n_tokens)
            : ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0, inp_3d, n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        // an unmasked MTP export needs a hidden row for every token, so in that case the
        // gather is deferred until after t_h_nextn is taken below
        const bool gather_now = !cparams.embeddings_nextn || cparams.embeddings_nextn_masked;

        if (il == n_layer - 1 && inp_out_ids && gather_now) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // The MTP head consumes the wide residual, before the head mixer collapses it. Export the
    // combine result itself rather than a reshape of it: a pure view gets no backend assignment
    // from the scheduler, and the readback in llama_context looks one up. It is contiguous, so
    // [n_embd, hc, rows] already has the [n_embd_out, rows] layout the reader expects, and it
    // carries exactly the right rows either way -- gathered above when masked, ungathered when not.
    if (cparams.embeddings_nextn) {
        cb(res_hc, "h_nextn", -1);
        res->t_h_nextn = res_hc;

        // deferred from the last layer: collapse to the output rows now that the export is taken
        if (!cparams.embeddings_nextn_masked && inp_out_ids) {
            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// The draft consumes and returns the wide HC residual.
static bool qwen4exp_mtp_kv_only_prefill_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_MTP_KV_ONLY_PREFILL");
        return env && atoi(env) != 0;
    }();
    return enabled;
}

llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN4EXP MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input");

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj     && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm       && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm       && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.hc_head_norm && "MTP block missing nextn.hc_head_norm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hc_dim);

    inp->tokens = mtp_token_input ? mtp_token_input : ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp->keep_tokens = mtp_token_input != nullptr;
    if (!mtp_token_input) ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->embd);

    inp->h_fb_rows = mtp_h_fb_rows;
    if (mtp_h_input && mtp_h_rows) {
        // Gather each sequence's carry row.
        inp->h_carry   = mtp_h_input;
        inp->h_rows    = mtp_h_rows;
        inp->row_keep  = mtp_h_row_keep;
        inp->h = ggml_get_rows(ctx0, mtp_h_input, ggml_view_1d(ctx0, mtp_h_rows, n_tokens, 0));
    } else {
        inp->h = mtp_h_input ? mtp_h_input : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
        inp->keep_h = mtp_h_input ? mtp_h_keep : nullptr;
        if (!mtp_h_input) ggml_set_input(inp->h);
    }
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = mtp_embedding ? mtp_embedding : layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd   = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    const bool has_gpu_chain_inputs = mtp_h_input || mtp_token_input || mtp_embedding;
    const bool kv_only_prefill = qwen4exp_mtp::kv_only_prefill_eligible(
            qwen4exp_mtp_kv_only_prefill_enabled(), cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP,
            hparams.n_layer_nextn, n_outputs, n_tokens,
            hparams.dsv4_compress_ratios[il], has_gpu_chain_inputs,
            cparams.embeddings, cparams.embeddings_nextn, cparams.embeddings_nextn_masked);

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = kv_only_prefill ? nullptr : build_inp_out_ids();

    auto * inp_mem = build_inp_mem_hybrid();
    inp_mem->uses_recr = false; // the MTP block is an attention layer
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp_mem->mctx);
    auto * inp_attn = inp_mem->get_attn();

    if (kv_only_prefill) {
        static std::atomic<uint64_t> activation_count{0};
        const uint64_t activation = activation_count.fetch_add(1, std::memory_order_relaxed) + 1;
        fprintf(stderr, "mtp-kv-only-prefill activation=%llu tokens=%lld outputs=%u layer=%d\n",
                (unsigned long long) activation, (long long) n_tokens, (unsigned) n_outputs, il);
    }

    // grouped RMSNorm over the wide stream: normalise each hc stream, then scale with the
    // head's [n_embd, hc] gamma, exactly as build_hc_mix does
    ggml_tensor * h_norm = ggml_mul(ctx0, ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps), layer.nextn.hnorm);
    cb(h_norm, "mtp_hnorm", il);

    // the token embedding is shared across the streams, so broadcast it to hc copies
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // eh_proj holds fc_embedding and fc_hidden side by side, so this one matmul is
    // fc_embedding @ e_norm + fc_hidden @ h_norm, applied to each stream independently.
    // Keeping the streams distinct here is the point of the hyper-connection residual:
    // pooling them before the projection would throw that away.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    // as one [2*n_embd, hc*n_tokens] matrix: a 3-D src1 is multiplied per token and re-reads the weight every time
    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj,
            ggml_reshape_2d(ctx0, concat, concat->ne[0], concat->ne[1]*concat->ne[2]), layer.nextn.eh_proj_s);
    res_hc = ggml_reshape_3d(ctx0, res_hc, res_hc->ne[0], concat->ne[1], concat->ne[2]);
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    cur = build_layer_attn(inp_attn, mctx_hyb, cur, inp_pos, sections, il, kv_only_prefill);

    // Outputless catch-up calls only need to populate this MTP layer's K/V cache.
    // Returning here avoids the attention query, HC post-attention mix, FFN, head mix,
    // and vocabulary projection; no caller consumes draft logits or hidden outputs.
    if (kv_only_prefill) {
        return;
    }

    if (inp_out_ids) {
        cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    // ---- MoE, identical to the trunk's build_layer_ffn ----
    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
            &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    // The next draft step re-enters here, so export the wide stream before it is collapsed.
    // As in the trunk, export the combine result rather than a reshape view of it.
    cb(res_hc, "h_nextn", -1);
    res->t_h_nextn = res_hc;

    // the head's own mixer collapses the streams and doubles as the output norm
    cur = build_hc_mix(res_hc,
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // deliberately no res->t_embd: it would be n_embd wide while the context sizes its
    // embedding buffer by n_embd_out (the wide stream). The driver reads t_h_nextn instead.

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN4EXP MTP: missing LM head (nextn.shared_head_head or model.output)");

    if (cparams.draft_detp && !layer.nextn.shared_head_head) {
        // de-TP draft: use the full single-device LM-head replica instead of the vocab-split
        // output weight, so the head matmul stays on the draft device with whole-vocab logits
        GGML_ASSERT(model.output_detp && "QWEN4EXP MTP: de-TP active but no LM-head replica");
        head_w = model.output_detp;
        head_s = model.output_detp_s;
    }

    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
    ggml_tensor * hidden_output = mtp_h_output ? mtp_h_output : mtp_h_input;
    if (hidden_output && mtp_h_fb_rows) {
        // Scatter feedback to each sequence's carry row.
        GGML_ASSERT(n_outputs > 0 && ggml_nelements(res_hc) == hc_dim * n_outputs);
        GGML_ASSERT(ggml_nelements(hidden_output) >= hc_dim * n_outputs);
        auto * rows = ggml_reshape_2d(ctx0, res_hc, hc_dim, n_outputs);
        auto * fb_indices = ggml_view_1d(ctx0, mtp_h_fb_rows, n_outputs, 0);
        auto * feedback = ggml_set_rows(ctx0, hidden_output, rows, fb_indices);
        ggml_set_name(feedback, "mtp_gpu_feedback_per_seq");
        ggml_build_forward_expand(gf, feedback);
        res->mtp_feedback = true;
    } else if (hidden_output) {
        GGML_ASSERT(n_outputs > 0 && ggml_nelements(res_hc) == hc_dim * n_outputs);
        GGML_ASSERT(ggml_nelements(hidden_output) == hc_dim * n_outputs ||
                (n_outputs == 1 && ggml_nelements(hidden_output) % hc_dim == 0));
        auto * rows = ggml_reshape_2d(ctx0, res_hc, hc_dim, n_outputs);
        auto * feedback = rows;
        if (ggml_nelements(hidden_output) != ggml_nelements(rows)) {
            GGML_ASSERT(n_outputs == 1);
            feedback = ggml_repeat(ctx0, rows, hidden_output);
        }
        feedback = ggml_cpy(ctx0, feedback, hidden_output);
        ggml_set_name(feedback, "mtp_gpu_feedback");
        ggml_build_forward_expand(gf, feedback);
        res->mtp_feedback = true;
    }
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// Block-sparse attention kernel (CUDA, Q8_0 K/V, head size 256): each query reads its selected
// blocks straight from the cache. It lets the fast path take prompt batches too.
// LLAMA_QSA_SPARSE_KERNEL=0 keeps the gather + flash attention path.
static bool qwen4exp_qsa_sparse_kernel_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_QSA_SPARSE_KERNEL");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

// Opt in to reading cell-selected multi-sequence QSA rows directly from the Q8 cache.
static bool qwen4exp_qsa_mblk_sparse_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_QSA_MULTI_SPARSE");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

static constexpr int32_t QWEN4EXP_QSA_ID_MODE_CELLS = 0x51534143;

// LLAMA_PLE_ONECAT=1 selects the fused SM70 PLE convolution path.  The
// default keeps the original graph chain for rollback and parity fallback.
static bool qwen4exp_ple_onecat_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_PLE_ONECAT");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

// GGML_CUDA_QSA_PP_REDUCE=1: opt-in fusion marker for the Qwen4Exp four-head
// score reduction used during prompt processing.  The CUDA matcher is private
// to the CUDA backend; the generic graph remains valid when the flag is off.
static bool qwen4exp_qsa_pp_reduce_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_QSA_PP_REDUCE");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

static constexpr int32_t QWEN4EXP_QSA_PP_MARKER = 0x51535050; // "QSPP"

// LLAMA_QSA_GATHER_MAX_TOKENS: opt-in query-count limit for the ordinary
// top-k gather + Flash Attention path. The default preserves the historical
// 16-query guard; opt-in values are capped at 256 queries.
static int qwen4exp_qsa_gather_max_tokens() {
    static const int max_tokens = [] {
        const char * env = getenv("LLAMA_QSA_GATHER_MAX_TOKENS");
        if (env == nullptr) {
            return 16;
        }

        char * end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end == env || *end != '\0' || value < 1 || value > 256) {
            return 16;
        }

        return (int) value;
    }();
    return max_tokens;
}

static void qwen4exp_qsa_path_trace(
        int64_t n_tok,
        int64_t width,
        const ggml_tensor * top_k,
        const ggml_tensor * k_all,
        const ggml_tensor * kq_mask,
        int64_t width_pad,
        bool flash_attn,
        bool v_trans,
        bool force_dense,
        bool chosen_gather) {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_QSA_PATH_TRACE");
        return env != nullptr && atoi(env) != 0;
    }();
    if (!enabled) {
        return;
    }

    const bool token_limit = n_tok <= qwen4exp_qsa_gather_max_tokens();
    const bool shape_ok = top_k->ne[3] == 1 && k_all->ne[3] == 1 &&
        kq_mask->ne[0] == k_all->ne[2] && kq_mask->ne[1] == n_tok && kq_mask->ne[3] == 1 &&
        k_all->ne[2] >= 2*width_pad;
    const std::string key = format("tok=%" PRId64 ":limit=%d:shape=%d:flash=%d:vtrans=%d:dense=%d:chosen=%d",
                                   n_tok, token_limit, shape_ok, flash_attn, v_trans, force_dense, chosen_gather);

    static std::mutex mutex;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lock(mutex);
    if (!seen.insert(key).second) {
        return;
    }

    fprintf(stderr,
            "qsa-path: top_k=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
            "K=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
            "mask=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
            "width=%" PRId64 " width_pad=%" PRId64 " kvlen=%" PRId64 " "
            "flash=%d vtrans=%d dense=%d token_limit=%" PRId64 " chosen=%s\n",
            top_k->ne[0], top_k->ne[1], top_k->ne[2], top_k->ne[3],
            k_all->ne[0], k_all->ne[1], k_all->ne[2], k_all->ne[3],
            kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3],
            width, width_pad, k_all->ne[2], flash_attn, v_trans, force_dense,
            (int64_t) qwen4exp_qsa_gather_max_tokens(), chosen_gather ? "gather" : "masked");
}

// Trace for the multi pooled fast path (which has no ordinary top_k, so qsa-path never
// sees it). Prints once per distinct shape: the shape stream is exactly the CUDA-graph
// variant stream, so this doubles as thrash evidence. Layer-calls/24 = decode steps.
static void qwen4exp_qsa_mblk_trace(
        int64_t n_tok,
        int64_t n_touch,
        int64_t n_rows,
        int64_t n_sel,
        int64_t width) {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_QSA_PATH_TRACE");
        return env != nullptr && atoi(env) != 0;
    }();
    if (!enabled) {
        return;
    }

    const std::string key = format("nt=%" PRId64 ":touch=%" PRId64 ":rows=%" PRId64 ":width=%" PRId64 ":cellsel=%d",
                                   n_tok, n_touch, n_rows, width,
                                   getenv("LLAMA_QSA_MULTI_CELLSEL") != nullptr && atoi(getenv("LLAMA_QSA_MULTI_CELLSEL")) != 0);

    static std::mutex mutex;
    static std::set<std::string> seen;
    static uint64_t calls = 0;

    std::lock_guard<std::mutex> lock(mutex);
    ++calls;

    if (!seen.insert(key).second) {
        return;
    }

    fprintf(stderr,
            "qsa-mblk: shape #%zu (layer-call #%llu, ~%llu steps): nt=%" PRId64 " touch=%" PRId64
            " rows=%" PRId64 " n_sel=%" PRId64 " width=%" PRId64 " cellsel=%d\n",
            seen.size(), (unsigned long long) calls, (unsigned long long) (calls/24),
            n_tok, n_touch, n_rows, n_sel, width,
            getenv("LLAMA_QSA_MULTI_CELLSEL") != nullptr && atoi(getenv("LLAMA_QSA_MULTI_CELLSEL")) != 0);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
static bool qwen4exp_qsa_blk_fast(
        const llama_memory_hybrid_idx_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        const llama_hparams & hparams,
        uint32_t ratio);

// multi-sequence pooled path (LLAMA_QSA_POOLED_MULTI=1): each sequence selects over its own
// rows of the pooled cache, so the whole fast path is layout-agnostic about the shared cells
static bool qwen4exp_qsa_mblk_fast(
        const llama_memory_hybrid_idx_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        const llama_hparams & hparams,
        uint32_t ratio);

class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);

        // multi mode folds every ubatch into the per-sequence block tables, whichever path
        // ends up consuming them, so a slow step cannot leave the tables behind the cells
        mctx->qsa_multi_register(k_idxs, ubatch);

        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias, cell_order);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        // the pooled block cache adds its update to this graph, or replaces this path outright
        const bool blk_now = mctx->blk_ok();
        if (blk_now != had_blk || (blk_now && qwen4exp_qsa_blk_fast(mctx, params.ubatch, *cparams, *hparams, ratio))) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const int64_t n_blocks = (n_kv + ratio - 1)/ratio;

        bool res = true;

        res &= params.ubatch.n_tokens % n_stream == 0;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= cell_blk->ne[0]  == n_kv;
        res &= cell_blk->ne[1]  == n_stream;
        if (cell_order) {
            res &= cell_order->ne[0] == n_kv && cell_order->ne[1] == n_stream;
        }
        res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= blk_pos->ne[0]   == 4*n_blocks*n_stream;
        res &= bias->ne[0]      == (blk_bias ? n_blocks : n_kv);
        res &= bias->ne[1]      == params.ubatch.n_tokens/n_stream;

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_order = nullptr;  // I32 [n_kv, n_stream], logical-position order
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;

    // whether the graph also updates the pooled block cache, and what decides the fast path
    bool had_blk = false;
    const llama_cparams * cparams = nullptr;
    const llama_hparams * hparams = nullptr;
};

// Fixed-cost decode path over the pooled block-key cache. Taken for small causal text batches
// once every query sees more complete blocks than the budget: each query then attends to its
// indexer_top_k/ratio best whole blocks plus its incomplete tail block, the reference selection.
static bool qwen4exp_qsa_blk_fast(
        const llama_memory_hybrid_idx_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        const llama_hparams & hparams,
        uint32_t ratio) {
    // Large gather requests use the reference per-query top-k selection, not
    // the decode-only pooled whole-block approximation. Keep decode unchanged.
    if (ubatch.n_tokens > 64 && ubatch.n_tokens <= (uint32_t) qwen4exp_qsa_gather_max_tokens()) {
        return false;
    }
    static const bool disabled = [] {
        const char * env = getenv("LLAMA_QSA_DENSE");
        return env && atoi(env) != 0;
    }();
    if (disabled || ratio == 0 || !mctx->blk_ok() || !cparams.causal_attn || !cparams.flash_attn || hparams.use_alibi) {
        return false;
    }
    // the gather path materializes every query's selected rows: small batches only
    const uint32_t max_tokens = qwen4exp_qsa_sparse_kernel_enabled() ? UINT32_MAX : 16;
    if (ubatch.n_tokens == 0 || ubatch.n_tokens > max_tokens || mctx->get_n_stream() != 1) {
        return false;
    }
    // blocks cut the cell array (blk_ok: cell = rank), so the first query's cell says how many complete blocks it has
    const int64_t c0    = mctx->get_idx()->ubatch_cell0();
    const int64_t n_sel = hparams.indexer_top_k / ratio;
    return (c0 + 1) / ratio >= n_sel + 1;
}

class llama_model_qwen4exp::llm_graph_input_qsa_blk : public llm_graph_input_i {
public:
    llm_graph_input_qsa_blk(const llama_memory_hybrid_idx_context * mctx, const llama_cparams & cparams,
            const llama_hparams & hparams, uint32_t ratio, bool fast) :
        mctx(mctx), cparams(cparams), hparams(hparams), ratio(ratio), fast(fast) {}
    virtual ~llm_graph_input_qsa_blk() = default;

    static int64_t n_touch(int64_t n_tokens, uint32_t ratio) {
        return (n_tokens + ratio - 1)/ratio + 1;
    }

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t r = ratio;
        const int64_t n = ubatch->n_tokens;

        // blk_ok() holds: token i sits in cell c0 + i, and blocks cut the cell array
        const auto *  idx = mctx->get_idx();
        const int64_t c0  = idx->ubatch_cell0();

        // every block holding a token of this ubatch is re-pooled; the rest of the rows pad
        // into the scratch row
        const int64_t nbt     = touch_src->ne[0];
        const int64_t b0      = c0/r;
        const int64_t b1      = (c0 + n - 1)/r;
        const int32_t scratch = (int32_t) mctx->get_blk()->get_size() - 1;

        GGML_ASSERT(ggml_backend_buffer_is_host(touch_src->buffer));
        int32_t * src = (int32_t *) touch_src->data;
        int32_t * dst = (int32_t *) touch_dst->data;
        int32_t * pos = (int32_t *) touch_pos->data;
        for (int64_t k = 0; k < nbt; ++k) {
            const int64_t b  = b0 + k;
            const bool    ok = b <= b1;
            src[k] = ok ? (int32_t) b : 0;
            dst[k] = ok ? (int32_t) b : scratch;
            // a pooled key is rotated to the position of its first token: b*r for text, (t, y, x) for an image
            int32_t sec_pos[4] = { 0, 0, 0, 0 };
            if (ok) {
                idx->cell_rope_pos((uint32_t) (b*r), sec_pos);
            }
            for (int64_t sec = 0; sec < 4; ++sec) {
                pos[sec*nbt + k] = sec_pos[sec];
            }
        }

        if (!fast) {
            return;
        }

        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);

        const int64_t n_sel  = hparams.indexer_top_k/r;
        const int64_t n_tail = tail_ids->ne[0];          // the tail block, then padding blocks
        const int64_t width  = sel_mask->ne[0];          // (n_sel + n_tail)*r cells per query

        const ggml_fp16_t zero    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);

        float       * v  = (float       *) vis->data;
        int32_t     * tb = (int32_t     *) tail_ids->data;
        ggml_fp16_t * m  = (ggml_fp16_t *) sel_mask->data;
        for (int64_t i = 0; i < n; ++i) {
            // complete blocks up to the query's cell are visible; the rest of its own
            // block, up to and including the query, is the always-visible tail
            const int64_t q   = c0 + i;
            const int64_t c   = (q + 1)/r;
            const int64_t rem = (q + 1)%r;
            v[i] = (float) c;
            for (int64_t j = 0; j < n_tail; ++j) {
                tb[i*n_tail + j] = rem ? (int32_t) c : 0;
            }
            // selected blocks are complete and visible, the tail is visible up to the query,
            // the padding blocks are off
            ggml_fp16_t * mi = m + i*width;
            for (int64_t cell = 0; cell < width; ++cell) {
                const int64_t t = cell - n_sel*r;
                mi[cell] = t < 0 ? zero : (t < rem ? zero : neg_inf);
            }
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        bool res = mctx->get_idx() != nullptr && mctx->blk_ok();
        res = res && qwen4exp_qsa_blk_fast(mctx, params.ubatch, cparams, hparams, ratio) == fast;
        res = res && touch_src->ne[0] == n_touch(params.ubatch.n_tokens, ratio);
        if (res && fast) {
            res = vis->ne[1] == params.ubatch.n_tokens && k_idxs->ne[0] == params.ubatch.n_tokens &&
                  sel_mask->ne[3] == params.ubatch.n_tokens;
        }
        return res;
    }

    ggml_tensor * touch_src = nullptr;  // I32 [n_touch]    block rows pooled from the indexer cache
    ggml_tensor * touch_dst = nullptr;  // I32 [n_touch]    rows written in the pooled cache
    ggml_tensor * touch_pos = nullptr;  // I32 [4*n_touch]  rope positions of those blocks

    // fast selection only
    ggml_tensor * k_idxs    = nullptr;  // I32 [n_tokens]         indexer cache cells of this ubatch
    ggml_tensor * vis       = nullptr;  // F32 [1, n_tokens]      complete blocks each query sees
    ggml_tensor * tail_ids  = nullptr;  // I32 [n_tail, n_tokens]           tail block, repeated as padding
    ggml_tensor * sel_mask  = nullptr;  // F16 [width, 1, 1, n_tokens]      attention mask over the gathered cells

    // graph node, not an input: the visibility bias built from vis, shared by the layers
    ggml_tensor * vis_bias  = nullptr;

    const llama_memory_hybrid_idx_context * mctx;
    const llama_cparams & cparams;
    const llama_hparams & hparams;
    const uint32_t ratio;
    const bool fast;
};

// Multi-sequence fast path over the per-sequence pooled rows (LLAMA_QSA_POOLED_MULTI=1).
// The gather + flash-attention consumption matches the single-sequence path's, but blocks
// and cells are named explicitly (row_cells / member_cells / tail_ids), so sequences may
// interleave freely in the shared cells array. The block-sparse CUDA kernel is not used:
// it assumes cell == rank, which only one sequence in one stream can promise.

// cell-granular selection (LLAMA_QSA_MULTI_CELLSEL=1): the selection top-k runs on CELLS
// (row scores repeated r-wide, k = n_sel*r + r - 1) so the attended set matches the
// per-step rebuild path exactly, up to ties and fp order. Same env as the memory side.
static bool qwen4exp_qsa_mblk_cellsel() {
    static const bool enabled = [] { const char * env = getenv("LLAMA_QSA_MULTI_CELLSEL"); return env != nullptr && atoi(env) != 0; }();
    return enabled;
}

// demotion tracing for this gate's own conditions (LLAMA_QSA_BLK_DEBUG=1); the table-side
// conditions log inside llama_memory_hybrid_idx::qsa_multi_fast_ok
static void qwen4exp_qsa_mblk_demote(const char * why, uint32_t n_tokens) {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_QSA_BLK_DEBUG");
        return env != nullptr && atoi(env) != 0;
    }();
    if (!enabled) {
        return;
    }

    static std::mutex mutex;
    static std::vector<std::pair<std::string, uint64_t>> counts;

    std::lock_guard<std::mutex> lock(mutex);

    uint64_t n = 0;

    for (auto & pc : counts) {
        if (pc.first == why) {
            n = pc.second++;
            if (n == 0 || n == 1000 || n == 100000) {
                fprintf(stderr, "qsa-blk: multi fast path demoted (%s), occurrence #%llu, nt=%u\n",
                        why, (unsigned long long) (n + 1), n_tokens);
            }
            return;
        }
    }

    counts.emplace_back(why, 1);

    fprintf(stderr, "qsa-blk: multi fast path demoted (%s), occurrence #1, nt=%u\n", why, n_tokens);
}

static bool qwen4exp_qsa_mblk_fast(
        const llama_memory_hybrid_idx_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        const llama_hparams & hparams,
        uint32_t ratio) {
    // Large gather requests use the reference per-query top-k selection, not
    // the decode-only pooled whole-block approximation. Keep decode unchanged.
    if (ubatch.n_tokens > 64 && ubatch.n_tokens <= (uint32_t) qwen4exp_qsa_gather_max_tokens()) {
        return false;
    }
    static const bool disabled = [] {
        const char * env = getenv("LLAMA_QSA_DENSE");
        return env && atoi(env) != 0;
    }();
    if (disabled || ratio == 0 || !mctx->blk_multi() || !cparams.causal_attn || !cparams.flash_attn || hparams.use_alibi) {
        qwen4exp_qsa_mblk_demote(disabled          ? "dense-env" :
                                 ratio == 0        ? "ratio0" :
                                 !mctx->blk_multi()? "no-multi-cache" :
                                 !cparams.causal_attn ? "non-causal" :
                                 !cparams.flash_attn  ? "no-flash" : "alibi", ubatch.n_tokens);
        return false;
    }
    // per-query streams of gathered cells, as the ordinary top-k gather path allows
    if (ubatch.n_tokens == 0) {
        return false;
    }
    if (ubatch.n_tokens > qwen4exp_qsa_gather_max_tokens()) {
        qwen4exp_qsa_mblk_demote("token-limit", ubatch.n_tokens);
        return false;
    }
    // every sequence must own more complete blocks than the budget (checked per sequence,
    // against each one's own registered prefix), with tables that match its cells
    return mctx->qsa_multi_fast_ok(ubatch, ratio, hparams.indexer_top_k / ratio);
}

class llama_model_qwen4exp::llm_graph_input_qsa_mblk : public llm_graph_input_i {
public:
    llm_graph_input_qsa_mblk(const llama_memory_hybrid_idx_context * mctx, const llama_cparams & cparams,
            const llama_hparams & hparams, uint32_t ratio, int64_t n_sel) :
        mctx(mctx), cparams(cparams), hparams(hparams), ratio(ratio), n_sel(n_sel) {}
    virtual ~llm_graph_input_qsa_mblk() = default;

    void set_input(const llama_ubatch * ubatch) override {
        llama_qsa_multi_plan plan;

        // the shapes were fixed at build time from the same tables this reads; the
        // registration in between must not move them (see qsa_multi_sizes)
        mctx->qsa_multi_sizes(*ubatch, ratio, n_sel, plan.n_touch, plan.n_rows, plan.width, plan.width_pad);
        GGML_ASSERT(plan.width == plan.width_pad);

        // row mode: n_sel whole rows + a 256-wide tail/pad block; cell mode: the same
        // top (n_sel*ratio + ratio - 1) cells the per-step rebuild path picks
        const int64_t k_sel = qwen4exp_qsa_mblk_cellsel() ? (int64_t) n_sel*ratio + ratio - 1 : (int64_t) n_sel*ratio;

        GGML_ASSERT(ggml_backend_buffer_is_host(member_ids->buffer));
        GGML_ASSERT(member_ids->ne[0] == (int64_t) ratio*plan.n_touch);
        GGML_ASSERT(touch_dst->ne[0] == plan.n_touch);
        GGML_ASSERT(touch_pos->ne[0] == 4*plan.n_touch);
        GGML_ASSERT(c_rows->ne[0] == plan.n_rows);
        GGML_ASSERT(row_cells->ne[0] == (int64_t) ratio && row_cells->ne[1] == plan.n_rows);
        GGML_ASSERT(vis2->ne[1] == (int64_t) ubatch->n_tokens);
        GGML_ASSERT(tail_ids->ne[0] == plan.width_pad - k_sel);
        GGML_ASSERT(tail_ids->ne[1] == (int64_t) ubatch->n_tokens);
        GGML_ASSERT(sel_mask->ne[0] == plan.width_pad && sel_mask->ne[3] == (int64_t) ubatch->n_tokens);

        // cache this ubatch's raw indexer keys, fold the tokens into the tables, then fill
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->qsa_multi_register(k_idxs, ubatch);
        mctx->qsa_multi_fill(plan, ubatch, ratio, n_sel);

        memcpy(member_ids->data, plan.member_cells.data(), plan.member_cells.size()*sizeof(int32_t));
        memcpy(touch_dst->data,  plan.touch_dst.data(),    plan.touch_dst.size()*sizeof(int32_t));
        memcpy(touch_pos->data,  plan.touch_pos.data(),    plan.touch_pos.size()*sizeof(int32_t));
        memcpy(c_rows->data,     plan.c_rows.data(),       plan.c_rows.size()*sizeof(int32_t));
        memcpy(row_cells->data,  plan.row_cells.data(),    plan.row_cells.size()*sizeof(int32_t));
        memcpy(vis2->data,       plan.vis2.data(),         plan.vis2.size()*sizeof(float));
        memcpy(tail_ids->data,   plan.tail_ids.data(),     plan.tail_ids.size()*sizeof(int32_t));
        memcpy(sel_mask->data,   plan.sel_mask.data(),     plan.sel_mask.size()*sizeof(ggml_fp16_t));
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        if (!qwen4exp_qsa_mblk_fast(mctx, params.ubatch, cparams, hparams, ratio)) {
            return false;
        }

        int64_t n_touch = 0, n_rows = 0, width = 0, width_pad = 0;
        mctx->qsa_multi_sizes(params.ubatch, ratio, n_sel, n_touch, n_rows, width, width_pad);

        const int64_t k_sel = qwen4exp_qsa_mblk_cellsel() ? (int64_t) n_sel*ratio + ratio - 1 : (int64_t) n_sel*ratio;

        bool res = true;

        res &= k_idxs->ne[0]    == (int64_t) params.ubatch.n_tokens;
        res &= member_ids->ne[0] == (int64_t) ratio*n_touch;
        res &= touch_dst->ne[0]  == n_touch;
        res &= c_rows->ne[0]     == n_rows;
        res &= row_cells->ne[0]  == (int64_t) ratio && row_cells->ne[1] == n_rows;
        res &= vis2->ne[1]       == (int64_t) params.ubatch.n_tokens;
        res &= tail_ids->ne[0]   == width_pad - k_sel;
        res &= tail_ids->ne[1]   == (int64_t) params.ubatch.n_tokens;
        res &= sel_mask->ne[0]   == width_pad;
        res &= sel_mask->ne[3]   == (int64_t) params.ubatch.n_tokens;

        return res;
    }

    ggml_tensor * k_idxs     = nullptr;   // I64 [n_tokens]             indexer cells of this ubatch
    ggml_tensor * member_ids = nullptr;   // I32 [ratio*n_touch]        cells of the re-pooled buckets
    ggml_tensor * touch_dst  = nullptr;   // I32 [n_touch]              pooled rows they land in (global)
    ggml_tensor * touch_pos  = nullptr;   // I32 [4*n_touch]            rope positions of those buckets
    ggml_tensor * c_rows     = nullptr;   // I32 [n_rows]               compact scoring row -> pooled row (global)
    ggml_tensor * row_cells  = nullptr;   // I32 [ratio, n_rows]        member cells of every compact scoring row
    ggml_tensor * vis2       = nullptr;   // F32 [2, n_tokens]          per query [slice start, slice end) in the
                                          //                             selection space (rows, or cells in cell mode)
    ggml_tensor * tail_ids   = nullptr;   // I32 [tail_w, n_tokens]     row mode: tail cells, padded; cell mode: zero pad ids
    ggml_tensor * sel_mask   = nullptr;   // F16 [width_pad, 1, 1, n_tokens] mask over the gathered cells

    const llama_memory_hybrid_idx_context * mctx;
    const llama_cparams & cparams;
    const llama_hparams & hparams;
    const uint32_t ratio;
    const int64_t n_sel;
};

llama_model_qwen4exp::llm_graph_input_qsa_blk * llama_model_qwen4exp::graph::get_qsa_blk_input(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        uint32_t                                ratio,
        bool                                    fast) {
    const auto it = qsa_blk_inps.find(ratio);
    if (it != qsa_blk_inps.end()) {
        GGML_ASSERT(it->second->fast == fast);
        return it->second;
    }

    auto inp = std::make_unique<llm_graph_input_qsa_blk>(mctx_hyb, cparams, hparams, ratio, fast);

    const int64_t nbt = llm_graph_input_qsa_blk::n_touch(n_tokens, ratio);

    inp->touch_src = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, nbt);
    inp->touch_dst = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, nbt);
    inp->touch_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*nbt);
    ggml_set_input(inp->touch_src);
    ggml_set_input(inp->touch_dst);
    ggml_set_input(inp->touch_pos);

    if (fast) {
        inp->k_idxs    = mctx_hyb->get_idx()->build_input_k_idxs(ctx0, ubatch);
        inp->vis       = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        // the selected blocks plus the tail block, padded to a FATTN_KQ_STRIDE multiple of cells
        // so the vector FA kernels stay eligible; the padding repeats the tail block, masked off
        const int64_t n_sel  = hparams.indexer_top_k/ratio;
        const int64_t n_wblk = GGML_PAD((n_sel + 1)*ratio, 256)/ratio;

        inp->tail_ids  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_wblk - n_sel, n_tokens);
        inp->sel_mask  = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_wblk*ratio, 1, 1, n_tokens);
        ggml_set_input(inp->vis);
        ggml_set_input(inp->tail_ids);
        ggml_set_input(inp->sel_mask);
    }

    auto * res_inp = inp.get();
    res->add_input(std::move(inp));
    qsa_blk_inps.emplace(ratio, res_inp);

    return res_inp;
}

llama_model_qwen4exp::llm_graph_input_qsa_mblk * llama_model_qwen4exp::graph::get_qsa_mblk_input(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        uint32_t                                ratio,
        int64_t                                 n_sel) {
    const auto it = qsa_mblk_inps.find(ratio);
    if (it != qsa_mblk_inps.end()) {
        return it->second;
    }

    auto inp = std::make_unique<llm_graph_input_qsa_mblk>(mctx_hyb, cparams, hparams, ratio, n_sel);

    int64_t n_touch = 0, n_rows = 0, width = 0, width_pad = 0;
    mctx_hyb->qsa_multi_sizes(ubatch, ratio, n_sel, n_touch, n_rows, width, width_pad);
    GGML_ASSERT(width == width_pad);

    const int64_t r      = ratio;
    // row mode pads the tail block up to the 256-quantized width; cell mode pads the
    // (n_sel*r + r - 1) selected cells to the same width — 2304 either way
    const int64_t k_sel  = qwen4exp_qsa_mblk_cellsel() ? n_sel*r + r - 1 : n_sel*r;
    const int64_t tail_w = width_pad - k_sel;
    const int64_t nt     = n_tokens;

    inp->k_idxs     = mctx_hyb->get_idx()->build_input_k_idxs(ctx0, ubatch);
    inp->member_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r*n_touch);
    inp->touch_dst  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_touch);
    inp->touch_pos  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_touch);
    inp->c_rows     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rows);
    inp->row_cells  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r, n_rows);
    inp->vis2       = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, nt);
    inp->tail_ids   = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, tail_w, nt);
    inp->sel_mask   = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, width_pad, 1, 1, nt);
    ggml_set_input(inp->k_idxs);
    ggml_set_input(inp->member_ids);
    ggml_set_input(inp->touch_dst);
    ggml_set_input(inp->touch_pos);
    ggml_set_input(inp->c_rows);
    ggml_set_input(inp->row_cells);
    ggml_set_input(inp->vis2);
    ggml_set_input(inp->tail_ids);
    ggml_set_input(inp->sel_mask);

    auto * res_inp = inp.get();
    res->add_input(std::move(inp));
    qsa_mblk_inps.emplace(ratio, res_inp);

    return res_inp;
}

// Re-pool the blocks this ubatch touched into the pooled cache: the same mean, norm and
// rotation build_qsa_top_k applies to every block, applied to these rows only. The members
// of block b are cells b*ratio.. (cell = rank, see blk_ok), so they are one row of a block view.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_blk_update(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_qsa_blk *               inp,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t r       = inp->ratio;
    const int64_t n_kv    = mctx_idx->get_n_kv();

    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    ggml_tensor * k_blk = ggml_view_2d(ctx0, k_all, idx_dim*r, n_kv/r, r*k_all->nb[2], 0);

    ggml_tensor * members = ggml_get_rows(ctx0, k_blk, inp->touch_src);
    const int64_t nbt = members->ne[1];
    members = ggml_reshape_3d(ctx0, members, idx_dim, r, nbt);

    // start the sum from two strided views: no separate copy of the first slice
    const auto member = [&](int64_t i) {
        return ggml_view_2d(ctx0, members, idx_dim, nbt, members->nb[2], i*members->nb[1]);
    };
    ggml_tensor * pooled = r > 1 ? ggml_add(ctx0, member(0), member(1)) : ggml_cont(ctx0, member(0));
    for (int64_t i = 2; i < r; ++i) {
        pooled = ggml_add(ctx0, pooled, member(i));
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);

    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, nbt, 1);
    pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, nbt);
    pooled = ggml_rope_multi(ctx0, pooled, inp->touch_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled = ggml_reshape_2d(ctx0, pooled, idx_dim, nbt);
    cb(pooled, "indexer_k_blk", il);

    ggml_tensor * cache = mctx_hyb->get_blk()->get_k_storage(il);
    cache = ggml_view_2d(ctx0, cache, idx_dim, cache->ne[1], cache->nb[1], 0);

    ggml_tensor * upd = ggml_set_rows(ctx0, cache, pooled, inp->touch_dst);
    ggml_build_forward_expand(gf, upd);

    return upd;
}

// Score the pooled blocks and pick each query's best indexer_top_k/ratio blocks.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_blk_select(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_qsa_blk *               inp,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t n_idx_h = hparams.indexer_n_head;
    const int64_t r       = inp->ratio;
    const int64_t n_kv    = mctx_idx->get_n_kv();
    const int64_t n_blk   = n_kv/r;
    const int64_t n_sel   = hparams.indexer_top_k/r;

    // cache this ubatch's raw indexer keys, then re-pool the blocks they touch
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    ggml_tensor * blk = build_qsa_blk_update(mctx_hyb, inp, sections, il);
    blk = ggml_view_2d(ctx0, blk, idx_dim, n_blk, blk->nb[1], 0);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // same score as build_qsa_top_k: rectified per-head dot products, summed over the heads.
    // Prompt batches on a long cache go through in token chunks, so the [n_blk, heads, tokens]
    // score stays around 64 MB (selection is per token, so chunking changes nothing)
    const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(n_tokens,
            (int64_t) (64ull << 20) / (int64_t) (n_blk*n_idx_h*sizeof(float))));

    ggml_tensor * top = nullptr;
    for (int64_t t0 = 0; t0 < n_tokens; t0 += chunk) {
        const int64_t nt = std::min(chunk, n_tokens - t0);

        ggml_tensor * qc = ggml_view_2d(ctx0, q, idx_dim, n_idx_h*nt, q->nb[1], t0*q->nb[2]);
        ggml_tensor * score = ggml_mul_mat(ctx0, blk, qc);
        score = ggml_reshape_3d(ctx0, score, n_blk, n_idx_h, nt);
        score = ggml_relu(ctx0, score);

        ggml_tensor * summed = nullptr;
        if (qwen4exp_qsa_pp_reduce_enabled() && n_idx_h == 4 && nt > 1 &&
                score->type == GGML_TYPE_F32 && ggml_is_contiguous(score)) {
            // Keep the source layout explicit for the private CUDA fusion:
            // [n_blocks, 4 heads, queries, streams=1].
            score = ggml_reshape_4d(ctx0, score, n_blk, 4, nt, 1);
            score = ggml_permute(ctx0, score, 1, 0, 2, 3);
            score = ggml_cont(ctx0, score);
            summed = ggml_sum_rows(ctx0, score);
            summed->op_params[0] = QWEN4EXP_QSA_PP_MARKER;
            summed = ggml_reshape_3d(ctx0, summed, n_blk, nt, 1);
        } else {
            for (int64_t h = 0; h < n_idx_h; ++h) {
                ggml_tensor * slice = ggml_view_2d(ctx0, score, n_blk, nt, score->nb[2], h*score->nb[1]);
                summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
            }
        }
        cb(summed, "indexer_score", il);

        // blocks at or past a query's tail are out of reach: 0 below vis, -1e30 from there on.
        // it depends on the query cells only, so a single-chunk graph builds it once for all layers
        ggml_tensor * vis_bias = nullptr;
        if (nt == n_tokens && inp->vis_bias != nullptr && inp->vis_bias->ne[0] == n_blk) {
            vis_bias = inp->vis_bias;
        } else {
            ggml_tensor * vis = ggml_view_2d(ctx0, inp->vis, 1, nt, inp->vis->nb[1], t0*inp->vis->nb[1]);
            ggml_tensor * idx = ggml_repeat_4d(ctx0, ggml_arange(ctx0, 0.0f, (float) n_blk, 1.0f), n_blk, nt, 1, 1);
            ggml_tensor * seen = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, idx, vis), -1.0f, -0.5f));
            vis_bias = ggml_scale_bias(ctx0, seen, 1e30f, -1e30f);
            if (nt == n_tokens) {
                inp->vis_bias = vis_bias;
            }
        }
        summed = ggml_add(ctx0, summed, vis_bias);

        ggml_tensor * tc = ggml_top_k(ctx0, summed, n_sel);
        if (!ggml_is_contiguous(tc)) {
            tc = ggml_cont(ctx0, tc);
        }
        top = top ? ggml_concat(ctx0, top, tc, 1) : tc;
    }
    cb(top, "indexer_top_blk", il);

    return top;
}

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias);
        qsa->had_blk = mctx_hyb->blk_ok();
        qsa->cparams = &cparams;
        qsa->hparams = &hparams;

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
        qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
        qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
        qsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

        const char * canonical = getenv("LLAMA_QSA_CANONICAL_TOPK");
        if (canonical && atoi(canonical) == 1 && cparams.n_seq_max > 1) {
            qsa->cell_order = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
            ggml_set_input(qsa->cell_order);
            ggml_set_name(qsa->cell_order, "qsa_logical_cell_order");
            static bool logged = false;
            if (!logged) {
                fprintf(stderr, "qsa-canonical-topk enabled: logical-position ordering, physical ids mapped back\n");
                logged = true;
            }
        }

        ggml_set_input(qsa->cell_blk);
        ggml_set_input(qsa->blk_cells);
        ggml_set_input(qsa->blk_pos);
        ggml_set_input(qsa->bias);

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // gathers per stream: blk_cells row s indexes stream s's own cells
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->blk_cells);
    members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_blocks, n_stream);

    // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_cont(ctx0,
                ggml_view_3d(ctx0, members, idx_dim, n_blocks, n_stream,
                        members->nb[2], members->nb[3], i*members->nb[1]));
        pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
    cb(pooled, "indexer_k_pooled", il);

    // count blocks along ne1: rms_norm launches gridDim.y = ne2, capped at 65535, and 262144/4 = 65536
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks*n_stream, 1);
    pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blocks*n_stream);
    pooled = ggml_rope_multi(ctx0, pooled, inp->blk_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    static const bool fused_indexer = [] {
        const char * value = getenv("LLAMA_QSA_INDEXER_IMPL");
        return value != nullptr && std::string(value) == "fused";
    }();
    ggml_tensor * score = nullptr;
    if (fused_indexer && idx_dim == 128 && n_idx_h == 4) {
        ggml_tensor * q_heads = ggml_reshape_4d(ctx0, q, idx_dim, n_idx_h, n_tps, n_stream);
        ggml_tensor * k_heads = ggml_reshape_4d(ctx0, pooled, idx_dim, 1, n_blocks, n_stream);
        ggml_tensor * weights = ggml_fill(ctx0,
                ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_idx_h, n_tps, 1, n_stream), 1.0f);
        ggml_tensor * mask = ggml_fill(ctx0,
                ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_blocks, n_tps, 1, n_stream), 0.0f);
        // Keep the existing block bias and cell mask below this score operation.
        score = ggml_lightning_indexer(ctx0, q_heads, k_heads, weights, mask);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
        score = ggml_reshape_3d(ctx0, score, n_blocks, n_tps, n_stream);
    } else {
        // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
        // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
        score = ggml_mul_mat(ctx0, pooled,
                ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream));
        score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
        score = ggml_relu(ctx0, score);

        // the heads sit side by side on ne[1] and there are only a few of them
        ggml_tensor * summed = nullptr;
        if (qwen4exp_qsa_pp_reduce_enabled() && n_idx_h == 4 && n_tps > 1 &&
                score->type == GGML_TYPE_F32 && ggml_is_contiguous(score)) {
            score = ggml_permute(ctx0, score, 1, 0, 2, 3);
            score = ggml_cont(ctx0, score);
            summed = ggml_sum_rows(ctx0, score);
            summed->op_params[0] = QWEN4EXP_QSA_PP_MARKER;
            summed = ggml_reshape_3d(ctx0, summed, n_blocks, n_tps, n_stream);
        } else {
            for (int64_t h = 0; h < n_idx_h; ++h) {
                ggml_tensor * slice = ggml_view_3d(ctx0, score, n_blocks, n_tps, n_stream,
                        score->nb[2], score->nb[3], h*score->nb[1]);
                summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
            }
        }

        score = summed;

    }
    cb(score, "indexer_score", il);

    // one value per block, so it is cheaper to bias here than after the cells are expanded
    if (blk_bias) {
        score = ggml_add(ctx0, score, inp->bias);
    }

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
    ggml_tensor * expanded = ggml_get_rows(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
    expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

    if (blk_bias) {
        // flash attention keeps the mask in f16; the scores are f32
        ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_tps, n_stream));
    } else {
        expanded = ggml_add(ctx0, expanded, inp->bias);
    }
    cb(expanded, "indexer_score_tokens", il);

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    ggml_tensor * top_k = nullptr;
    if (inp->cell_order) {
        // TopK selects threshold ties and emits selected rows in its column order.
        // Make that order logical rather than dependent on fragmented physical KV cells.
        // Scores remain bit-identical; map the chosen indices back before gathering K/V.
        ggml_tensor * rows = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));
        rows = ggml_get_rows(ctx0, rows, inp->cell_order);
        ggml_tensor * ordered = ggml_cont(ctx0, ggml_permute(ctx0, rows, 1, 0, 2, 3));
        ggml_tensor * selected = ggml_cont(ctx0, ggml_top_k(ctx0, ordered, width));
        selected = ggml_reshape_2d(ctx0, selected, width*n_tps, n_stream);
        ggml_tensor * ids = ggml_view_3d(ctx0, inp->cell_order, 1, n_kv, n_stream,
                inp->cell_order->nb[0], inp->cell_order->nb[1], 0);
        top_k = ggml_cont(ctx0, ggml_get_rows(ctx0, ids, selected));
    } else {
        top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));
    }

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    // Small batches (decode, speculative verify) on a long cache: gather the top_k cells'
    // K/V rows and mask values and attend over those only. This is the masked dense path
    // below restricted to its finite entries, so the result is the same up to summation
    // order, but the cost no longer grows with the cache: each token becomes its own
    // stream (ne[3]) holding its own gathered cells, padded to a FATTN_KQ_STRIDE multiple
    // with -inf mask entries so the vector kernels stay eligible.
    {
        const int64_t width     = top_k->ne[0];
        const int64_t n_tok     = top_k->ne[1];
        const int64_t width_pad = GGML_PAD(width, 256);

        ggml_tensor * k_all = mctx_cur->get_k(ctx0, il);
        ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);

        static const bool force_dense = [] {
            const char * env = getenv("LLAMA_QSA_DENSE");
            return env && atoi(env) != 0;
        }();

        const bool v_trans = v_all->nb[1] > v_all->nb[2];
        const bool sparse  = !force_dense && cparams.flash_attn && !v_trans &&
            top_k->ne[3] == 1 && k_all->ne[3] == 1 && n_tok <= qwen4exp_qsa_gather_max_tokens() &&
            kq_mask->ne[0] == k_all->ne[2] && kq_mask->ne[1] == n_tok && kq_mask->ne[3] == 1 &&
            k_all->ne[2] >= 2*width_pad;

        qwen4exp_qsa_path_trace(n_tok, width, top_k, k_all, kq_mask, width_pad,
                                cparams.flash_attn, v_trans, force_dense, sparse);

        if (sparse) {
            const int64_t n_embd_k = k_all->ne[0]*k_all->ne[1];
            const int64_t n_embd_v = v_all->ne[0]*v_all->ne[1];

            // For large prefill batches, avoid materializing padded K/V in F32 and
            // then converting it for FA. F16 GET_ROWS cannot be followed by PAD, so
            // repeat valid per-query ids to fill the padded columns instead; the
            // attention mask below still marks those duplicate rows as -inf.
            // Keep the established F32 path for <=64 tokens and for padding shapes
            // that cannot be formed without an empty or oversized prefix view.
            const int64_t pad_rows = width_pad - width;
            // Read at graph construction so independent contexts can compare the
            // direct gather with its unfused reference while sharing model weights.
            const char * f16_env = n_tok > 64 ? getenv("LLAMA_QSA_F16_GATHER") : nullptr;
            const bool f16_enabled = f16_env == nullptr || atoi(f16_env) != 0;
            const bool f16_gather = n_tok > 64 && f16_enabled && pad_rows > 0 && pad_rows <= width;
            const char * trace_env = getenv("LLAMA_QSA_PATH_TRACE");
            if (n_tok > 64 && trace_env && atoi(trace_env) != 0) {
                static std::mutex trace_mutex;
                static bool seen_mode[2] = {false, false};
                std::lock_guard<std::mutex> lock(trace_mutex);
                const int mode = f16_gather ? 1 : 0;
                if (!seen_mode[mode]) {
                    fprintf(stderr, "qsa-prefill-gather: mode=%s tokens=%" PRId64 " width=%" PRId64 " padded=%" PRId64 "\n",
                            f16_gather ? "direct-f16" : "reference-f32-pad-cast-f16", n_tok, width, width_pad);
                    seen_mode[mode] = true;
                }
            }

            // top_k is [width, n_tok, 1, 1]; append the first pad_rows indices
            // within each query row so GET_ROWS already produces width_pad rows.
            ggml_tensor * top_k_cont = ggml_is_contiguous(top_k) ? top_k : ggml_cont(ctx0, top_k);
            ggml_tensor * top_k_2d = ggml_reshape_2d(ctx0, top_k_cont, width, n_tok);
            ggml_tensor * ids = nullptr;
            if (f16_gather) {
                ggml_tensor * duplicate_ids = ggml_view_2d(ctx0, top_k_2d, pad_rows, n_tok,
                                                           top_k_2d->nb[1], 0);
                ggml_tensor * padded_ids = ggml_concat(ctx0, top_k_2d, duplicate_ids, 0);
                ids = ggml_reshape_1d(ctx0, ggml_cont(ctx0, padded_ids), width_pad*n_tok);
            } else {
                // [width*n_tok] cell ids, token-major, for gathering whole K/V rows
                ids = ggml_reshape_1d(ctx0, top_k_cont, width*n_tok);
            }

            auto gather = [&](ggml_tensor * cache, int64_t n_embd) {
                ggml_tensor * rows = ggml_view_2d(ctx0, cache, n_embd, cache->ne[2], cache->nb[2], 0);
                ggml_tensor * out = ggml_get_rows(ctx0, rows, ids);
                if (f16_gather) {
                    ggml_tensor * gathered_f16 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16,
                                                                     out->ne[0], out->ne[1]);
                    gathered_f16->op     = GGML_OP_GET_ROWS;
                    gathered_f16->src[0] = rows;
                    gathered_f16->src[1] = ids;
                    out = gathered_f16;
                    return ggml_reshape_4d(ctx0, out, cache->ne[0], cache->ne[1], width_pad, n_tok);
                }

                // Keep the small-batch F32 path unchanged.
                out = ggml_reshape_3d(ctx0, out, n_embd, width, n_tok);
                out = ggml_pad(ctx0, out, 0, pad_rows, 0, 0);                 // zero rows, masked below
                if (n_tok > 64) {
                    // The large-prefill oracle keeps FA input type and dispatch
                    // identical; only GET_ROWS/PAD materialization differs.
                    out = ggml_cast(ctx0, out, GGML_TYPE_F16);
                }
                return ggml_reshape_4d(ctx0, out, cache->ne[0], cache->ne[1], width_pad, n_tok);
            };
            ggml_tensor * k = gather(k_all, n_embd_k);
            ggml_tensor * v = gather(v_all, n_embd_v);

            // per-token mask values at the selected cells: view the mask as one-element rows
            ggml_tensor * m = ggml_view_3d(ctx0, kq_mask, 1, kq_mask->ne[0], n_tok,
                    ggml_element_size(kq_mask), kq_mask->nb[1], 0);
            m = ggml_get_rows(ctx0, m, ggml_is_contiguous(top_k) ? ggml_reshape_2d(ctx0, top_k, width, n_tok)
                                                                 : ggml_cont(ctx0, ggml_reshape_2d(ctx0, top_k, width, n_tok)));
            m = ggml_reshape_4d(ctx0, m, width, 1, 1, n_tok);
            if (width_pad > width) {
                ggml_tensor * pad = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, width_pad - width, 1, 1, n_tok);
                m = ggml_concat(ctx0, m, ggml_fill(ctx0, pad, -INFINITY), 0);
            }
            m = ggml_cast(ctx0, m, GGML_TYPE_F16);

            // build_attn_mha splits q into k->ne[3] == n_tok streams: one query per stream
            ggml_tensor * cur = build_attn_mha(q_cur, k, v, nullptr, m, nullptr, nullptr, 0, kq_scale, il);
            cb(cur, "kqv_out", il);

            if (inp->self_v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
            }

            return cur;
        }
    }

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

// Attend each query to its selected blocks plus its tail block. With cell == position a block
// is one row of a [ratio cells] view of the K/V cache, so whole blocks are gathered at once.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa_blk(
        llm_graph_input_attn_kv * inp,
        llm_graph_input_qsa_blk * blk_inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_blk,
        float                     kq_scale,
        int                       il) {
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));

    const int64_t r      = blk_inp->ratio;
    const int64_t n_tok  = top_blk->ne[1];
    const int64_t n_wblk = top_blk->ne[0] + blk_inp->tail_ids->ne[0];

    // per query: its selected blocks, then its tail block (repeated as padding)
    ggml_tensor * ids = ggml_concat(ctx0, top_blk, blk_inp->tail_ids, 0);

    {
        ggml_tensor * k_all = mctx_cur->get_k(ctx0, il);
        ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);
        // decode-sized batches keep the gather + flash attention path (as fast there); prompt batches,
        // which the gather path cannot take, use the sparse kernel
        // (q and the cached K/V are already Hadamard-rotated alike when the cache rotates them)
        const bool sparse_ok = qwen4exp_qsa_sparse_kernel_enabled() && n_tok > 16 &&
            k_all->type == GGML_TYPE_Q8_0 && v_all->type == GGML_TYPE_Q8_0 && k_all->ne[0] == 256 && v_all->ne[0] == 256 &&
            k_all->ne[3] == 1 && v_all->nb[1] <= v_all->nb[2] && r == 4 && q_cur->type == GGML_TYPE_F32;
        if (sparse_ok) {
            // block-sparse kernel: reads each query's blocks straight from the Q8_0 cache
            ggml_tensor * q = ggml_is_contiguous(q_cur) ? q_cur : ggml_cont(ctx0, q_cur);
            q = ggml_reshape_3d(ctx0, q, q->ne[0], q->ne[1], n_tok);
            ggml_tensor * k3 = ggml_view_3d(ctx0, k_all, k_all->ne[0], k_all->ne[1], k_all->ne[2], k_all->nb[1], k_all->nb[2], 0);
            ggml_tensor * v3 = ggml_view_3d(ctx0, v_all, v_all->ne[0], v_all->ne[1], v_all->ne[2], v_all->nb[1], v_all->nb[2], 0);
            ggml_tensor * out = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 256, q->ne[1], n_tok);
            // FLASH_ATTN_EXT params: scale, max_bias, softcap, prec; [6] the block size
            const float   fparams[3] = { kq_scale, 0.0f, 0.0f };
            const int32_t prec       = GGML_PREC_F32;
            const int32_t ratio_i32  = (int32_t) r;
            memcpy((char *) out->op_params + 0*sizeof(int32_t), fparams, sizeof(fparams));
            memcpy((char *) out->op_params + 3*sizeof(int32_t), &prec, sizeof(prec));
            memcpy((char *) out->op_params + 6*sizeof(int32_t), &ratio_i32, sizeof(ratio_i32));
            out->op     = GGML_OP_FLASH_ATTN_EXT;
            out->src[0] = q;
            out->src[1] = k3;
            out->src[2] = v3;
            out->src[3] = blk_inp->sel_mask;
            out->src[5] = ids;
            cb(out, "kqv_out", il);
            ggml_tensor * cur = ggml_reshape_2d(ctx0, out, 256*q->ne[1], n_tok);
            if (inp->self_v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
            }
            return cur;
        }
    }

    ids = ggml_reshape_1d(ctx0, ids, n_wblk*n_tok);

    // one gather per cache, straight to the F16 that flash attention consumes
    auto gather = [&](ggml_tensor * cache) {
        GGML_ASSERT(cache->ne[3] == 1);
        const int64_t n_embd = cache->ne[0]*cache->ne[1];
        ggml_tensor * rows = ggml_view_2d(ctx0, cache, n_embd*r, cache->ne[2]/r, r*cache->nb[2], 0);

        ggml_tensor * f32 = ggml_get_rows(ctx0, rows, ids);
        ggml_tensor * g = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, f32->ne[0], f32->ne[1], f32->ne[2], f32->ne[3]);
        g->op     = GGML_OP_GET_ROWS;
        g->src[0] = rows;
        g->src[1] = ids;

        return ggml_reshape_4d(ctx0, g, cache->ne[0], cache->ne[1], n_wblk*r, n_tok);
    };

    ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);
    GGML_ASSERT(v_all->nb[1] <= v_all->nb[2] && "block attention needs a non-transposed V cache");

    ggml_tensor * k = gather(mctx_cur->get_k(ctx0, il));
    ggml_tensor * v = gather(v_all);
    ggml_tensor * m = blk_inp->sel_mask;

    // build_attn_mha splits q into k->ne[3] == n_tok streams: one query per stream
    ggml_tensor * cur = build_attn_mha(q_cur, k, v, nullptr, m, nullptr, nullptr, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

// Re-pool the buckets this ubatch touched into the per-sequence rows of the pooled cache.
// Same build as build_qsa_blk_update, but the members come from the plan's cell ids instead
// of a block view of the indexer cache, so the cells may sit anywhere in the shared stream.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_mblk_update(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_qsa_mblk *              inp,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t r       = inp->ratio;
    const int64_t nbt     = inp->touch_dst->ne[0];

    ggml_tensor * k_all  = mctx_idx->get_k(ctx0, il);
    ggml_tensor * k_rows = ggml_view_2d(ctx0, k_all, idx_dim, k_all->ne[2], k_all->nb[2], 0);

    ggml_tensor * members = ggml_get_rows(ctx0, k_rows, inp->member_ids);   // [idx_dim, r*nbt]
    members = ggml_reshape_3d(ctx0, members, idx_dim, r, nbt);

    const auto member = [&](int64_t i) {
        return ggml_view_2d(ctx0, members, idx_dim, nbt, members->nb[2], i*members->nb[1]);
    };
    ggml_tensor * pooled = r > 1 ? ggml_add(ctx0, member(0), member(1)) : ggml_cont(ctx0, member(0));
    for (int64_t i = 2; i < r; ++i) {
        pooled = ggml_add(ctx0, pooled, member(i));
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);

    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, nbt, 1);
    pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, nbt);
    pooled = ggml_rope_multi(ctx0, pooled, inp->touch_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled = ggml_reshape_2d(ctx0, pooled, idx_dim, nbt);
    cb(pooled, "indexer_k_mblk", il);

    ggml_tensor * cache = mctx_hyb->get_blk()->get_k_storage(il);
    cache = ggml_view_2d(ctx0, cache, idx_dim, cache->ne[1], cache->nb[1], 0);

    ggml_tensor * upd = ggml_set_rows(ctx0, cache, pooled, inp->touch_dst);
    ggml_build_forward_expand(gf, upd);

    return upd;
}

// Score every pooled row once, mask each query down to its own sequence's complete blocks,
// pick the best indexer_top_k/ratio rows per query and map them to their member cells.
// Returns I32 [n_sel*ratio, n_tokens] cell ids, the selected blocks' members per query.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_mblk_select(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_qsa_mblk *              inp,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t n_idx_h = hparams.indexer_n_head;
    const int64_t r       = inp->ratio;
    const int64_t n_rows  = inp->row_cells->ne[1];
    const int64_t n_sel   = inp->n_sel;

    GGML_ASSERT(n_rows > 0 && n_sel < n_rows);

    qwen4exp_qsa_mblk_trace(n_tokens, inp->touch_dst->ne[0], n_rows, n_sel, inp->sel_mask->ne[0]);

    // cache this ubatch's raw indexer keys, then re-pool the buckets they touch
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    ggml_tensor * blk = build_qsa_mblk_update(mctx_hyb, inp, sections, il);

    // score the compact row set: the active sequences' own rows, gathered by id from the
    // freshly updated cache (the get_rows keeps the read ordered after the set_rows), so
    // the cost tracks the tokens in play instead of the row partition's full span
    blk = ggml_get_rows(ctx0, blk, inp->c_rows);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // same score as build_qsa_top_k, chunked the same way; the [n_rows, tokens] score and
    // its range bias stay inside the same 64 MB budget
    const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(n_tokens,
            (int64_t) (64ull << 20) / (int64_t) (n_rows*n_idx_h*sizeof(float))));

    // selection space: rows (n_sel whole rows + a forced tail) or, with
    // LLAMA_QSA_MULTI_CELLSEL=1, cells — the same top (n_sel*r + r - 1) cells the
    // per-step rebuild path picks, straggler cells and the competing tail included
    const bool     cellsel = qwen4exp_qsa_mblk_cellsel();
    const int64_t  n_axis  = cellsel ? n_rows*r : n_rows;
    const int64_t  k_sel   = cellsel ? n_sel*r + r - 1 : n_sel;

    GGML_ASSERT(n_axis > 0 && k_sel < n_axis);

    ggml_tensor * top = nullptr;
    for (int64_t t0 = 0; t0 < n_tokens; t0 += chunk) {
        const int64_t nt = std::min(chunk, n_tokens - t0);

        ggml_tensor * qc = ggml_view_2d(ctx0, q, idx_dim, n_idx_h*nt, q->nb[1], t0*q->nb[2]);
        ggml_tensor * score = ggml_mul_mat(ctx0, blk, qc);
        score = ggml_reshape_3d(ctx0, score, n_rows, n_idx_h, nt);
        score = ggml_relu(ctx0, score);

        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * slice = ggml_view_2d(ctx0, score, n_rows, nt, score->nb[2], h*score->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
        }
        cb(summed, "indexer_score", il);

        // cell mode: give every cell its row's score. The repeat must lay out [r, n_rows, nt]
        // (member fastest) so the flat index row*r + j IS the cell id of row_cells' flat view
        if (cellsel) {
            ggml_tensor * src3 = ggml_reshape_3d(ctx0, summed, 1, n_rows, nt);
            ggml_tensor * dst3 = ggml_new_tensor_3d(ctx0, summed->type, r, n_rows, nt);
            summed = ggml_repeat(ctx0, src3, dst3);
            summed = ggml_reshape_2d(ctx0, summed, n_axis, nt);
        }

        // per-query [base, end) of its own sequence's selection space: rows below base belong
        // to other sequences, entries at or above end are the query's own invisible tail.
        // built on the graph from two per-query floats, like the single-sequence vis bias.
        // both differences subtract the [1, nt] column view FROM the [n_axis, nt, 1, 1] index
        // grid: ggml_sub asserts ggml_can_repeat(b, a), i.e. every dim of a must be a multiple
        // of b's, so only this direction is accepted (fix1: the hi term called
        // ggml_sub(vis_end, idx) and tripped that assert on the first fast decode).
        // cell mode uploads cell-space windows [slice0*r, slice0*r + rank + 1) — the same
        // construction, no special case for the tail (c*r + rem = rank + 1).
        ggml_tensor * vis_base = ggml_view_2d(ctx0, inp->vis2, 1, nt, inp->vis2->nb[1], t0*inp->vis2->nb[1]);
        ggml_tensor * vis_end  = ggml_view_2d(ctx0, inp->vis2, 1, nt, inp->vis2->nb[1], t0*inp->vis2->nb[1] + inp->vis2->nb[0]);

        ggml_tensor * idx = ggml_repeat_4d(ctx0, ggml_arange(ctx0, 0.0f, (float) n_axis, 1.0f), n_axis, nt, 1, 1);
        // step(-(idx - base) - 0.5): 1 on entries below the sequence's range
        ggml_tensor * lo  = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, idx, vis_base), -1.0f, -0.5f));
        // step(+(idx - end) + 0.5): 1 on entries at or past the exclusive end (invisible tail, pad, later sequences)
        ggml_tensor * hi  = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, idx, vis_end),   1.0f,  0.5f));
        ggml_tensor * out = ggml_add(ctx0, lo, hi);
        summed = ggml_add(ctx0, summed, ggml_scale_bias(ctx0, out, -1e30f, 0.0f));

        ggml_tensor * tc = ggml_top_k(ctx0, summed, k_sel);
        if (!ggml_is_contiguous(tc)) {
            tc = ggml_cont(ctx0, tc);
        }

        ggml_tensor * cells = nullptr;
        if (cellsel) {
            // row_cells flattened IS the cell -> member map: element (j, row) sits at
            // row*r + j, exactly the cell id — one gather, no layout shuffle
            ggml_tensor * flat = ggml_view_2d(ctx0, inp->row_cells, 1, n_axis, inp->row_cells->nb[0], 0);
            cells = ggml_get_rows(ctx0, flat, ggml_reshape_1d(ctx0, tc, k_sel*nt));
            cells = ggml_reshape_2d(ctx0, cells, k_sel, nt);
        } else {
            // rows to cells: one gather on the plan's row table, laid out per query as
            // [selected block, member within the block]
            cells = ggml_get_rows(ctx0, inp->row_cells, ggml_reshape_1d(ctx0, tc, n_sel*nt));
            cells = ggml_reshape_3d(ctx0, cells, r, n_sel, nt);
            cells = ggml_cont(ctx0, ggml_permute(ctx0, cells, 1, 0, 2, 3));
            cells = ggml_reshape_2d(ctx0, cells, n_sel*r, nt);
        }

        top = top ? ggml_concat(ctx0, top, cells, 1) : cells;
    }
    cb(top, "indexer_top_mblk", il);

    return top;
}

// Attend each query to its selected blocks plus its tail cells, gathered by cell id from
// the shared cache. The dense gather + flash attention shape matches the ordinary top-k
// gather path; the sequence mask is the plan's sel_mask, not the [n_kv] KQ mask.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa_mblk(
        llm_graph_input_attn_kv * inp,
        llm_graph_input_qsa_mblk * blk_inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             sel_cells,
        float                     kq_scale,
        int                       il) {
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));

    const int64_t n_tok  = sel_cells->ne[1];
    const int64_t width  = blk_inp->sel_mask->ne[0];   // already padded for the FA stride

    ggml_tensor * k_all = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);

    GGML_ASSERT(k_all->ne[3] == 1 && "multi pooled attention needs a unified cache");
    GGML_ASSERT(v_all->nb[1] <= v_all->nb[2] && "multi pooled attention needs a non-transposed V cache");

    // row mode: selected blocks then the tail block; cell mode: selected physical cells
    // then masked dummy ids. The tagged sparse mode consumes physical cell ids directly.
    ggml_tensor * ids = ggml_concat(ctx0, sel_cells, blk_inp->tail_ids, 0);

    const bool sparse_cells = qwen4exp_qsa_mblk_sparse_enabled() &&
            qwen4exp_qsa_sparse_kernel_enabled() && qwen4exp_qsa_mblk_cellsel() &&
            blk_inp->ratio == 4 && width % blk_inp->ratio == 0 &&
            sel_cells->ne[0] + blk_inp->tail_ids->ne[0] == width &&
            sel_cells->ne[1] == n_tok && blk_inp->tail_ids->ne[1] == n_tok &&
            q_cur->type == GGML_TYPE_F32 && q_cur->ne[0] == 256 &&
            k_all->type == GGML_TYPE_Q8_0 && v_all->type == GGML_TYPE_Q8_0 &&
            k_all->ne[0] == 256 && v_all->ne[0] == 256 &&
            k_all->ne[3] == 1 && v_all->ne[3] == 1 && v_all->nb[1] <= v_all->nb[2] &&
            q_cur->ne[1] % k_all->ne[1] == 0 &&
            (q_cur->ne[1] / k_all->ne[1] == 24 || q_cur->ne[1] / k_all->ne[1] == 16 ||
             q_cur->ne[1] / k_all->ne[1] == 12 || q_cur->ne[1] / k_all->ne[1] == 8 ||
             q_cur->ne[1] / k_all->ne[1] == 6 || q_cur->ne[1] / k_all->ne[1] == 4 ||
             q_cur->ne[1] / k_all->ne[1] == 3 || q_cur->ne[1] / k_all->ne[1] == 2 ||
             q_cur->ne[1] / k_all->ne[1] == 1);

    if (sparse_cells) {
        ggml_tensor * q = ggml_is_contiguous(q_cur) ? q_cur : ggml_cont(ctx0, q_cur);
        q = ggml_reshape_3d(ctx0, q, q->ne[0], q->ne[1], n_tok);
        ids = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx0, ids);
        ggml_tensor * k3 = ggml_view_3d(ctx0, k_all, k_all->ne[0], k_all->ne[1], k_all->ne[2],
                k_all->nb[1], k_all->nb[2], 0);
        ggml_tensor * v3 = ggml_view_3d(ctx0, v_all, v_all->ne[0], v_all->ne[1], v_all->ne[2],
                v_all->nb[1], v_all->nb[2], 0);
        ggml_tensor * out = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 256, q->ne[1], n_tok);
        const float fparams[3] = { kq_scale, 0.0f, 0.0f };
        const int32_t prec = GGML_PREC_F32;
        const int32_t ratio_i32 = (int32_t) blk_inp->ratio;
        const int32_t cell_id_mode = QWEN4EXP_QSA_ID_MODE_CELLS;
        memcpy((char *) out->op_params + 0*sizeof(int32_t), fparams, sizeof(fparams));
        memcpy((char *) out->op_params + 3*sizeof(int32_t), &prec, sizeof(prec));
        memcpy((char *) out->op_params + 6*sizeof(int32_t), &ratio_i32, sizeof(ratio_i32));
        memcpy((char *) out->op_params + 7*sizeof(int32_t), &cell_id_mode, sizeof(cell_id_mode));
        out->op     = GGML_OP_FLASH_ATTN_EXT;
        out->src[0] = q;
        out->src[1] = k3;
        out->src[2] = v3;
        out->src[3] = blk_inp->sel_mask;
        out->src[5] = ids;
        cb(out, "kqv_out", il);
        ggml_tensor * cur = ggml_reshape_2d(ctx0, out, 256*q->ne[1], n_tok);
        if (inp->self_v_rot) {
            cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
        }
        return cur;
    }

    ids = ggml_reshape_1d(ctx0, ids, width*n_tok);

    // one gather per cache, straight to the F16 that flash attention consumes (as the
    // single-sequence block path's gather): the F32 get_rows fixes the shape, the F16
    // twin carries the data at half the bytes — and on Volta the F32 variant would be
    // converted back to F16 by the tile FA kernel anyway (no F32 vec case at D=256).
    // LLAMA_QSA_F16_GATHER=0 reverts to the plain F32 gather for A/B attribution.
    static const bool f16_gather = [] {
        const char * env = getenv("LLAMA_QSA_F16_GATHER");
        return env == nullptr || atoi(env) != 0;
    }();

    auto gather = [&](ggml_tensor * cache) {
        GGML_ASSERT(cache->ne[3] == 1);
        const int64_t n_embd = cache->ne[0]*cache->ne[1];
        ggml_tensor * rows = ggml_view_2d(ctx0, cache, n_embd, cache->ne[2], cache->nb[2], 0);

        ggml_tensor * out = ggml_get_rows(ctx0, rows, ids);

        if (f16_gather) {
            ggml_tensor * g = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, out->ne[0], out->ne[1], out->ne[2], out->ne[3]);
            g->op     = GGML_OP_GET_ROWS;
            g->src[0] = rows;
            g->src[1] = ids;

            out = g;
        }

        return ggml_reshape_4d(ctx0, out, cache->ne[0], cache->ne[1], width, n_tok);
    };
    ggml_tensor * k = gather(k_all);
    ggml_tensor * v = gather(v_all);
    ggml_tensor * m = blk_inp->sel_mask;

    // build_attn_mha splits q into k->ne[3] == n_tok streams: one query per stream
    ggml_tensor * cur = build_attn_mha(q_cur, k, v, nullptr, m, nullptr, nullptr, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il,
        bool                      kv_only) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    if (kv_only) {
        GGML_ASSERT(cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP);
        GGML_ASSERT(n_outputs == 0 && n_tokens > 0);
        GGML_ASSERT(hparams.dsv4_compress_ratios[il] == 0);

        // Match the normal K/V projection, normalization, RoPE and cache rotations.
        // The draft context only needs these writes to attend over its prompt later.
        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_normed", il);

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Kcur = ggml_rope_multi(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

        if (inp->self_k_rot) {
            Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp->self_k_rot);
        }
        if (inp->self_v_rot) {
            Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp->self_v_rot);
        }

        cb(Kcur, "mtp_kv_only_Kcur", il);
        cb(Vcur, "mtp_kv_only_Vcur", il);

        const auto * mctx_cur = inp->mctx;
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, inp->get_k_idxs(), il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, inp->get_v_idxs(), il));

        return nullptr;
    }

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    // with the pooled block cache valid, every graph keeps it current; small decode batches
    // also select and attend through it at a cost independent of the context length
    const uint32_t ratio = qsa ? hparams.dsv4_compress_ratios[il] : 0;
    const bool     blk   = qsa && mctx_hyb->blk_ok();
    const bool     fast  = blk && qwen4exp_qsa_blk_fast(mctx_hyb, ubatch, cparams, hparams, ratio);

    // the per-sequence pooled cache (multi-sequence runs): same fixed-cost selection and
    // attention, but every sequence scores and gathers its own rows and cells
    const bool mfast = qsa && !blk && mctx_hyb->blk_multi() &&
        qwen4exp_qsa_mblk_fast(mctx_hyb, ubatch, cparams, hparams, ratio);

    llm_graph_input_qsa_blk  * blk_inp  = blk  ? get_qsa_blk_input(mctx_hyb, ratio, fast) : nullptr;
    llm_graph_input_qsa_mblk * mblk_inp = mfast ? get_qsa_mblk_input(mctx_hyb, ratio, hparams.indexer_top_k/ratio) : nullptr;

    ggml_tensor * top_k      = nullptr;
    ggml_tensor * top_blk    = nullptr;
    ggml_tensor * top_mcells = nullptr;

    if (mfast) {
        top_mcells = build_qsa_mblk_select(mctx_hyb, mblk_inp, cur, inp_pos, sections, il);
    } else if (fast) {
        top_blk = build_qsa_blk_select(mctx_hyb, blk_inp, cur, inp_pos, sections, il);
    } else if (qsa) {
        top_k = build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il);
        if (blk) {
            build_qsa_blk_update(mctx_hyb, blk_inp, sections, il);
        }
    }

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_mcells) {
        cur = build_attn_qsa_mblk(inp, mblk_inp, Qcur, Kcur, Vcur, top_mcells, kq_scale, il);
    } else if (top_blk) {
        cur = build_attn_qsa_blk(inp, blk_inp, Qcur, Kcur, Vcur, top_blk, kq_scale, il);
    } else if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    static const bool mono_enabled = [] { const char * e = getenv("LLAMA_MONOMOE_SM70"); return e && atoi(e) != 0; }();
    const auto & layer = model.layers[il];
    ggml_tensor * gate = layer.ffn_gate_up_exps ? layer.ffn_gate_up_exps : layer.ffn_gate_exps;
    const bool quant_gate = gate && (gate->type == GGML_TYPE_IQ2_XXS || gate->type == GGML_TYPE_IQ2_XS ||
        gate->type == GGML_TYPE_IQ2_S || gate->type == GGML_TYPE_IQ3_XXS || gate->type == GGML_TYPE_IQ3_S);
    const bool quant_down = layer.ffn_down_exps && (layer.ffn_down_exps->type == GGML_TYPE_Q2_0 || layer.ffn_down_exps->type == GGML_TYPE_IQ4_NL);
    const bool mono_ok = mono_enabled && n_tokens > 0 && n_tokens <= 32 && loras->empty() && !ft_moe &&
        cur->type == GGML_TYPE_F32 && ggml_is_contiguous(cur) && quant_gate && quant_down &&
        (layer.ffn_gate_inp->type == GGML_TYPE_BF16 || layer.ffn_gate_inp->type == GGML_TYPE_F32) &&
        !layer.ffn_up_exps_s && !layer.ffn_gate_exps_s && !layer.ffn_down_exps_s &&
        hparams.n_expert_groups <= 1 && hparams.swiglu_clamp_exp[il] == 0.0f && n_expert <= 512 && n_expert_used <= 10 &&
        ggml_backend_dev_type(model.dev_layer(il)) != GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_tensor * moe_out = mono_ok
        ? ggml_monomoe_sm70(ctx0, cur, layer.ffn_gate_inp, gate,
            layer.ffn_gate_up_exps ? nullptr : layer.ffn_up_exps, layer.ffn_down_exps,
            n_expert_used, hparams.expert_weights_scale)
        : build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * start = ft_moe ? ft_moe->begin(ctx0,moe_out) : nullptr;
        if (start) { ggml_build_forward_expand(gf,start); }
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        if (start) { moe_out = ft_moe->finish(ctx0,start,ffn_shexp); }
        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        return rows->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        // one strided copy per slot; a ggml_cont first would add a second full copy
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    if (model.ple_device) {
        auto * zeros = ggml_scale(ctx0, ggml_cast(ctx0, rows, GGML_TYPE_F32), 0.0f);
        zeros = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, zeros, 1, n_heads * n_tokens), emb);
        cb(zeros, "ple_rows_zero", -1);
        // A mirrored consumer forces the six-way reduction before the island boundary.
        emb = ggml_add(ctx0, zeros, emb);
        cb(emb, "ple_rows_join", -1);
    }
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * conv_out = nullptr;
    ggml_tensor * normalized_3d = ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs);

    if (qwen4exp_ple_onecat_enabled()) {
        ggml_tensor * conv_states_all = inp->mctx->get_p_l(il);
        const auto kv_head = inp->mctx->get_head();
        const int64_t row_total = conv_states_all->ne[0];
        GGML_ASSERT(row_total == hist * hc_dim);

        auto it = rs_rows.find(conv_states_all);
        if (it == rs_rows.end()) {
            it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
        }

        ggml_tensor * state = ggml_reshape_3d(ctx0, it->second, hist, hc_dim, n_seqs);
        cb(state, "conv_state_at", il);

        const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;
        ggml_tensor * fused = ggml_ple_conv_sm70(ctx0, normalized_3d, state,
                model.layers[il].ple_conv1d, (int32_t) dil, (int32_t) n_slots);

        conv_out = ggml_view_3d(ctx0, fused, hc_dim, n_seq_tokens, n_seqs,
                fused->nb[1], fused->nb[2], 0);
        conv_out = ggml_reshape_3d(ctx0, conv_out, n_embd, hc, n_tokens);

        const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
        const uint32_t mem_size = inp->mctx->get_size();
        for (int64_t slot = 0; slot < n_slots; ++slot) {
            ggml_tensor * tail_raw = ggml_view_3d(ctx0, fused, hc_dim, hist, n_seqs,
                    fused->nb[1], fused->nb[2],
                    (n_seq_tokens + slot * hist) * fused->nb[1]);
            ggml_tensor * tail = ggml_transpose(ctx0, tail_raw);
            ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                    hist * hc_dim, n_seqs, conv_states_all->nb[1],
                    (slot * mem_size + kv_head) * row_size);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
        }
    } else {
        // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
        ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il), normalized_3d,
                hist, hc_dim, il);

        for (int64_t k = 0; k < kern; ++k) {
            // tap k reads (kern-1-k)*dilation positions back
            const int64_t start = hist - (kern - 1 - k) * dil;

            ggml_tensor * shifted = ggml_cont(ctx0,
                    ggml_transpose(ctx0,
                            ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                    padded->nb[1], padded->nb[2],
                                    ggml_row_size(padded->type, start))));

            // column k of the [kern, hc_dim] kernel is one weight per channel
            ggml_tensor * wk = ggml_cont(ctx0,
                    ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                            model.layers[il].ple_conv1d->nb[1],
                            k * model.layers[il].ple_conv1d->nb[0]));
            // this kernel keeps the file type, so cast it before it multiplies an f32 activation
            wk = ggml_reshape_1d(ctx0, wk, hc_dim);
            if (wk->type != GGML_TYPE_F32) {
                wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
            }

            ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
            conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
        }

        conv_out = ggml_silu(ctx0, conv_out);
        conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    }
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
