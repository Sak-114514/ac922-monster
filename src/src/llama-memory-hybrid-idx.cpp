#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <cstdlib>

static bool qsa_blk_debug() {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QSA_BLK_DEBUG"); return e && atoi(e) != 0; }();
    return enabled;
}
#define QSA_BLK_INVALIDATE(why) do { if (blk_valid && qsa_blk_debug()) { LLAMA_LOG_INFO("qsa-blk: layout invalidated by %s\n", why); } blk_valid = false; } while (0)

// LLAMA_QSA_POOLED_MULTI=1: per-sequence pooled indexer cache for unified multi-sequence runs.
// Off by default; the single-sequence pooled cache and its fast path are untouched either way.
static bool qsa_multi_env() {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QSA_POOLED_MULTI"); return e && atoi(e) != 0; }();
    return enabled;
}

// round5: cell-granular selection — the fast path picks the same top (indexer_top_k + r - 1)
// CELLS the per-step rebuild path picks (straggler cells included, tail competes on score),
// instead of n_sel whole rows plus a forced tail; the parity delta shrinks to ties.
// The same env is read on the graph side (qwen4exp) — keep the spellings in sync.
static bool qsa_multi_cellsel() {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QSA_MULTI_CELLSEL"); return e && atoi(e) != 0; }();
    return enabled;
}

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
        has_recurrent |= model.hparams.is_recr(il) && (!filter_recr || filter_recr(il));
    }

    if (mem_idx) {
        for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
            if (filter_idx(il)) {
                blk_ratio = std::max(blk_ratio, model.hparams.dsv4_compress_ratios[il]);
            }
        }
    }

    if (mem_idx && blk_ratio > 0) {
        // the per-sequence tables cut every layer's blocks on the same position line, so the
        // multi cache needs one compress ratio across the indexer layers (the single-sequence
        // pooled cache re-derives its rows per layer and does not care)
        bool ratios_uniform = true;
        for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
            if (filter_idx && filter_idx(il)) {
                ratios_uniform &= model.hparams.dsv4_compress_ratios[il] == blk_ratio;
            }
        }

        const bool want_multi = qsa_multi_env() && n_seq_max > 1 && unified && ratios_uniform;

        if (!want_multi && qsa_multi_env() && n_seq_max > 1 && qsa_blk_debug()) {
            LLAMA_LOG_INFO("qsa-blk: per-sequence pooled cache disabled (unified = %d, uniform ratios = %d)\n",
                    (int) unified, (int) ratios_uniform);
        }

        if (n_seq_max == 1 || want_multi) {
            hparams_blk = hparams_idx;

            // one row per block plus a scratch row for padded updates
            const uint32_t n_pad_blk = std::max<uint32_t>(n_pad, 1);
            const uint32_t size_blk  = GGML_PAD(kv_size/blk_ratio + 1, n_pad_blk);

            LLAMA_LOG_INFO("%s: creating pooled indexer key cache, size = %u blocks\n", __func__, size_blk);

            mem_blk.reset(new llama_kv_cache(
                model, hparams_blk, GGML_TYPE_F32, GGML_TYPE_F32, false, offload, unified,
                size_blk, 1, n_pad_blk, 0, LLAMA_SWA_TYPE_NONE,
                nullptr, filter_idx, nullptr, nullptr, "blk_"));

            // vectorize the pooled rows per sequence: every sequence gets a fixed share of
            // the row space (a sequence longer than share*blk_ratio tokens keeps using the
            // per-step rebuild path). created with n_seq_max == 1 above, so the cache itself
            // stays a flat row store and the allocation below is host-side bookkeeping only.
            qsa_multi = want_multi;
            if (qsa_multi) {
                qsa_multi_share = (size_blk - 1)/n_seq_max;

                qsa_multi_seqs.resize(n_seq_max);
                for (auto & st : qsa_multi_seqs) {
                    st.members.assign(qsa_multi_share*blk_ratio, -1);
                }
                qsa_multi_pooled_to.assign(n_seq_max, -1);

                // the single-sequence fast path assumes cell == rank over the whole cache;
                // with several sequences interleaved in one cells array that would mix them,
                // so it must never engage here (see also clear() and seq_rm()).
                blk_valid = false;

                LLAMA_LOG_INFO("%s: per-sequence pooled indexer rows: %u seqs x %u blocks\n", __func__, n_seq_max, qsa_multi_share);
            }
        }
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    // an empty cache is trivially laid out by position; blocks are rebuilt as they fill.
    // in multi mode the single-sequence fast path must stay off (interleaved sequences)
    if (qsa_multi) {
        qsa_multi_reset();
    }
    blk_valid = !qsa_multi;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    // An attention-only draft has bookkeeping cells but no recurrent state to restore.
    if (!get_mem_recr()->seq_rm(seq_id, has_recurrent ? p0 : -1, has_recurrent ? p1 : -1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    const bool res = get_mem_attn()->seq_rm(seq_id, p0, p1);

    // removal keeps every remaining cell at its position; a later misplaced insert is caught
    // when its ubatch is applied. Emptying the cache makes the layout valid again.
    if (res && get_mem_attn()->seq_pos_max(seq_id < 0 ? 0 : seq_id) < 0 && (seq_id <= 0)) {
        if (!blk_valid && qsa_blk_debug()) {
            LLAMA_LOG_INFO("qsa-blk: layout valid again (cache emptied)\n");
        }
        // multi mode never re-engages the single-sequence fast path
        blk_valid = !qsa_multi;
    }

    // the per-sequence tables track the cells the caches just dropped; p1 == -1 is the tail
    // removal the decode rollback and context-shift paths use, which keeps the dense prefix
    // intact, so only the buckets at or above p0/ratio lose their members and pooled rows.
    // any other removal shape falls back to a full invalidation of the sequence.
    if (res && qsa_multi) {
        const auto handle = [&](llama_seq_id s) {
            if (p1 == -1) {
                qsa_multi_truncate(s, p0);
            } else {
                qsa_multi_drop(s);
            }
        };
        if (seq_id < 0) {
            for (uint32_t s = 0; s < qsa_multi_seqs.size(); ++s) {
                handle(s);
            }
        } else if ((uint32_t) seq_id < qsa_multi_seqs.size()) {
            handle(seq_id);
        }
    }

    return res;
}

llama_pos llama_memory_hybrid_idx::seq_pos_min(llama_seq_id seq_id) const {
    return has_recurrent ? llama_memory_hybrid::seq_pos_min(seq_id) : get_mem_attn()->seq_pos_min(seq_id);
}

llama_pos llama_memory_hybrid_idx::seq_pos_max(llama_seq_id seq_id) const {
    return has_recurrent ? llama_memory_hybrid::seq_pos_max(seq_id) : get_mem_attn()->seq_pos_max(seq_id);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);
    QSA_BLK_INVALIDATE("seq_cp");

    // a copied range shares cells between two position lines; re-derive both tables later
    if (qsa_multi) {
        if ((uint32_t) seq_id_src < qsa_multi_seqs.size()) {
            qsa_multi_drop(seq_id_src);
        }
        if ((uint32_t) seq_id_dst < qsa_multi_seqs.size()) {
            qsa_multi_drop(seq_id_dst);
        }
    }

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);
    QSA_BLK_INVALIDATE("seq_keep");

    if (qsa_multi) {
        for (uint32_t s = 0; s < qsa_multi_seqs.size(); ++s) {
            if ((int32_t) s != seq_id) {
                qsa_multi_drop(s);
            }
        }
    }

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);
    QSA_BLK_INVALIDATE("seq_add");

    // positions move, so every bucket boundary of the shifted sequences moves with them
    if (qsa_multi) {
        if (seq_id < 0) {
            qsa_multi_reset();
        } else if ((uint32_t) seq_id < qsa_multi_seqs.size()) {
            qsa_multi_drop(seq_id);
        }
    }

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);
    QSA_BLK_INVALIDATE("seq_div");

    if (qsa_multi) {
        if (seq_id < 0) {
            qsa_multi_reset();
        } else if ((uint32_t) seq_id < qsa_multi_seqs.size()) {
            qsa_multi_drop(seq_id);
        }
    }

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    if (mem_blk) {
        for (const auto & buft_size : mem_blk->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // pooled block keys are not part of the saved state; a partial (recurrent-only) restore,
    // as the server's context checkpoints use, leaves the attention and indexer cells alone
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        QSA_BLK_INVALIDATE("state_read");

        // restored cells were never registered through a ubatch: rebuild the tables from scratch
        if (qsa_multi) {
            qsa_multi_reset();
        }
    }

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (qsa_multi && (uint32_t) seq_id < qsa_multi_seqs.size()) {
        // a whole-sequence removal (p0 == 0 truncates to nothing): drop the table
        qsa_multi_truncate(seq_id, 0);
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_blk() const {
    return mem_blk.get();
}

//
// per-sequence pooled indexer cache (LLAMA_QSA_POOLED_MULTI=1)
//
// Invariants while a sequence's table is usable (used && valid && !gappy && !overflow):
//  - positions 0..n_tok-1 of the sequence each name exactly one member cell, in dense order,
//    so bucket b covers positions [b*ratio, (b+1)*ratio) and rank == position;
//  - every bucket below dirty_pb has its pooled row already written; buckets in
//    [dirty_pb, n_blk) are re-pooled by the next fast ubatch that touches the sequence.
// A tail rollback (seq_rm p1 == -1) truncates the table at p0 and re-attaches the survivors
// of the straddling bucket, so MTP rejections cost one partial re-pool, not a rebuild.
//

void llama_memory_hybrid_idx::qsa_multi_reset() const {
    for (uint32_t s = 0; s < qsa_multi_seqs.size(); ++s) {
        qsa_multi_drop(s);
    }
}

void llama_memory_hybrid_idx::qsa_multi_drop(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= qsa_multi_seqs.size()) {
        return;
    }

    auto & st = qsa_multi_seqs[seq_id];

    if (st.used && qsa_blk_debug()) {
        LLAMA_LOG_INFO("qsa-blk: seq %d table dropped\n", seq_id);
    }

    st.used     = false;
    st.valid    = false;
    st.gappy    = false;
    st.overflow = false;
    st.n_tok    = 0;
    st.n_blk    = 0;
    st.dirty_pb = 0;

    qsa_multi_pooled_to[seq_id] = -1;

    std::fill(st.members.begin(), st.members.end(), -1);
}

void llama_memory_hybrid_idx::qsa_multi_truncate(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || (uint32_t) seq_id >= qsa_multi_seqs.size()) {
        return;
    }

    auto & st = qsa_multi_seqs[seq_id];

    if (!st.used) {
        return;
    }

    if (p0 <= 0 || !st.valid || st.gappy || st.overflow) {
        // whole-history removal, or a table that cannot be edited in place
        if (qsa_blk_debug()) {
            LLAMA_LOG_INFO("qsa-blk: seq %d table dropped by rollback at %d\n", seq_id, p0);
        }
        qsa_multi_drop(seq_id);

        return;
    }

    const int64_t r = blk_ratio;

    const int64_t pb0 = (int64_t) p0 / r;

    if (pb0 >= st.n_blk) {
        // the removal is above everything the table tracks
        return;
    }

    // forget the removed buckets, then re-attach the survivors of the straddling one:
    // its members below p0 are still in the cache and must stay visible
    for (int64_t pb = pb0; pb < st.n_blk; ++pb) {
        std::fill(st.members.begin() + pb*r, st.members.begin() + (pb + 1)*r, -1);
    }

    const auto & cells = mem_idx->get_cells(seq_id);

    int64_t last_survivor = -1;

    for (uint32_t j = 0; j < cells.size(); ++j) {
        if (cells.is_empty(j) || !cells.seq_has(j, seq_id)) {
            continue;
        }

        const int64_t p = cells.pos_get(j);

        if (p < pb0*r || p >= (llama_pos) p0) {
            continue;
        }

        st.members[pb0*r + (p % r)] = (int32_t) j;

        last_survivor = std::max(last_survivor, p);
    }

    // the dense prefix now ends at the last survivor: never at p0 itself, or a position no
    // cell holds would count as complete and its dummy member cell could be gathered
    st.n_tok    = last_survivor >= 0 ? last_survivor + 1 : pb0*r;
    st.n_blk    = (st.n_tok + r - 1)/r;
    st.dirty_pb = std::min(st.dirty_pb, pb0);

    // rows the removed tokens were pooled into are stale now; never commit them later
    qsa_multi_pooled_to[seq_id] = -1;

    if (qsa_blk_debug()) {
        LLAMA_LOG_INFO("qsa-blk: seq %d table truncated at pos %d (block %lld)\n", seq_id, p0, (long long) pb0);
    }
}

void llama_memory_hybrid_idx::qsa_multi_rescan(llama_seq_id seq_id) const {
    auto & st = qsa_multi_seqs[seq_id];

    // rebuild the whole table from the cells; positions that do not number the members
    // 1:1 (mrope repeats, skipped embeddings, duplicate cells) mark the sequence gappy and
    // it keeps using the per-step rebuild path until it is dropped
    std::fill(st.members.begin(), st.members.end(), -1);

    st.used     = true;
    st.gappy    = false;
    st.overflow = false;
    st.n_tok    = 0;
    st.n_blk    = 0;
    st.dirty_pb = 0;

    const int64_t r = blk_ratio;

    const auto & cells = mem_idx->get_cells(seq_id);

    for (uint32_t j = 0; j < cells.size(); ++j) {
        if (cells.is_empty(j) || !cells.seq_has(j, seq_id)) {
            continue;
        }

        const int64_t p = cells.pos_get(j);
        const int64_t pb = p / r;

        if (pb >= (int64_t) qsa_multi_share) {
            st.overflow = true;
            continue;
        }

        if (p > st.n_tok) {
            st.gappy = true;
        }

        int32_t & m = st.members[pb*r + (p % r)];

        if (m != -1) {
            st.gappy = true;
        }

        m = (int32_t) j;

        st.n_tok = std::max(st.n_tok, p + 1);
    }

    st.n_blk = (st.n_tok + r - 1)/r;
    st.valid = true;

    // every pooled row of this sequence is stale now; the next fast ubatch re-pools [0, tail]
    qsa_multi_pooled_to[seq_id] = -1;

    if (qsa_blk_debug()) {
        LLAMA_LOG_INFO("qsa-blk: seq %d table rescanned: %lld tokens, %lld blocks, gappy = %d\n",
                seq_id, (long long) st.n_tok, (long long) st.n_blk, (int) st.gappy);
    }
}

bool llama_memory_hybrid_idx::qsa_multi_fast_ok(const llama_ubatch & ubatch, uint32_t ratio, uint32_t n_sel) const {
    if (!qsa_multi || ratio == 0) {
        return false;
    }

    // demotion tracing (LLAMA_QSA_BLK_DEBUG=1): each distinct reason logs its first
    // occurrence and then every 1000th, so a demotion loop is visible without spamming
    const auto demote = [&](const char * why) {
        if (qsa_blk_debug()) {
            static std::mutex mutex;
            static std::map<std::string, uint64_t> counts;

            std::lock_guard<std::mutex> lock(mutex);

            const uint64_t n = counts[why]++;

            if (n == 0 || n == 1000 || n == 100000) {
                LLAMA_LOG_INFO("qsa-blk: multi fast path demoted (%s), occurrence #%llu, nt=%u pos0=%d\n",
                        why, (unsigned long long) (n + 1), ubatch.n_tokens,
                        ubatch.n_tokens > 0 ? (int) ubatch.pos[0] : -1);
            }
        }

        return false;
    };

    // blocks are cut on each sequence's own position line, so the indexer and attention cells
    // must all live in the one shared stream (a unified cache)
    if (get_mem_attn()->get_n_stream() != 1) {
        return demote("streams");
    }

    if (ubatch.n_tokens == 0) {
        return false;
    }

    const int64_t r = ratio;

    // tokens of each sequence, and the position the sequence's dense prefix has reached
    std::vector<int64_t> cnt(qsa_multi_seqs.size(), 0);

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1) {
            // a token shared by several sequences would need every one of their block
            // ranges at once; leave it to the per-step rebuild path
            return demote("shared-token");
        }

        const llama_seq_id s = ubatch.seq_id[i][0];

        if (s < 0 || (uint32_t) s >= qsa_multi_seqs.size()) {
            return demote("seq-range");
        }

        const auto & st = qsa_multi_seqs[s];

        if (!st.used || !st.valid || st.gappy || st.overflow) {
            return demote(!st.used ? "seq-unused" : !st.valid ? "seq-invalid" : st.gappy ? "seq-gappy" : "seq-overflow");
        }

        if (ubatch.pos[i] != st.n_tok + cnt[s]) {
            // not a dense continuation of the sequence: leave the table alone, the slow
            // path's registration decides whether this is a gap (gappy) or a replay
            return demote("pos-not-dense");
        }

        cnt[s]++;
    }

    // round5: below the calibrated crossover the per-step rebuild path is cheaper. The
    // threshold is on the AGGREGATE active context — both paths' costs are aggregate
    // properties of the ubatch (fast scores O(sum blocks), slow rebuilds O(n_kv)), so a
    // mixed long+short pair can still take the fast path. Provisional default 32768
    // (mid of the 16K-48K estimate); recalibrate from E2 t_compute per NOTES round 5.
    // LLAMA_QSA_MULTI_MIN_CTX=0 disables the gate.
    static const int64_t min_ctx = [] {
        const char * e = getenv("LLAMA_QSA_MULTI_MIN_CTX");
        return e ? atoll(e) : (int64_t) 32768;
    }();

    if (min_ctx > 0) {
        int64_t total = 0;

        for (uint32_t s = 0; s < qsa_multi_seqs.size(); ++s) {
            if (cnt[s] == 0) {
                continue;
            }

            total += qsa_multi_seqs[s].n_tok + cnt[s];
        }

        if (total < min_ctx) {
            return demote("below-min-ctx");
        }
    }

    for (uint32_t s = 0; s < qsa_multi_seqs.size(); ++s) {
        if (cnt[s] == 0) {
            continue;
        }

        const auto & st = qsa_multi_seqs[s];

        // the sequence must own more complete blocks than the selection budget, its tail
        // block included, exactly like the single-sequence fast path's (c0+1)/ratio test
        if ((st.n_tok + 1)/r < (int64_t) n_sel + 1) {
            return demote("too-few-blocks");
        }

        // the ubatch must still fit in the sequence's share of pooled rows
        if (st.n_tok + cnt[s] > (int64_t) qsa_multi_share*r) {
            return demote("row-share-overflow");
        }
    }

    return true;
}

void llama_memory_hybrid_idx::qsa_multi_sizes(const llama_ubatch & ubatch, uint32_t ratio, uint32_t n_sel,
        int64_t & n_touch, int64_t & n_rows, int64_t & width, int64_t & width_pad) const {
    GGML_ASSERT(qsa_multi);

    const int64_t r = ratio;

    std::vector<int64_t> cnt(qsa_multi_seqs.size(), 0);

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_seq_id s = ubatch.seq_id[i][0];

        GGML_ASSERT(s >= 0 && (uint32_t) s < qsa_multi_seqs.size() && ubatch.n_seq_id[i] == 1);

        cnt[s]++;
    }

    int64_t touch = 0;

    // the active sequences' block counts, in slot order: their compact scoring slices
    // are the prefix sums, so the selection cost tracks the tokens actually in play
    // instead of the row partition's full span (base(seq) = seq*share stays global)
    int64_t live = 0;

    for (uint32_t s = 0; s < qsa_multi_seqs.size(); ++s) {
        if (cnt[s] == 0) {
            continue;
        }

        const auto & st = qsa_multi_seqs[s];

        const int64_t pre = st.n_tok;
        const int64_t to  = (pre + cnt[s] - 1)/r;

        // re-pool from the first stale bucket, or the ubatch's first new bucket, whichever
        // comes first (the registration that follows cannot move this below pre/ratio)
        const int64_t from = std::min(std::min(st.dirty_pb, pre/r), to);

        touch += to - from + 1;

        live += to + 1;
    }

    // both widths pad to the same 256 multiple the attention cache pads its views to, so
    // the graph (and its CUDA-graph capture) stays constant across batches and rejections;
    // pad entries land in the scratch row / are masked out per query
    n_touch = GGML_PAD(touch, 256);

    n_rows = std::min<int64_t>(GGML_PAD(live, 256), (int64_t) mem_blk->get_size());

    const int64_t n_wblk = GGML_PAD((int64_t) n_sel*r + r, 256)/r;

    width     = n_wblk*r;
    width_pad = width;
}

void llama_memory_hybrid_idx::qsa_multi_register(const llama_ubatch & ubatch, const int64_t * cell_ids) const {
    if (!qsa_multi) {
        return;
    }

    const int64_t r = blk_ratio;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const int32_t n_seq = ubatch.n_seq_id[i];

        for (int32_t k = 0; k < n_seq; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];

            if (s < 0 || (uint32_t) s >= qsa_multi_seqs.size()) {
                // a sequence beyond the row partition keeps using the per-step rebuild path
                continue;
            }

            auto & st = qsa_multi_seqs[s];

            if (!st.valid) {
                qsa_multi_rescan(s);
            }

            st.used = true;

            const int64_t p  = ubatch.pos[i];
            const int64_t pb = p / r;

            if (pb >= (int64_t) qsa_multi_share) {
                st.overflow = true;
                continue;
            }

            if (p > st.n_tok) {
                // a hole below the dense prefix: the bucket boundaries stop matching the ranks
                st.gappy = true;
            }

            const int32_t c = (int32_t) cell_ids[i];

            int32_t & m = st.members[pb*r + (p % r)];

            if (m != -1 && m != c) {
                // the same position re-registered on another cell without a rollback in
                // between: the member sets are no longer keyed by position alone
                st.gappy = true;
            }

            // a token written into a bucket makes its pooled row stale even when the cell id
            // repeats: after a rejection the re-decoded token at the same cell is a different
            // key, and the pool is a mean over the members
            st.dirty_pb = std::min(st.dirty_pb, pb);

            m = c;

            st.n_tok = std::max(st.n_tok, p + 1);
            st.n_blk = (st.n_tok + r - 1)/r;
        }
    }
}

void llama_memory_hybrid_idx::qsa_multi_fill(llama_qsa_multi_plan & plan, const llama_ubatch & ubatch,
        uint32_t ratio, uint32_t n_sel) const {
    GGML_ASSERT(qsa_multi);

    const int64_t r     = ratio;
    const int64_t nt    = ubatch.n_tokens;
    const int32_t scr   = (int32_t) mem_blk->get_size() - 1;   // scratch row of the pooled cache

    // per-sequence token counts, as in qsa_multi_sizes (the registration has already run)
    std::vector<int64_t> cnt(qsa_multi_seqs.size(), 0);

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_seq_id s = ubatch.seq_id[i][0];

        GGML_ASSERT(s >= 0 && (uint32_t) s < qsa_multi_seqs.size() && ubatch.n_seq_id[i] == 1);

        cnt[s]++;
    }

    //
    // 1. the re-pooled buckets: [from, to] per sequence, the same ranges qsa_multi_sizes
    //    computed before the registration (from = min(dirty, pre/r, to) in both)
    //

    plan.member_cells.assign(plan.n_touch*r, 0);
    plan.touch_dst   .assign(plan.n_touch, scr);
    plan.touch_pos   .assign(4*plan.n_touch, 0);

    int64_t t = 0;

    for (uint32_t s = 0; s < qsa_multi_seqs.size() && t < plan.n_touch; ++s) {
        if (cnt[s] == 0) {
            continue;
        }

        const auto & st = qsa_multi_seqs[s];

        const int64_t pre = st.n_tok - cnt[s];
        const int64_t to  = (st.n_tok - 1)/r;
        const int64_t from = std::min(std::min(st.dirty_pb, pre/r), to);

        // the graph this plan feeds re-pools [from, to]; qsa_multi_commit() makes the rows
        // current once that graph has actually run
        qsa_multi_pooled_to[s] = to;

        for (int64_t pb = from; pb <= to; ++pb, ++t) {
            GGML_ASSERT(t < plan.n_touch);

            const int32_t * members = st.members.data() + pb*r;

            for (int64_t j = 0; j < r; ++j) {
                // holes (an incomplete bucket after a rollback) read the dummy cell 0 and
                // land in the scratch row or a masked-out row: never selected, never wrong
                plan.member_cells[t*r + j] = members[j] >= 0 ? members[j] : 0;
            }

            if (pb < st.n_blk) {
                plan.touch_dst[t] = (int32_t) (qsa_multi_base(s) + pb);
            }

            // rope of the bucket's first member (for text that is position pb*ratio)
            const int32_t c0 = members[0];

            if (c0 >= 0) {
                const auto & cells = mem_idx->get_cells(s);

                const llama_pos p = cells.pos_get(c0);
                const auto &    e = cells.ext_get(c0);

                const bool is_2d = e.x != 0 || e.y != 0 || p == 0;

                plan.touch_pos[0*plan.n_touch + t] = p;
                plan.touch_pos[1*plan.n_touch + t] = is_2d ? e.y : p;
                plan.touch_pos[2*plan.n_touch + t] = is_2d ? e.x : p;
                plan.touch_pos[3*plan.n_touch + t] = p;
            }
        }
    }

    GGML_ASSERT(t <= plan.n_touch);

    //
    // 2. the compact scoring space: slice s holds sequence s's own rows [0, to_s], at the
    //    prefix sum of the earlier active slices. c_rows maps compact -> global pooled row
    //    (pad -> scratch) for the one get_rows that feeds the scorer; row_cells maps
    //    compact -> member cells. The pooled cache itself keeps its global base(seq) layout.
    //

    plan.c_rows   .assign(plan.n_rows, scr);
    plan.row_cells.assign(plan.n_rows*r, 0);

    std::vector<int64_t> slice0(qsa_multi_seqs.size(), -1);

    int64_t start = 0;

    for (uint32_t s = 0; s < qsa_multi_seqs.size(); ++s) {
        if (cnt[s] == 0) {
            continue;
        }

        const auto & st = qsa_multi_seqs[s];

        const int64_t to = (st.n_tok - 1)/r;

        slice0[s] = start;

        for (int64_t pb = 0; pb <= to && start + pb < plan.n_rows; ++pb) {
            plan.c_rows[start + pb] = (int32_t) (qsa_multi_base(s) + pb);

            const int32_t * members = st.members.data() + pb*r;

            for (int64_t j = 0; j < r; ++j) {
                plan.row_cells[(start + pb)*r + j] = members[j] >= 0 ? members[j] : 0;
            }
        }

        start += to + 1;
    }

    //
    // 3. per-query window, the pad ids, the mask
    //

    // row mode selects n_sel whole rows and forces the tail block in; cell mode selects
    // the same top (n_sel*r + r - 1) CELLS the per-step rebuild path selects (stragglers
    // included, tail competing on score), so the two agree up to ties and fp order
    const bool      cellsel = qsa_multi_cellsel();
    const int64_t   n_sel_r = (int64_t) n_sel*r;
    const int64_t   k_sel   = cellsel ? n_sel_r + r - 1 : n_sel_r;
    const int64_t   tail_w  = plan.width_pad - k_sel;

    GGML_ASSERT(tail_w > 0);

    plan.vis2   .assign(2*nt, 0.0f);
    plan.tail_ids.assign(tail_w*nt, 0);
    plan.sel_mask.assign(plan.width_pad*nt, ggml_fp32_to_fp16(-INFINITY));

    std::vector<int64_t> seen(qsa_multi_seqs.size(), 0);

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_seq_id s = ubatch.seq_id[i][0];

        const auto & st = qsa_multi_seqs[s];

        const int64_t rank = st.n_tok - cnt[s] + seen[s];

        seen[s]++;

        const int64_t c   = (rank + 1)/r;
        const int64_t rem = (rank + 1) - c*r;

        if (cellsel) {
            // the query's own cells with position <= rank are one contiguous window:
            // [slice0*r, slice0*r + rank + 1) — the c complete blocks (c*r cells) plus
            // exactly the rem visible tail members, since c*r + rem = rank + 1
            plan.vis2[2*i + 0] = (float) (slice0[s]*r);
            plan.vis2[2*i + 1] = (float) (slice0[s]*r + rank + 1);
            // pad ids stay 0 (dummy cell, masked off)
        } else {
            // the query's own rows, in the compact scoring space: [slice0, slice0 + c)
            plan.vis2[2*i + 0] = (float) slice0[s];
            plan.vis2[2*i + 1] = (float) (slice0[s] + c);

            // the tail block's members up to and including this query are always visible
            for (int64_t j = 0; j < tail_w; ++j) {
                plan.tail_ids[i*tail_w + j] = j < rem ? st.members[c*r + j] : 0;
            }
        }

        // cell mode: the k_sel selected cells are all genuinely visible, the pad is off.
        // row mode: selected blocks are complete and below the tail; the tail is visible
        // up to the query; the padding repeats the tail block, masked off
        ggml_fp16_t * mi = plan.sel_mask.data() + i*plan.width_pad;

        for (int64_t cell = 0; cell < plan.width_pad; ++cell) {
            const int64_t t2 = cell - k_sel;

            mi[cell] = ggml_fp32_to_fp16(t2 < 0 ? 0.0f : (cellsel ? -INFINITY : (t2 < rem ? 0.0f : -INFINITY)));
        }
    }

    // per-step window log (LLAMA_QSA_BLK_DEBUG=1): the fast path's own view of each query,
    // one line per sequence. Enable for short parity runs only; capped to avoid spam.
    if (qsa_blk_debug()) {
        static uint64_t steps = 0;

        if (steps++ < 4096) {
            for (uint32_t s = 0; s < qsa_multi_seqs.size(); ++s) {
                if (cnt[s] == 0) {
                    continue;
                }

                const auto & st = qsa_multi_seqs[s];

                const int64_t rank0 = st.n_tok - cnt[s];
                const int64_t c0    = (rank0 + 1)/r;

                LLAMA_LOG_INFO("qsa-blk: step #%llu seq %u n_tok=%lld slice=[%lld,%lld) c0=%lld rem0=%lld "
                        "touch=%lld rows=%lld width=%lld dirty_pb=%lld\n",
                        (unsigned long long) steps, s, (long long) st.n_tok,
                        (long long) slice0[s], (long long) (slice0[s] + (st.n_tok + r - 1)/r),
                        (long long) c0, (long long) ((rank0 + 1) - c0*r),
                        (long long) plan.n_touch, (long long) plan.n_rows, (long long) plan.width,
                        (long long) st.dirty_pb);
            }
        }
    }
}

void llama_memory_hybrid_idx::qsa_multi_commit() const {
    for (uint32_t s = 0; s < qsa_multi_seqs.size(); ++s) {
        const int64_t pooled_to = qsa_multi_pooled_to[s];

        if (pooled_to < 0) {
            continue;
        }

        qsa_multi_seqs[s].dirty_pb = std::max(qsa_multi_seqs[s].dirty_pb, pooled_to + 1);

        qsa_multi_pooled_to[s] = -1;
    }
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias, ggml_tensor * cell_order) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    if (cell_order) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cell_order->buffer));
        GGML_ASSERT(cell_order->type == GGML_TYPE_I32 && ggml_is_contiguous(cell_order));
        GGML_ASSERT(cell_order->ne[0] == n_kv && cell_order->ne[1] == n_ns);
    }

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        if (cell_order) {
            int32_t * ids = (int32_t *) cell_order->data + s*n_kv;
            for (int64_t j = 0; j < n_kv; ++j) ids[j] = (int32_t) j;
            const auto before = [&cells](int32_t a, int32_t b) {
                const bool empty_a = cells.is_empty(a);
                const bool empty_b = cells.is_empty(b);
                if (empty_a != empty_b) return !empty_a;
                if (empty_a) return a < b;
                const auto pa = cells.pos_get(a), pb = cells.pos_get(b);
                if (pa != pb) return pa < pb;
                const auto & ea = cells.ext_get(a);
                const auto & eb = cells.ext_get(b);
                if (ea.y != eb.y) return ea.y < eb.y;
                if (ea.x != eb.x) return ea.x < eb.x;
                const auto & sa = cells.seq_get_all(a);
                const auto & sb = cells.seq_get_all(b);
                if (sa != sb) {
                    for (size_t q = 0; q < sa.size(); ++q) {
                        if (sa[q] != sb[q]) return (bool) sa[q];
                    }
                }
                return a < b;
            };
            if (!std::is_sorted(ids, ids+n_kv, before)) std::sort(ids, ids+n_kv, before);
        }

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // rank cells instead of using the position when positions do not number the cells:
        // - mrope repeats one position across an image;
        // - a context that skipped an image's embeddings (the MTP draft context) keeps the target's positions, which
        //   jump past the image and so run ahead of its cells, beyond the cell window
        if (((dup && ubatch->is_pos_2d()) || oor) && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            const auto before = [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            };

            // one sequence written in order leaves its cells sorted already: O(n_kv) instead of a sort per ubatch
            if (!std::is_sorted(order.begin(), order.end(), before)) {
                std::sort(order.begin(), order.end(), before);
            }

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const bool      q2 = ubatch->is_pos_2d();
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = q2 ? ubatch->pos[i + n_tokens]   : 0;
                const llama_pos qx = q2 ? ubatch->pos[i + n_tokens*2] : 0;

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    // a token placed anywhere but in the next cell invalidates the pooled block keys
    if (mem && mem->get_mem_blk() && mem->get_blk_valid() && ctx_idx &&
        get_attn()->has_ubatch() && !get_attn()->ubatch_cells_follow()) {
        if (qsa_blk_debug()) {
            const auto & ub = get_attn()->get_ubatch();
            LLAMA_LOG_INFO("qsa-blk: layout invalidated by a misplaced ubatch (n_tokens=%u, pos0=%d)\n", ub.n_tokens, ub.n_tokens ? ub.pos[0] : -1);
        }
        mem->set_blk_valid(false);
    }

    // apply() runs before this ubatch's graph here, so this commits the previous fast
    // ubatch's re-pool (device ordering makes those rows current before anything reads
    // them; a rollback in between cleared its pending entry already)
    if (mem && mem->get_qsa_multi()) {
        mem->qsa_multi_commit();
    }

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

llama_kv_cache * llama_memory_hybrid_idx_context::get_blk() const {
    return mem ? mem->get_mem_blk() : nullptr;
}

bool llama_memory_hybrid_idx_context::blk_ok() const {
    if (get_blk() == nullptr || !mem->get_blk_valid() || !get_attn()->ubatch_cells_follow()) {
        return false;
    }
    // the pooled blocks name cells of both caches, so they must agree; if they do not, the layout is gone for good
    if (get_idx() == nullptr || !get_idx()->ubatch_cells_follow() || get_attn()->ubatch_cell0() != get_idx()->ubatch_cell0()) {
        mem->set_blk_valid(false);
        return false;
    }
    return true;
}

bool llama_memory_hybrid_idx_context::blk_multi() const {
    return mem != nullptr && mem->get_qsa_multi() && mem->get_mem_blk() != nullptr;
}

bool llama_memory_hybrid_idx_context::qsa_multi_fast_ok(const llama_ubatch & ubatch, uint32_t ratio, uint32_t n_sel) const {
    GGML_ASSERT(mem != nullptr);

    return mem->qsa_multi_fast_ok(ubatch, ratio, n_sel);
}

void llama_memory_hybrid_idx_context::qsa_multi_sizes(const llama_ubatch & ubatch, uint32_t ratio, uint32_t n_sel,
        int64_t & n_touch, int64_t & n_rows, int64_t & width, int64_t & width_pad) const {
    GGML_ASSERT(mem != nullptr);

    mem->qsa_multi_sizes(ubatch, ratio, n_sel, n_touch, n_rows, width, width_pad);
}

void llama_memory_hybrid_idx_context::qsa_multi_register(const ggml_tensor * k_idxs, const llama_ubatch * ubatch) const {
    if (mem == nullptr || !mem->get_qsa_multi()) {
        return;
    }

    GGML_ASSERT(ggml_backend_buffer_is_host(k_idxs->buffer));

    // the indexer cache mirrors the attention slots, and a unified cache numbers them plainly
    GGML_ASSERT(mem->get_mem_attn()->get_n_stream() == 1);

    mem->qsa_multi_register(*ubatch, (const int64_t *) k_idxs->data);
}

void llama_memory_hybrid_idx_context::qsa_multi_fill(llama_qsa_multi_plan & plan, const llama_ubatch * ubatch,
        uint32_t ratio, uint32_t n_sel) const {
    GGML_ASSERT(mem != nullptr);

    mem->qsa_multi_fill(plan, *ubatch, ratio, n_sel);
}

void llama_memory_hybrid_idx_context::qsa_multi_commit() const {
    if (mem == nullptr) {
        return;
    }

    mem->qsa_multi_commit();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias, ggml_tensor * cell_order) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias, cell_order);
}
