#pragma once

#include "llama-memory-hybrid.h"

#include <memory>
#include <vector>

//
// llama_qsa_multi_plan
//
// One ubatch's worth of host-side bookkeeping for the per-sequence pooled indexer cache
// (LLAMA_QSA_POOLED_MULTI=1, see llama_memory_hybrid_idx). Filled by
// llama_memory_hybrid_idx::qsa_multi_fill() after the ubatch's tokens are registered,
// consumed by the qwen4exp multi-sequence QSA graph inputs:
//   member_cells / touch_dst / touch_pos : re-pool these buckets into the pooled cache
//   vis2                                : per query [row base, row end) of its own sequence
//   row_cells                           : pooled row -> member cells, for the selected blocks
//   tail_ids / sel_mask                 : always-visible tail cells and the attention mask
//
struct llama_qsa_multi_plan {
    int64_t n_touch   = 0;   // buckets re-pooled this ubatch, padded to a 256 multiple (graph shape; pad entries land in the scratch row)
    int64_t n_rows    = 0;   // compact scoring rows (sum of the active sequences' block counts), padded to a 256 multiple
    int64_t width     = 0;   // selected + tail cells per query (== width_pad)
    int64_t width_pad = 0;   // width, kept for symmetry with the single-sequence path

    std::vector<int32_t>     member_cells;  // I32 [ratio*n_touch]
    std::vector<int32_t>     touch_dst;     // I32 [n_touch]           global pooled-cache rows (base + pb, or scratch)
    std::vector<int32_t>     touch_pos;     // I32 [4*n_touch]
    std::vector<float>       vis2;          // F32 [2*n_tokens] row mode: [slice0, slice0 + complete blocks);
                                            //                cell mode (LLAMA_QSA_MULTI_CELLSEL=1):
                                            //                [slice0*ratio, slice0*ratio + rank + 1) in cells
    std::vector<int32_t>     c_rows;        // I32 [n_rows]            compact row -> global pooled-cache row (pad -> scratch)
    std::vector<int32_t>     row_cells;     // I32 [ratio*n_rows]      compact row -> member cells; flat index row*ratio + j
                                            //                         is the cell id of the cell-granular selection space
    std::vector<int32_t>     tail_ids;      // I32 [width_pad - k_sel, n_tokens] row mode: the tail block's members,
                                            //                        padded; cell mode: all-zero pad ids (masked -inf)
    std::vector<ggml_fp16_t> sel_mask;      // F16 [width_pad, n_tokens] first k_sel columns 0, pad -inf
                                            //                        (row mode: the tail columns are 0 up to rem)
};

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
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
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // Pooled indexer keys, one F32 row per block of `ratio` cells (row n_blocks is a
    // scratch row), kept up to date incrementally while blk_valid holds. blk_valid means a
    // single sequence written in order, every cell holding the token of the same rank (for
    // text that is the position of the same number), so a block's members are ratio
    // consecutive cells and a finished block's pooled key never changes. Anything that breaks
    // that (copies, shifts, restores, other placements) clears it until the cache is emptied
    // again.
    llama_kv_cache * get_mem_blk() const;   // nullptr when the model carries no indexer
    uint32_t get_blk_ratio() const { return blk_ratio; }
    bool     get_blk_valid() const { return blk_valid; }
    void     set_blk_valid(bool v) const { blk_valid = v; }

    // Per-sequence pooled indexer cache for unified multi-sequence runs (LLAMA_QSA_POOLED_MULTI=1).
    // With more than one sequence sharing one cells array the single-sequence fast path above can
    // never engage (blk_valid stays false); instead each sequence keeps its own block table:
    //   rows [base(seq), base(seq)+n_blk) of mem_blk hold that sequence's pooled block keys,
    //   base(seq) = seq * share, share = (mem_blk rows - 1) / n_seq_max
    // Blocks are cut on each sequence's own position line (bucket = pos/ratio, slot = pos%ratio),
    // so interleaved cells of other sequences never leak into a block. Every sequence falls back
    // to the ordinary per-step rebuild path (set_input_qsa) unless its table is usable:
    //   qsa_multi_fast_ok : eligibility of the current ubatch, from the tables (no mutation)
    //   qsa_multi_sizes   : the graph shapes this ubatch produces (same state as fast_ok)
    //   qsa_multi_register: fold a ubatch's tokens into the tables (fast and slow inputs alike)
    //   qsa_multi_fill    : the plan the fast graph's inputs upload
    bool get_qsa_multi() const { return qsa_multi; }
    uint32_t get_qsa_multi_share() const { return qsa_multi_share; }

    bool qsa_multi_fast_ok(const llama_ubatch & ubatch, uint32_t ratio, uint32_t n_sel) const;

    void qsa_multi_sizes(const llama_ubatch & ubatch, uint32_t ratio, uint32_t n_sel,
                        int64_t & n_touch, int64_t & n_rows, int64_t & width, int64_t & width_pad) const;

    // cell_ids are the indexer cache destinations of the ubatch's tokens, as written by
    // llama_kv_cache_context::set_input_k_idxs (a plain cell index per token in a unified cache)
    void qsa_multi_register(const llama_ubatch & ubatch, const int64_t * cell_ids) const;

    void qsa_multi_fill(llama_qsa_multi_plan & plan, const llama_ubatch & ubatch,
                        uint32_t ratio, uint32_t n_sel) const;

    // the fast graph's re-pool of this ubatch ran to completion: the touched rows are now
    // current. called from the context's apply(), so a failed or aborted ubatch leaves its
    // rows marked stale and the next fast step simply pools them again
    void qsa_multi_commit() const;

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, ggml_tensor * cell_order = nullptr) const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // multi-sequence pooled cache maintenance (all const-mutable: driven from set_input)
    void   qsa_multi_reset() const;                 // drop every sequence's table
    void   qsa_multi_drop(llama_seq_id seq_id) const;              // whole-sequence invalidation
    void   qsa_multi_truncate(llama_seq_id seq_id, llama_pos p0) const; // tail rollback at p0
    void   qsa_multi_rescan(llama_seq_id seq_id) const;            // rebuild one table from the cells
    int64_t qsa_multi_base(llama_seq_id seq_id) const { return (int64_t) seq_id * qsa_multi_share; }

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;
    llama_hparams hparams_blk;
    bool has_recurrent = false;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    uint32_t blk_ratio = 0;
    mutable bool blk_valid = true;
    std::unique_ptr<llama_kv_cache> mem_blk;

    // LLAMA_QSA_POOLED_MULTI=1 with n_seq_max > 1 and a unified cache
    bool     qsa_multi = false;
    uint32_t qsa_multi_share = 0;   // pooled rows reserved per sequence id

    struct qsa_multi_seq_t {
        bool     used     = false;  // the sequence has been seen
        bool     valid    = false;  // the table matches the cells (else rescan on next touch)
        bool     gappy    = false;  // positions do not number the members 1:1 -> slow path
        bool     overflow = false;  // sequence longer than its row share -> slow path
        int64_t  n_tok    = 0;      // positions registered (dense prefix)
        int64_t  n_blk    = 0;      // buckets tracked = ceil(n_tok/ratio)
        int64_t  dirty_pb = 0;      // first bucket whose pooled row is stale/absent
        std::vector<int32_t> members;  // [share*ratio] cell of each member, -1 = hole
    };
    mutable std::vector<qsa_multi_seq_t> qsa_multi_seqs;  // [n_seq_max]

    // rows the current ubatch's fast graph re-pooled, committed by qsa_multi_commit()
    mutable std::vector<int64_t> qsa_multi_pooled_to;     // [n_seq_max], -1 = none
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // the pooled block-key cache and whether the current ubatch may use / must update it
    llama_kv_cache * get_blk() const;
    bool blk_ok() const;

    // per-sequence pooled cache (LLAMA_QSA_POOLED_MULTI=1): thin wrappers over the memory's
    // tables, so the qwen4exp graph inputs stay dumb about where the bookkeeping lives
    bool blk_multi() const;
    bool qsa_multi_fast_ok(const llama_ubatch & ubatch, uint32_t ratio, uint32_t n_sel) const;
    void qsa_multi_sizes(const llama_ubatch & ubatch, uint32_t ratio, uint32_t n_sel,
                        int64_t & n_touch, int64_t & n_rows, int64_t & width, int64_t & width_pad) const;
    void qsa_multi_register(const ggml_tensor * k_idxs, const llama_ubatch * ubatch) const;
    void qsa_multi_fill(llama_qsa_multi_plan & plan, const llama_ubatch * ubatch,
                        uint32_t ratio, uint32_t n_sel) const;
    void qsa_multi_commit() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, ggml_tensor * cell_order = nullptr) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
