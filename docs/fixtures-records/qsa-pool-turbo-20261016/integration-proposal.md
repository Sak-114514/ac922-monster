# Integration proposal (not applied)

Source under review: `/home/debian/ac922-monster/experiments/standard-opt-20261014/src/`, runtime V42/V43 family. The live source was not edited. Use the exact current branch as the patch base because the context file has later V43 changes.

## A. `llama-memory-hybrid-idx.{h,cpp}`

1. Add `llama-qsa-multi-phase-ledger.h` as a private internal helper and replace the current `qsa_multi_seq_t::{valid,dirty_pb}` plus parallel `qsa_multi_pooled_to[]` with one `phase_ledger` per sequence. The helper in this directory defines the intended transition semantics.
2. `qsa_multi_drop(seq)` calls `ledger.invalidate()` and retains its existing clearing of `used/gappy/overflow/n_tok/n_blk/members`.
3. `qsa_multi_rescan(seq)` rebuilds `members` from current `mem_idx` cells, then calls `ledger.rebuilt()`; this is **not** a pooled-row commit. `dirty_from` must remain 0.
4. `qsa_multi_register()` uses `ledger.valid` for rescan and `ledger.mark_dirty(pb)` on every token re-register, including an overwrite after rollback.
5. `qsa_multi_fill()` uses `ledger.dirty_from` to choose `from`, then calls `ledger.schedule_update(to)`. Keep the existing graph dependency: `build_qsa_mblk_update()` returns the `SET_ROWS` result and the selection scorer reads rows from that result before any selection.
6. Keep `qsa_multi_commit()` only as a **successful graph completion** event. It calls `ledger.graph_completed()` for the rows whose update graph really completed. Do not infer success merely from `valid=true` or from `qsa_multi_rescan()`.
7. Add a public narrowly scoped `qsa_multi_invalidate_seq(seq)` wrapper over `qsa_multi_drop(seq)`, callable by the hybrid-index context. Do not clear the global pooled cache or other sequence tables.
8. On a graph abort/error after `qsa_multi_fill()` scheduled rows, call `qsa_multi_invalidate_seq()` for the sequence IDs in that graph before the next fast-path decision. A pending `pooled_to` range must never be committed after failure.
9. Existing `seq_rm`, `seq_cp`, `seq_add/div/keep`, and state restore/drop logic already route to per-sequence `qsa_multi_drop/truncate`; preserve those paths. `rollback_tail(p0)` cancels pending work and dirties `p0/ratio` (the straddling bucket); removal to position 0 or slot reuse invalidates the whole ledger.

## B. `freeze_current()` in `llama-memory-hybrid-idx.cpp`

Current guard rejects every `mem->get_qsa_multi()` frame. Permit it only when the current applied ubatch has exactly one unique sequence, every token has exactly one `seq_id`, `n_stream==1`, indexer and attention contexts both own the ubatch, and the frame snapshot is complete. After all snapshot allocations/checks succeed but before returning the frozen context, call `mem->qsa_multi_invalidate_seq(seq_id)`. Frozen `qsa_multi_register/fast_ok/fill` must remain disabled; `set_input_qsa()` continues reading the frozen `cells_snapshot` and ordinary cell selector.

The ordering is deliberate: applying the current ubatch establishes its cell metadata; `freeze_current()` copies that metadata; invalidation happens before the graph can enqueue/write the frozen frame's KV/indexer tensors. A failed freeze must not invalidate state because Turbo will not consume that snapshot.

## C. Context gate snippet (`llama-context.cpp`, not included in core patch)

Replace the current blanket `get_blk()==nullptr && !blk_multi() && !LLAMA_QSA_POOLED_MULTI` rejection with the actual memory mode check:

```cpp
const bool frame_memory_ready = frame_hybrid_idx != nullptr &&
        frame_hybrid_idx->get_idx() != nullptr &&
        (frame_hybrid_idx->get_blk() == nullptr ||
         (frame_hybrid_idx->get_qsa_multi() && frame_hybrid_idx->blk_multi()));
```

This keeps Turbo enabled for no-pool memory and the per-sequence pool, while still rejecting the single global pooled cache (`get_blk()!=nullptr && get_qsa_multi()==false`). The memory object's constructed mode is authoritative; do not use an environment-variable check as a substitute.

## D1. Earlier logical CELLSEL idea (superseded by D2)

Before enabling `mfast` in production, align it with `build_qsa_top_k()`:

- The reference uses `width=min(n_kv,indexer_top_k+ratio-1)` and causal `set_input_qsa()` gives all currently visible incomplete-tail cells `+1e9`, so the tail is always included. Multi CELLSEL currently has the same cardinality only when `indexer_top_k % ratio == 0`, but no tail priority. Gate on divisibility (or pass the exact budget) and upload per-query `tail_start`; add `+1e9` to cells in `[tail_start, vis_end)` before top-k. Keep invisible rows at `-1e30` and the first `k_sel` attention mask entries unmasked.
- For cutoff ties under physical-cell reordering, use logical candidate order in both paths. Existing `LLAMA_QSA_CANONICAL_TOPK=1` makes the reference selector sort by `(position, ext, seq-set, physical-id)` before top-k and map ids back. In multi CELLSEL, each active seq slice is already `(bucket, member rank)`. Ensure canonical ordering is enabled for the reference slow fallback whenever multi-pool is configured.
- Require the configured CUDA TOP_K implementation to be stable for equal values. `GGML_CUDA_TOPK_STABLE=1` is consulted only in the radix implementation; the CUB branch does not consult it. Identify the build's actual dispatch before claiming deterministic tie parity.
- Keep `LLAMA_QSA_MULTI_SPARSE=0`: the direct sparse attention consumer changes accumulation order and is independent of selector-id parity.

## D2. Physical-id CELLSEL patch (supersedes the earlier canonical-order suggestion)

The staging patch qsa-physical-cellsel.patch keeps the pooled GEMM compact, then scatters each query's visible compact cells to their original physical cell IDs before TOP_K. The patch also gives every foreign, future, or padded candidate a unique scratch row beyond n_kv, and applies +1e9 to the causal partial-tail entries before scatter. TOP_K sees only the first n_kv physical rows, in the same physical cell-ID order as the unpooled selector. Thus exact cutoff ties follow the original physical-ID order; LLAMA_QSA_CANONICAL_TOPK is explicitly demoted because it would change the baseline tie rule.

The graph input stores its build-time padded n_kv and makes can_reuse fail when the current indexer context reports a different padded n_kv; the physical score buffer and TOP_K row dimension otherwise become stale across KV-pad buckets. The GGML shape is base F32[1,n_kv+n_axis,nt], source F32[1,n_axis,nt], indices I32[n_axis,nt]. The ggml_set_rows assertions match: base/source ne0=1 and ne2=nt, source ne1=indices ne0=n_axis, source ne2 divides indices ne1=nt, and indices ne3=1. CUDA dispatch reaches set-rows.cu, whose F32/I32 path has no SM70 exclusion. Unique scratch rows avoid concurrent duplicate destination writes. No new CUDA kernel is needed for correctness by static source inspection.

The patch refuses CELLSEL unless indexer_top_k divides by ratio and GGML_CUDA_TOPK_STABLE=1. AC922 CUDA 12.4 exposes CCCL 2.3.0, so the current build uses the radix TOP_K path, where the stable flag makes equal-score selection follow physical input order. A future CCCL 3.2+ DeviceTopK build must revisit this because the CUB path declares nondeterminism and ignores that environment variable.

This restores exact candidate budget and selection/mask behavior while keeping score GEMM proportional to active pooled rows. TOP_K returns to O(n_kv) inspection and the patch adds an F32 fill, SET_ROWS scatter, and contiguous view copy; performance is unmeasured. It is a staged correctness candidate, not a speed result.

## Required GPU qualification after core integration

1. `CELLSEL=1`, `MULTI_SPARSE=0`, stable physical-id selector: compare selected **physical cell IDs**, candidate budget, and causal mask against ordinary top-k for dense 2/4-sequence histories, including tail remainders 1..ratio-1, exact cutoff ties, and interleaved physical cell order.
2. A sequence hole must demote from `mfast`; verify ordinary fallback mask contains no foreign, future, or absent cells.
3. Turbo freeze a seeded sequence, resume mutable slow path, then verify first re-enabled pooled graph writes every dirty row before its first score read; test 4→1→4, tail rollback, and forced graph failure. Other sequence IDs must retain their ledgers.
4. Only after these pass, run logits/semantic parity. Do not use this host fixture as an end-to-end numerical proof.
