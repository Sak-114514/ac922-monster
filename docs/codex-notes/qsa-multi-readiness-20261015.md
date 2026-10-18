# QSA multi readiness audit (2026-10-15)

Scope: read-only audit of the private source at
`/home/debian/ac922-monster/experiments/standard-opt-20261014/src`, its current experiment fixtures/logs, and earlier QSA artifacts. No engine build, model load/request, GPU work, or source edits were done. This note is the only file created.

## Finding

The private source contains a plausible per-sequence pooled path with explicit fallback guards, but the current experiment tree has no QSA-multi CPU fixture and no current QSA-multi activation/semantic log. Do not treat constructor allocation or an old `status=pass` benchmark as proof that the current binary engaged the fast path or preserved output.

The proposed path caches pooled indexer keys by each sequence's position buckets and explicitly maps selected cells back to KV cell IDs. Cell selection is a separate opt-in. With defaults, both `LLAMA_QSA_POOLED_MULTI` and `LLAMA_QSA_MULTI_CELLSEL` are off; `LLAMA_QSA_MULTI_MIN_CTX` defaults to 32768. At 2 x 8K, the aggregate context is below that threshold, so the path demotes unless the test sets `LLAMA_QSA_MULTI_MIN_CTX=0`.

## Exact gates and controls

- `src/llama-memory-hybrid-idx.cpp:26-40`: `LLAMA_QSA_POOLED_MULTI` and `LLAMA_QSA_MULTI_CELLSEL` are read once into static values; both default off. Set them before process start.
- `src/llama-memory-hybrid-idx.cpp:108-160`: multi storage is created only for `n_seq_max > 1`, a unified cache, and uniform indexer compression ratios. The per-sequence row share is fixed at construction.
- `src/llama-memory-hybrid-idx.cpp:679-768`: eligibility requires one cache stream, one sequence ID per ubatch token, a used/valid/dense table, contiguous positions, sufficient blocks, room in that sequence's fixed row share, and the aggregate-active-context threshold. Shared-token ubatches and gappy/mrope layouts demote safely.
- `src/models/qwen4exp.cpp:979-997`: `LLAMA_QSA_GATHER_MAX_TOKENS` defaults to 16 (valid override up to 256). The multi pooled path is decode/small-verify only: it falls back above 64 tokens and is bounded by this setting below that.
- `src/models/qwen4exp.cpp:1503-1537`: `LLAMA_QSA_DENSE=1`, non-causal attention, no flash attention, ALiBi, missing multi cache, or token limits demote the path.
- `src/models/qwen4exp.cpp:944-950,2648-2663`: `LLAMA_QSA_MULTI_SPARSE=1` is a separate opt-in for the tagged Q8 cell-ID sparse attention kernel. It additionally requires `LLAMA_QSA_MULTI_CELLSEL=1`, ratio 4 and specific Q8/head/cache geometry. Keep it off for the first qualification.
- `src/models/qwen4exp.cpp:1046-1084`: the current graph-shape trace is `LLAMA_QSA_PATH_TRACE=1`; it emits `qsa-mblk:` lines with `cellsel=`. `LLAMA_QSA_BLK_DEBUG=1` reports demotion reasons and lifecycle events. An older artifact sets `LLAMA_QSA_MBLK_TRACE=1`, which this current trace helper does not read.

Recommended isolated candidate flags for 2 x 8K are:
```
LLAMA_QSA_POOLED_MULTI=1
LLAMA_QSA_MULTI_CELLSEL=1
LLAMA_QSA_MULTI_MIN_CTX=0
LLAMA_QSA_MULTI_SPARSE=0
LLAMA_QSA_PATH_TRACE=1
LLAMA_QSA_BLK_DEBUG=1
```
Leave `LLAMA_QSA_DENSE` unset/0. Hold the already qualified `LLAMA_QSA_INDEXER_IMPL`, gather setting, MTP config, and all other runtime flags identical between baseline and candidate. For 64K, the default 32768 context gate should admit eligible decode ubatches. If MTP verify ubatches exceed 16 tokens, inspect the trace/demotion first and set a matching `LLAMA_QSA_GATHER_MAX_TOKENS` (no more than 64 for the multi fast path), identically in both arms.

## What work it removes

The slow route in `src/models/qwen4exp.cpp:1978-2014` gathers and re-pools the full indexer KV history into block keys, then normalizes and RoPEs all blocks on each graph. The multi route in `src/llama-memory-hybrid-idx.cpp:771-824,888-1005` plans only dirty/new block updates and `src/models/qwen4exp.cpp:2431-2472` writes those rows. That removes repeated full-history pooling, norm and RoPE work.

It does **not** reduce the query-to-block dot-product matrix to newly touched rows. The slow scorer covers `n_blocks` rows at `src/models/qwen4exp.cpp:2041-2068`; the multi scorer gathers the compact live block rows and runs `ggml_mul_mat` over that set at `src/models/qwen4exp.cpp:2505-2538`. `qsa_multi_sizes` pads the sum of active sequences' live blocks at `src/llama-memory-hybrid-idx.cpp:789-824`. For ubatches containing all active slots, both score approximately all live KV blocks (≈ nKV/ratio). Multi compacts to sequences present in the current ubatch, so idle slots are omitted; it still recomputes query-to-block dot products for every compact row on every step. Its main savings are full-history pooling/norm/RoPE and fixed-width K/V attention gather. Cell selection repeats block scores over cells and top-k selects 2051 cells for ratio 4, adding selection cost.

## Correctness and replay path to qualify

- Sequence isolation: each sequence owns its own row range and position-bucket member table (`src/llama-memory-hybrid-idx.h:194-211`); query visibility and selected-cell IDs are constructed per query in `src/llama-memory-hybrid-idx.cpp:976-1067`. Cellsel maps flat index `row * ratio + member` to the matching member cell; graph selection/gather is at `src/models/qwen4exp.cpp:2523-2600`.
- Tail rollback / MTP rejection: `seq_rm(..., p1=-1)` calls `qsa_multi_truncate` at `src/llama-memory-hybrid-idx.cpp:253-297`; truncation preserves surviving cells in the partial bucket, marks the bucket dirty and clears pending pool commits at lines 524-592. Dirty rows are marked current only after the graph through the next `apply()` at lines 1097-1108 and 1555-1585.
- Whole-state restore/drop: non-partial `state_read` resets the host tables before reading restored cache cells (`src/llama-memory-hybrid-idx.cpp:410-450)); `state_drop`, `clear`, `seq_cp`, `seq_keep`, `seq_add`, and `seq_div` drop/reset affected tables (`src/llama-memory-hybrid-idx.cpp:238-377,453-472`). Partial recurrent-only restore intentionally keeps KV and the tables.
- Graph reuse: `llm_graph_input_qsa_mblk::can_reuse` rechecks eligibility and all shape-bearing inputs (`src/models/qwen4exp.cpp:1588-1614)); touch, row, and attention widths are padded to 256 in `qsa_multi_sizes` (`src/llama-memory-hybrid-idx.cpp:804-824`). Static code gives shape-stable steady steps, but the current tree has no replay trace for this path. Require distinct-shape `qsa-mblk:` records and repeated hot replay after MTP rewind/reset.

Static review cannot prove actual table/cell parity, CUDA graph replay, device ordering, or the restore/rollback paths. In particular, the old note's cellsel reasoning is not a validation result.

## Existing fixtures and runs

- Current experiment `fixtures/qsa-dense-fattn-tail.cpp` and its README only test a small CUDA dense FlashAttention tail against an FP32 reference. They do not construct QSA scores, multi-sequence cell maps, or cache lifecycle state. No QSA-multi source fixture/test was found under current `fixtures` or the private source tests; no current `notes`/`results` file contains `qsa-mblk` or a `LLAMA_QSA_POOLED_MULTI` execution record.
- Historical `experiments/codex-xxs-20261011-1003/p6-pooled-sparse/feature-env.sh` enables pooled-multi + cellsel + sparse + MIN_CTX=0, but uses the obsolete `LLAMA_QSA_MBLK_TRACE` name. Its server log only confirms allocation (4 sequences x 4096 pooled rows at lines 249, 255), not fast-path graph shapes. `suite.json` has one-, two- and four-stream request results and hashes/text, but no baseline token-ID comparison. It combines the experimental sparse attention arm and does not qualify sequence isolation or exact generation.
- Historical `experiments/hw-orchestrate-20261014/run-a3` sets pooled-multi=1, `LLAMA_QSA_GATHER_MAX_TOKENS=32`, `LLAMA_QSA_SPARSE_KERNEL=1`, and `LLAMA_QSA_INDEXER_IMPL=fused`, but omits cellsel, MIN_CTX and `LLAMA_QSA_MULTI_SPARSE` (so the multi sparse attention arm stays off). Its server log confirms allocation (4 x 8256 rows at lines 250, 256). The 2 x 8K case is below the default 32K gate; the 4 x 8K case could pass depending on ubatch active sequences, but the log has no current `qsa-mblk:` trace. Its controller stores `output_sha256`, not token IDs/text parity. `run-a4b-carry-validation` is a 4 x 256-token carry test under the same 32K gate and cannot exercise multi fast selection. These are not current-source semantic proof.
- Historical 2026-10-02 `m-cellsel-2x8k` did run pooled-multi + cellsel + MIN_CTX=0 but recorded `parity=False` (TG 21.45/16.88). See local `flashnext-llama-experiment/TECH-FACTS.md:93-97`. This disproved the original claim that cellsel alone restores 8K parity and speed. Requalify only against the current frozen runtime and same indexer implementation in both arms.
- The older 2 x 64K pooled-multi result (+74% reported, and B+H parity in the historical report) is evidence for an earlier binary/config only. It does not qualify the current private source, current cellsel mode, or current runtime.

## Minimal gates

1. **Local CPU fixture:** no relevant fixture is already available to run. The smallest meaningful no-GPU addition would exercise the production host bookkeeping/planner with two interleaved sequences, ratio 4, and synthetic cache cells. Assert disjoint per-sequence member/visibility windows, selected cell IDs, partial-bucket rollback/re-decode, `clear`/`state_drop`/rescan, and demotion for shared-token or gappy positions. A standalone ggml CPU cellsel graph can additionally check repeat/get_rows flat-index layout, but it must consume the production plan; a duplicated toy algorithm is not an adequate gate.
2. **Full-model correctness before throughput:** use the exact currently qualified model/runtime and same `LLAMA_QSA_INDEXER_IMPL` in both arms; 2 concurrent 8K prompts, fixed seed/temp 0, 512 token IDs per slot; baseline multi=0 vs candidate multi=1/cellsel=1/MIN_CTX=0. Capture `qsa-mblk:` shapes with `cellsel=1`, table-rescan/truncate logs, first token divergence, MTP accepted counts, and exact per-slot IDs. Then exercise slot reuse/state clear, MTP rejection rewind, shared-prefix/seq-copy, whole-state restore and context shift; require safe demotion/rescan where the code intentionally falls back and no cross-slot cell IDs.
3. **Only after that semantic gate:** repeat eligible 2 x 64K with MTP and direct per-step compute timing, retaining accepted-token counts and output IDs. Compare with multi disabled; separate throughput from semantic and graph-capture evidence. Check configured per-slot capacity: the fast path has a fixed `share = pooled_rows / n_seq_max` and demotes when one sequence exceeds `share * ratio` (`llama-memory-hybrid-idx.cpp:140-150,762-765`).

## Recommendation

Treat current multi QSA as an isolated candidate. Cellsel is required for reference cell-budget semantics, but the only historical 8K cellsel run failed token parity; no current CPU fixture or current-source multi trace closes that gap. It can reduce per-step full-history pooling work at long contexts, but it does not reduce the block-score matrix itself. Keep the standard/or current qualified route until the exact-token, rewind/reset and graph-replay gates above pass.
