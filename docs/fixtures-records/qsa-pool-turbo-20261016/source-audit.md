# Source and history audit

## Exact implementation paths

- `src/src/llama-context.cpp:3002-3005`: current Turbo gate rejects any pooled cache (`get_blk()!=nullptr`, `blk_multi()`, and `LLAMA_QSA_POOLED_MULTI` all reject). Correct discriminator is memory mode: allow no pooled cache, or allow `get_qsa_multi() && blk_multi()`; reject the single shared pooled cache.
- `src/src/llama-memory-hybrid-idx.cpp:1592-1634`: frozen frame intentionally has `blk_ok()==false`, `blk_multi()==false`, and `qsa_multi_fast_ok()==false`; `qsa_multi_register()` and commit wrappers no-op under `frame_frozen`. Keep this isolation.
- `src/src/llama-memory-hybrid-idx.cpp:1675+`: `freeze_current()` currently rejects `mem->get_qsa_multi()`. It snapshots the seq's cell metadata from the applied indexer cache and returns the frozen context; the exact insertion point for per-seq drop is after all snapshot allocations succeed and before returning it.
- `src/src/models/qwen4exp.cpp:1180-1197`: ordinary QSA input set writes indexer K ids, calls `qsa_multi_register`, then fills ordinary cell/block metadata. This is the slow fallback rescan point.
- `src/src/models/qwen4exp.cpp:1521+`: `qwen4exp_qsa_mblk_fast()` eligibility is checked before graph input `set_input`; therefore after a frozen frame invalidates its ledger, the first mutable graph selects the ordinary path. Its `set_input` rescans/registers; a later fast graph can repool.
- `src/src/llama-memory-hybrid-idx.cpp:594-650`: `qsa_multi_rescan` rebuilds the host members table and sets `dirty_pb=0`, `valid=true`; this means table valid, not pool rows valid. A following `qsa_multi_fill` schedules all rows from bucket zero.
- `src/src/llama-memory-hybrid-idx.cpp:888-950`: `qsa_multi_fill()` creates `member_cells`/`touch_dst` and schedules `qsa_multi_pooled_to`; `src/src/models/qwen4exp.cpp:2460-2480` `SET_ROWS` returns the updated cache tensor; selection immediately `GET_ROWS` reads that updated tensor in the same graph.
- `src/src/llama-memory-hybrid-idx.cpp:1097-1108` and `1576+`: pool-row commit currently happens on a later `apply()`, not directly at graph completion. The graph error path in `src/src/llama-context.cpp:3370+` must invalidate affected sequences if compute fails, or an uncommitted pending row range could be treated as successful at a later apply.
- `src/src/llama-memory-hybrid-idx.cpp:500-590`: sequence removal/copy/shift/rollback already route through `qsa_multi_drop()` or `qsa_multi_truncate()`; keep those existing sequence-local lifecycle hooks.
- `src/src/llama-memory-hybrid-idx.cpp:1118-1190` and `1250-1390`: ordinary selector constructs per-cell `cell_blk`, complete block mappings, and causal mask. The current partial tail gets a `+1e9` score bias (the block-bias form uses `bid_idx >= tail_start`; cell-bias form uses `idx >= tail_start`).
- `src/src/models/qwen4exp.cpp:2496-2600`: mblk CELLSEL repeats each pooled row score r times, masks outside `[slice0*r, slice0*r+rank+1)`, then top-k selects `n_sel*r+r-1`; it currently has no matching `+1e9` tail priority. The source comment says the tail competes by score. This is a semantic mismatch, not a mere fp-order difference.
- `src/src/models/qwen4exp.cpp:1924-2105`: ordinary selector scores by first mean-pooling raw indexer keys within each complete ratio block, then key norm/RoPE and query dot score. In CELLSEL, repeated pooled row scores are therefore the correct per-cell scores for complete blocks. The injected-score CPU oracle tests this mapping, not the model's CUDA matmul rounding.
- `src/src/models/qwen4exp.cpp:965-970`: `LLAMA_QSA_MULTI_SPARSE` is opt-in. In mblk it switches attention consumption to the custom `FLASH_ATTN_EXT` sparse-cell path (see `src/ggml/src/ggml-cuda/qsa-attn.cu`). It can preserve selected cell IDs but changes accumulation order; leave disabled until GPU parity.

## Prior evidence and its limits

- `/Users/lyx/Desktop/ssh/flashnext-llama-experiment/GPT-HANDOVER.md:15-18` and `TECH-FACTS.md:19-21` report old B+H 2×64K TG 13.2/17.5→38.7/39.3, aggregate 30.7→77.9, with 512 generated token IDs exact. The run was a six-card TP recipe, not current Turbo/3+3.
- `TECH-FACTS.md:48-49` notes `LLAMA_QSA_F16_GATHER` was default-on and not isolated in the B+H binary. It does not establish current pooled/CELLSEL/Turbo compatibility.
- `TECH-FACTS.md:73-79` documents the old CELLSEL layout fix and the need to disable `MIN_CTX` for 2×8K.
- `TECH-FACTS.md:96` records the old `m-cellsel-2x8k` case failed parity (TG 21.45/16.88); this disproves universal parity, though it predates current runtime.
- `qsa-longctx/REPORT.md:62-70` says fair-share was required to reproduce concurrent overlap. Historical multi TG numbers cannot be compared without prefill overlap configuration.
- `TECH-FACTS.md`/current logs provide no current GPU evidence for CELLSEL with tail bias, canonical ties, Turbo invalidation, or `MULTI_SPARSE`; all are still qualification gates.

## Switch semantics

- `LLAMA_QSA_POOLED_MULTI=1` (`llama-memory-hybrid-idx.cpp:26-40`, constructor around 112-170) allocates a pooled row cache and per-sequence host member tables only when unified cache, `n_seq_max>1`, and QSA ratios are uniform. It enables eligibility for `mfast`; by itself the default `CELLSEL=0` is row top-k, which selects whole pooled blocks and changes the cell budget/selection granularity.
- `LLAMA_QSA_MULTI_CELLSEL=1` (`llama-memory-hybrid-idx.cpp:35-40`, `qwen4exp.cpp:1479-1485`) switches top-k to cell-space and repeats each complete block's score r times. It is required for the reference-width cell budget when `indexer_top_k % ratio == 0`; current source still misses the reference partial-tail priority and tie/order qualifications described above.
- `LLAMA_QSA_MULTI_SPARSE=1` (`qwen4exp.cpp:965-970`) only selects the custom direct sparse attention consumer after the `mfast` selector. It does not make candidate IDs more exact and should remain off during the selector/cache qualification.
- `LLAMA_QSA_MULTI_MIN_CTX` and `qwen4exp_qsa_gather_max_tokens()` only decide eligibility/dispatch. A demoted request is ordinary reference QSA and cannot be counted as a pooled/CELLSEL pass.

The old P6 record listed all three flags on but did not emit a qsa-mblk selection trace; it demonstrated allocation/benchmark setup, not proof of active cell selection or parity. The historic B+H proof was different: old multi path + H, a TP run, 512-ID parity, with F16 gather also enabled by default. Neither validates current Turbo freeze lifecycle.

## Physical-scatter patch review (staged, not applied)

The additional staged patch qsa-physical-cellsel.patch adds two per-query host inputs:

- cell_dst_ids[ratio*n_rows, n_tokens]: visible entries map compact row_cells positions to physical KV ids; all invalid/future/foreign/padded candidates map to unique scratch ids n_kv+j.
- cell_score_bias[ratio*n_rows, n_tokens]: +1e9 on the current partial tail, 0 elsewhere, matching set_input_qsa().

The Qwen graph repeats the pooled block score per member, adds this bias, scatters with ggml_set_rows() into an F32 score tensor of shape [1,n_kv+n_axis,n_tokens] initialized to -INFINITY, views the first n_kv physical rows, makes them contiguous, and calls the same ggml_top_k() op. Thus physical tie order is preserved when stable radix top-k is active, without a global logical-to-physical remap after selection. It explicitly demotes when top_k is not divisible by ratio or when canonical top-k is requested.

Static shape check against ggml_set_rows() assertions: base/source ne0=1, base/source ne2=n_tokens, source ne1=indices ne0=n_axis, source ne2 % indices ne1 = n_tokens % n_tokens, indices ne3=1. CUDA set-rows.cu dispatch handles F32/F32/I32 and the op is in the CUDA scheduler switch; no SM70 exclusion was found. Invalid destinations are unique per compact logical index, preventing duplicate writes within one query. Same scratch row ids may repeat across different query columns, but those are separate output addresses.

Correctness limits remain: the compact GEMM changes its row-count shape relative to the full reference GEMM, so near-tie score rounding must be checked on GPU. Also the top-k global scan returns O(n_kv) work and the fill/scatter/contiguous-copy add traffic; no speed claim is made. GGML_CUDA_TOPK_STABLE=1 is required. CUDA 12.4 headers on this host report CCCL_VERSION 2003002 (2.3.0), below the source's >=3.2 DeviceTopK branch; a future runtime with DeviceTopK must not assume the stable env is honored.
