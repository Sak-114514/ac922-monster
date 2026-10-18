# Recurrent-state preservation across four sequential prefills

Date: 2026-10-16 UTC  
Scope: read-only audit of the V80 target-only T4 versus P4/T16 logits capture and the private recurrent allocator/GDN/PLE state paths. No source changes, build, model launch, or GPU use was performed.

## Finding

The V80 forced-target capture confirms a real target-compute mismatch: all 16 matched F32 logit rows differ between T4 and P4/T16 (per-row max-absolute range about 1.12–4.89 and RMS about 0.21–0.72), while all raw CPU and GPU argmax IDs match and all 248,320 logits per row are finite. The capture uses one target context, clears memory between arms, and performs no MTP proposal, accept, or rollback (`fixtures/verify-forced-t4-vs-p4t16-20261016/results/root-forced-target-verify-v80-t4-vs-p4t16-20261016t081127z/run.json`). It compares four serial one-sequence T4 calls with one four-sequence T16 call, so it does not isolate allocator behavior from batched target-compute behavior.

The V80 runtime’s before-MTP prefill F32 CRCs match between serial and P4 for each flow: `real8k-0=61a207e3`, `real8k-1=41b303ad`, `real8k-2=d0707a7d`, and `real8k-3=d96412b5`. The P4 runner records each flow’s logits immediately after that flow’s own target prefill. These CRCs are consistent with the stated passing per-lane prefill gates; they do not prove that a lane’s recurrent payload stayed unchanged while later lanes were prefetched.

I found **no concrete recurrent allocator bug** in the fresh V80 sequence order 0→1→2→3. The source predicts that each single-sequence prefill owns a separate recurrent cell and separate SSM/R-convolution/PLE-history rows, then the first T16 `find_slot()` sees the same row order and need not move those cells. This rules out a simple “adding seq1–3 compacts/shifts and overwrites seq0” explanation for the inspected allocator path. It does not prove the target-only T16 computation is correct: batched target kernels, actual `s_copy`/state gather, the PLE/GDN graph, and the full allocator-to-operator binding remain untested by a CPU metadata audit.

## Source trace for the V80 sequence order

The forced schedule uses the V35 context contract: one context, four sequence slots, non-unified Q8 KV, four rollback snapshots per sequence (`n_rs_seq=4`), and four prompts assigned to sequence IDs 0–3 (`fixtures/verify-forced-t4-vs-p4t16-20261016/target-forced-schedule.json`; runtime confirms recurrent cache size 4 cells / 4 sequences / `rs_seq=4`). The T4/P4 arms both clear the same target memory with data before starting (`forced-verify-capture.cpp:199–202, 285–300`). The V35 prefill routine submits one full logical prompt call at a time, with only its final prompt token requesting logits (`fixtures/dual-context-v35-onebank4-combined-20261016/dual-context-v35-onebank4.cpp:459–483`). After this decode it reads/dumps outputs; speculative processing consumes target hidden rows and runs draft decodes on the distinct draft context (`:484–515`, `src/common/speculative.cpp:1815–1847`). Target rollback via `llama_memory_seq_rm(target, ...)` occurs after target verification acceptance, not while the four prompts are being prefetched (`dual-context-v35-onebank4.cpp:841–850`).

For each prompt, the production hybrid-memory allocator splits the logical batch while keeping recurrent rollback rows together, runs recurrent `prepare()` first, and returns a context containing the ubatches (`src/llama-memory-hybrid.cpp:69–121`). `prepare()` simulates `find_slot()` over the prompt’s ubatches and restores the cells/used/head metadata afterward (`src/llama-memory-recurrent.cpp:506–527`). It does not write recurrent tensor payloads. Each real ubatch then calls `mctx->apply()` before graph construction; recurrent `apply()` calls `find_slot()` again (`src/llama-context.cpp:3106–3110`, `src/llama-memory-recurrent.cpp:1288–1305`). Although `prepare()` does not restore its look-ahead values for `n`/`rs_z`, the subsequent real `apply()` recomputes them before the graph reads those fields.

With a fresh size-4 recurrent cache and ordered single-sequence calls:

| Prompt | Fresh empty cell selected | Existing sequence rows that `find_slot()` should leave intact |
|---|---:|---|
| seq0 | 0 | none |
| seq1 | 1 | seq0/cell0 |
| seq2 | 2 | seq0/cell0, seq1/cell1 |
| seq3 | 3 | seq0/cell0, seq1/cell1, seq2/cell2 |

The allocator starts its empty-cell search at `head`, uses the first empty cell for a sequence with no tail, and only swaps tail-cell metadata when a destination cell differs from that sequence’s existing tail (`src/llama-memory-recurrent.cpp:603–680`). Ordered seq IDs 0–3 on a fresh cache make the empty cell equal the seq ID. For each new lane, its empty cell is also the only zero-refcount candidate within that single-lane range, so `rs_z` points at the new lane's row; `build_rs()` zeros that row before gathering it (`src/llama-memory-recurrent.cpp:704–733`, `src/llama-graph.cpp:4071–4080`). Older cells have their own source references and are outside that range. Within one sequence’s later prefill ubatches, its tail cell is uniquely owned, so `has_cell` avoids allocating or moving it. No recurrent tensor payload copy occurs in `find_slot()` itself.

After all four prompts, the first T16 batch has four contiguous four-token sequences. Their tails are cells 0–3 in the same order, so the gather/reorder loop’s destination `s + min` equals each existing tail; it performs no swaps (`llama-memory-recurrent.cpp:613–680`). `s_copy()` maps each current row through `src0` and, for rollback, through `rs_idx * mem_size + src0` (`:1346–1370`). With `mem_size=4`, each sequence’s five rollback rows occupy its own stride of four recurrent cells.

SSM, convolution history, and PLE history share this allocator mapping:

- The recurrent allocator creates `r_l` and `s_l` rows of `mem_size * (1 + n_rs_seq)` and a separate PLE row tensor `p_l` when a layer has PLE convolution state (`llama-memory-recurrent.cpp:104–117`). The V80 runtime confirms `n_rs_seq=4` and 4 logical cells.
- The graph fills `s_copy` once per recurrent input; `llama_memory_recurrent_context::s_copy()` supplies the source row and one-shot rollback offset (`llama-graph.cpp:388–402`, `llama-memory-recurrent.cpp:1346–1370`). The recurrent-state graph zeros the selected `rs_z` row if one exists and gathers each sequence’s state (`llama-graph.cpp:4056–4089`).
- Qwen GDN R-history and PLE convolution-history writes use `slot * mem_size + kv_head` for destination row addressing (`models/qwen4exp.cpp:3352–3404`, `:3491–3524`). In the fresh seq0→seq3 prefill, `kv_head` is the allocated cell for that sequence; later sequences should write their own rows.

The prior GDN X4 batch-vs-serial fixture remains narrower than this question: it initialized valid synthetic state rows and compared the CUDA GDN operation, but did not obtain its rows from `llama_memory_recurrent::find_slot()`. The earlier V30b audit documented this same boundary; its CPU/GPU GDN probe does not validate the allocator-to-operator binding.

## Precise CPU allocator fixture point

The closest reusable host fixture is `fixtures/memory-frame-multiubatch-20261015/multiubatch-freeze-test.cpp`. Its `make_memory()` currently passes `no_layers` to attention, recurrent, and indexer filters (`:64–74`), so it tests real cell/split metadata but allocates no per-layer recurrent state payload. The V80 gap can be closed locally without loading weights or touching CUDA:

1. Add a separate fixture mode/helper that loads the real GGUF with `vocab_only=true`, `n_gpu_layers=0`, and `CUDA_VISIBLE_DEVICES=""`, then constructs the production `llama_memory_hybrid_idx`/`llama_memory_recurrent` with `mem_size=4`, `n_seq_max=4`, `n_rs_seq=4`, and F32 R/S storage. Reuse the current fixture’s GGUF metadata setup, and explicitly populate any recurrent/PLE shape and per-layer flags that vocab-only loading leaves unset. Select one real GDN layer plus one PLE-conv layer (or a layer satisfying both predicates) in the recurrent layer filter. Keep attention/indexer layer tensors filtered; retain their real metadata/cell bookkeeping.
2. For seq IDs 0,1,2,3 in order, submit one logical prompt per sequence using multiple ubatches. Before each allocator prepare/apply, snapshot all earlier sequences’ five S/R/P row slices and cell metadata. After apply, assert that the new sequence’s `s_copy` resolves to its own expected cell/rollback stride and that no earlier cell metadata or payload changed. Write unique CPU sentinel patterns into the current sequence’s destination rows to make a later cross-row overwrite observable.
3. Submit one T16 batch (four sequences × four rows) against the populated allocator and assert the `find_slot()` result maps tails to rows 0–3 without a metadata swap; read `s_copy` for all four sequences and verify current/rollback indices, R/S/P canaries, and cell pos/tail/seq-id ownership.

This fixture would give direct evidence about allocator row assignment, sequential slot preservation, and rollback-stride mapping. It would **not** validate the full GDN/PLE compute graph or explain the remaining target F32 difference. If it passes, the next discriminator is a target-graph test that uses the real allocator-generated state rows for serial and T16 batches, followed by a per-row comparison of `s_copy`, PLE-history IDs, and GDN state outputs. That must be a separately authorized model/GPU test; this audit did not run it.

## Evidence provenance

- Forced target run: `fixtures/verify-forced-t4-vs-p4t16-20261016/results/root-forced-target-verify-v80-t4-vs-p4t16-20261016t081127z/analysis.json`, `run.json`, and `target-forced-schedule.json`.
- V80 prefill CRCs and per-lane order: `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v35-onebank4-combined-20261016/results/root-p4-v35-onebank4-v80-ep6-d3-16k-20261016t073508z/runtime.log`.
- Existing allocator/GDN boundary: `fixtures/dual-context-v30-onebank4d3-20261016/V30B-DIVERGENCE-TRACE.md`, sections “Row and state mapping in the current source” and “First diagnostic gate: GDN X4 batch versus serial”.
- Recurrent cache parameters and rollback stride: V80 runtime lines logging `size = 2251.41 MiB (4 cells, 48 layers, 4 seqs, 4 rs_seq)`, plus source references above.
