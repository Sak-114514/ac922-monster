# Current reserve-class candidate audit

Date: 2026-10-16 UTC  
Scope: read-only comparison of donor commit `da3fbb65d926a2bb10618b4dfe945e9f9bf18ea5` with the private AC922 source and saved V32 timing artifacts. No source, build, runtime, or GPU changes were made.

## Decision

Do **not** port the donor's reserve-class cache as the next V32 performance patch. The current V32 trace proves substantial graph-build, allocation, and input-setting cost, and it records graph rebuilds when shapes vary. It does **not** prove that those calls repeatedly perform full scheduler reservations. In this source, batch-shape changes drive graph reuse/rebuild decisions; `sched_reserve()` is a separate, dirty-flagged operation. A possible allocator growth/re-reserve remains unmeasured because V32 did not enable the existing scheduler allocation trace.

The donor patch addresses a different mechanism: its on-demand pipeline-parallel reservation remembered only the most recent `(n_tokens, n_seqs, n_outputs)` class. Alternating prefill/decode classes therefore repeated a full-context reserve and drain. The commit keeps a set of classes and skips that duplicate reserve when a prior class covers the request. Its commit message reports 0.6–0.9 s per switch in that donor branch. That result is not evidence that the same path is active in AC922 V32. [Donor commit](https://github.com/lukolszewski/llama.cpp-multigpu/commit/da3fbb65d926a2bb10618b4dfe945e9f9bf18ea5)

## What V32 actually measured

The saved timing breakdown is from V32's clean run at:

`fixtures/dual-context-v32-parent-walkfast-2bank2-20261016/results/root-p4-v32-parent-walkfast-shortctx16k-hierarchical-ep6-d3-20261016t052412z/`

`LLAMA_STEP_TIMING_PERIOD=32` reports per-`process_ubatch` averages across each 32-call window. In the target windows with token counts 8–1024, the counters were:

| Target context | Rebuilds / 32 calls | Build avg | Alloc avg | Input avg | Apply + build + alloc + input avg |
|---|---:|---:|---:|---:|---:|
| bank 0 | 7 | 548.398 ms | 81.596 ms | 584.426 ms | 1,223.814 ms |
| bank 1 | 7 | 670.592 ms | 125.351 ms | 500.482 ms | 1,330.504 ms |

These are **amortized per-call averages**, not a single reserve duration. `step-timing` increments `n_build` only on the graph-build branch (`src/llama-context.cpp:3287–3290`) and divides all accumulated stage counters by the 32-call period when printing (`:3368–3380`). Thus seven of the 32 calls rebuilt; the other calls may still incur input-setting work. The numbers identify a real host-side graph/build/alloc/input problem, but do not identify a repeated reservation as its cause.

V32's exact environment enables `LLAMA_PREFILL_LAZY_RESERVE=1` and `LLAMA_PREFILL_SEPARATE_STREAM_RESERVE=1`, with target `ROOT_UBATCH=1024`. The runtime logs four lazy-reserve selections at 256 tokens against a 1024-token max (`lazy-prefill-workspace`), plus eight `sched_reserve: reserving ...` entries over the clean runtime log. It contains **zero** `sched-alloc-timing` entries and zero `re-reserve for` warnings. The eight explicit reserve entries are not a per-ubatch event stream and do not carry the requested batch-shape tuple, so they cannot establish reserve churn on alternating `n_tokens`/`n_seqs`/`n_outputs`.

`GGML_SCHED_ALLOC_TIMING` is absent from V32's exact environment. In current source it is read once by a function-local static during `ggml_backend_sched_alloc_graph()`, and only logs calls taking at least 25 ms (`ggml/src/ggml-backend.cpp:3713–3748`). Its trace would have reported `backend_ids_changed`, whether the allocator re-reserved, and separate wait/reserve/retry durations.

## Current source behavior

1. **Full context reserve is not keyed by each ubatch.** `llama_context::sched_reserve()` returns immediately unless `sched_need_reserve` is set (`src/llama-context.cpp:1292–1297`). Context setup calls it (`:1133`); decode/encode entry also calls it (`:3478`, `:3791`), but ordinary batch-shape changes do not mark the flag. The flag is set by graph/configuration changes such as next-N embeddings, layer-input embeddings, causal attention, sampler/offload changes, LoRA, and control vectors (`:2524–2545`, `:2551–2560`, `:2576–2628`, `:2633–2688`). When dirty, a reserve synchronizes the context and performs PP/TG/PP graph reservations (`:1301–1422`).

2. **The shape variation does cause graph rebuilds and tensor-split drains.** `process_ubatch()` accepts reuse only when the current result's graph parameters pass `res->can_reuse(gparams)` (`src/llama-context.cpp:3182–3193`). The reusable result pool is only two entries, selected by whether `n_outputs` is nonzero (`src/llama-context.cpp:4754–4759`; storage `src/llama-context.h:563`). The graph reuse predicate checks exact token/sequence/output and sampler-related structure, including `n_tokens`, `n_seq_tokens`, `n_seqs`, `n_seqs_unq`, outputs, sequence IDs, and MTP-step shape (`src/llama-graph.h:903–989`). This is deliberately stricter than a reservation-class coverage test.

3. If reuse fails, tensor-split execution synchronizes before resetting the scheduler/result and building a new graph (`src/llama-context.cpp:3250–3308`). That can explain repeated **graph rebuild/drain** cost as shapes move between prefill and decode. It is not equivalent to `sched_reserve()` and the donor's shape-class cache would not suppress it. V32 has `LLAMA_MTP_DRAFT_GRAPH_CACHE=0`; target result reuse remains the small two-entry pool above.

4. **Allocator fallback is separate and conditional.** Every new graph is passed to `ggml_backend_sched_alloc_graph()` (`src/llama-context.cpp:3303–3308`). The scheduler first tries `ggml_gallocr_alloc_graph()` if backend buffer types did not change. Only when that attempt fails or backend buffer types changed does `ggml_backend_sched_alloc_splits()` synchronize the backends, call `ggml_gallocr_reserve_n()`, and retry allocation (`ggml/src/ggml-backend.cpp:1986–2049`). This is where a larger graph may grow the high-water reservation. Skipping that fallback on the assumption that a prior class covers the graph would be unsafe; actual allocator success is the capacity/lifetime gate.

5. **The donor's direct pipeline use is absent.** V32 is the nested EP6 tensor-split profile with pipeline parallelism disabled. Current context construction permits pipeline parallelism only for layer split (`src/llama-context.cpp:1100–1105`), and current source has no `reserve_on_demand`, `reserve_key_t`, or `reserve_keys` path. The donor's exact shape-class list and its 3090 pipeline measurements therefore cannot be transplanted as-is.

## Safe next step and patch boundary

First collect allocator evidence in a future already-authorized V32-like run by adding `GGML_SCHED_ALLOC_TIMING=1` **before context/backend initialization**. Keep the existing source/binary and all semantic gates unchanged. Correlate `sched-alloc-timing` with `step-timing` by context/scheduler and time window; material repeated `reserved=1` events would justify a narrower investigation. Because the current trace omits `n_seqs` and `n_outputs` from allocator records, include those fields in any future opt-in call-site diagnostic if a precise shape-key analysis is needed.

Do not add a cache that makes `ggml_gallocr_alloc_graph()` skip its real allocation attempt, suppresses a required `reserve_n`, relaxes `llm_graph_result::can_reuse()`, or drops sampler/MTP/sequence identity from graph compatibility. If profiling shows only `model.build_graph()`/input-setting and graph drains, the donor reserve cache is ruled out as a remedy; investigate graph construction or safe reuse separately. If repeated full `sched_reserve()` calls are actually observed for the same context/configuration, identify the caller that dirtied `sched_need_reserve` first; only then consider a default-off, per-context reservation-class optimization, retaining exact `n_seqs` and invalidating on every graph/configuration/backend-generation change.

## Evidence provenance

- V32 exact environment: `fixtures/dual-context-v32-parent-walkfast-2bank2-20261016/v32-parent-walkfast-exact-env.json` (does not set `GGML_SCHED_ALLOC_TIMING`).
- V32 stage totals and rebuild interpretation: `notes/v32-timing-breakdown.md`, especially the 32-call target timing table and source definition above.
- Remote V32 clean runtime-log counters: 8 `sched_reserve: reserving ...`, 4 lazy-prefill selections, 0 `sched-alloc-timing`, and 0 `re-reserve for` lines.
- Local and remote private source hashes matched during this audit: `llama-context.cpp` `a402784401300f4e333205e0ab592804e8073f1d2d718017f04391d2a1e05f57`; `llama-graph.cpp` `bedf928940be6ef6e8e5f78aaa0027ca52f17cb53d80827fd10b2e503ade3d68`; `ggml-backend.cpp` `7aab7ecc95489eb53e0e3de92c964cd84707972d9c8f5d94bd4c37dcae47adb0`.
- Donor mechanism and its own measured scope: commit `da3fbb65d926a2bb10618b4dfe945e9f9bf18ea5`.

## Addendum: what the V32 rebuild counter and V35 lane padding establish

### V32: exact reject causes are not present in the saved log

The V32 clean runtime's target-context windows report `rebuilt=7` in each 32-call period, with `tokens=8..1024` (`runtime.log` lines 4916 and 4927). In this code, that counter increments when `process_ubatch()` does not take the `res->can_reuse(gparams)` branch and rebuilds the graph (`src/llama-context.cpp:3182–3193, 3287–3290`). The period only logs min/max token counts, not each rejected old/new parameter pair (`:3060–3070, 3368–3380`). Neither V32's exact environment nor its runtime log contains `LLAMA_GRAPH_RESULT_DEBUG=2` or `graph-reuse-reject` records. Therefore the saved evidence cannot map each of the seven rebuilds to one exact field; the following are source-supported candidate signature transitions, not a per-rebuild verdict.

The V32 runner gives a concrete explanation for repeated shape changes within that window:

- Each prompt is one logical `llama_decode()` call with one sequence, and only its final token requests output (`fixtures/dual-context-v32-parent-walkfast-2bank2-20261016/dual-context-v24-shortctx.cpp:452–476`). At target `UBATCH=1024`, an 8K prompt is split into 1024-token ubatches: interior ubatches have no output, while the last one has one output.
- The following D3 target verification has up to two sequences and at most eight rows; the V32 runner inserts one seed plus actual draft rows and does not pad each lane to a fixed row count (`:634–647`). Its `n_tokens`, `n_seq_tokens`, active sequence layout, and `n_outputs` can differ from the prompt ubatches.
- `get_gf_res_prev()` keeps only a zero-output result and a positive-output result (`src/llama-context.cpp:4754–4759`; storage `src/llama-context.h:563`). The prompt's final one-output graph and multi-lane verify graph therefore occupy the same positive-output slot. Reuse still requires exact graph compatibility; it does not treat that slot as a reserve class.

This makes prompt/verify signature changes a strong explanation for some rebuilds. The source also has input-level reject conditions: QSA attention-mask dimensions compare exact `n_kv`, rows per stream, and stream count (`src/llama-graph.cpp:63–79`); recurrent inputs compare `n_rs`, sequence counts, `head`, and `rs_z` (`:404–419`, `:1318–1337`). A restored or shifted recurrent-cell layout can therefore reject reuse even if ubatch dimensions match. V32 has no `failed_inputs` detail, so the exact count caused by each signature/input difference is unavailable.

The `nextn` extraction setting is **not evidenced as a per-call cause**. V32 enables target extraction once in bank setup with `llama_set_embeddings_nextn(target, true, false)` (`dual-context-v24-shortctx.cpp:351–365`). `common_speculative_init()` requests the same target mode (`src/common/speculative.cpp:520–529`); `llama_context::set_embeddings_nextn()` marks reserve dirty only if its flags change (`src/llama-context.cpp:2524–2534`). The graph reuse signature does compare `embeddings_nextn` and `embeddings_nextn_masked` (`src/llama-graph.h:991–1001`), so a real mode transition would matter, but this run shows one stable setting rather than toggles.

For a diagnostic-only future authorized run, `LLAMA_GRAPH_RESULT_DEBUG=2` produces bounded `graph-reuse-reject` records with old/new batch summaries, parameter differences, and failed input names/state. This diagnostic covers `DEFAULT` and `DECODER`, not `DECODER_MTP` (`src/llama-graph.h:45–48`, `src/llama-graph.cpp:1498–1513, 1757–1833`), so it can adjudicate target rebuilds but not generic MTP tail rebuilds. Neither V32 nor V35 enabled it; each runtime log has zero `graph-reuse-reject` records.

### V35: the four-active target graph is stable; rebuilds move to the EOS tail

The V35 V80 runner opts into four target rows per active lane for D3. `target_verify_pad::make_plan()` appends one seed and fills each active lane to three draft/pad rows (`fixtures/dual-context-v35-onebank4-combined-20261016/target-verify-pad-plan-v35.h:12–13, 97–109`). The runner submits every planned row with output requested, including pad rows (`dual-context-v35-onebank4.cpp:662–697`). Thus graph token/output counts are 16 for four active lanes and 12 for three active lanes; this fixes the shape while four lanes are active, but a lane-count transition changes `n_tokens`, `n_seqs`, `n_seqs_unq`, `n_outputs`, and possibly the sequence-ID vector.

V35's saved V80 runtime directly confirms the main window:

- Frames 281–312 are **32/32** `active_lanes=4`, `total_rows=16` target submissions.
- The target context's `step-timing` window at `n=416` reports `rebuilt=0 tokens=16..16` (runtime log line 151298).
- The same 32 frames' preceding D3 MTP entries all report `lanes=4 depth=3 rebuilt=0`.
- Frame 313 moves to three lanes and 12 rows. Later, the mixed EOS tail window at `n=448` reports `tokens=4..16 rebuilt=18` (line 164188); it mixes lane-count/row-shape transitions and is not evidence against the steady four-active path.

The V35 tail also has a separate generic-draft cause. Around the four-to-three-lane transition, the log records `MTP catch-up/seed fusion eligible lanes=3`, then `unrolled depth=3 downgraded reason=depth-3-unsupported-lane-count`, followed by `compact MTP lane layout changed during GPU chain` and fallback to generic MTP. This matches the policy: D3 supports two lanes, or four lanes only with `LLAMA_MTP_UNROLL_D3_LANES4=1`; three-lane D3 remains unsupported (`src/src/llama-mtp-lane-policy.h:26–34`). The later draft-context timing windows show `rebuilt=32 tokens=1..10` and then `rebuilt=32 tokens=1..5` (`runtime.log` lines 159721 and 164277). A concrete source signature in this generic fallback is `nextn_layer_offset`: the per-head fallback sets it to `i` before each draft-head `llama_decode()` and restores it to zero at the end (`src/common/speculative.cpp:2554–2569, 2689–2691`), while `llm_graph_params::allow_reuse()` requires an exact offset match (`src/src/llama-graph.h:986–989`). That prevents reusing one head's graph for another head's decode. Variable token/output and memory signatures can add further rejects; because MTP is excluded from `graph-reuse-reject` diagnostics, the saved run cannot prove how much of the 32/32 is due to offsets versus those other fields. This is generic MTP fallback/tail work, not the four-active unrolled D3 graph, whose per-frame logs show `rebuilt=0` for frames 281–312.

### Cache status and bounded candidate, if future evidence justifies one

There is still no general per-shape graph-result cache. Target code retains only the two zero/nonzero-output results described above. `LLAMA_MTP_DRAFT_GRAPH_CACHE` is off in both V32 and V35, and the unrolled MTP source explicitly avoids the previously broken family-cache restore path (`src/llama-context.cpp:2865–2874`). Do not enable it to chase these counters.

If future per-reject diagnostics show recurring exact target signatures, a bounded first experiment could cap a per-context LRU at **five entries** for V32's two-sequence workload: zero-output prompt ubatches for sequence 0 and 1, one-output final-prompt ubatches for sequence 0 and 1, and the two-lane verification shape. This is only a cap for those known common families; verify rows, sequence IDs, or memory state outside those signatures must miss and use the existing rebuild path. The key must not be only `(n_tokens,n_seqs,n_outputs)`: include `n_seq_tokens`, `n_seqs_unq`, equal-sequence state and sequence IDs, token-vs-embedding input kind, output layout, graph type, samplers, static cparams including both nextn flags and `nextn_layer_offset`, MTP step signatures where applicable, and the memory-input shape/state values checked by each input (`n_kv`/stream dimensions plus recurrent `n_rs`, `head`, and `rs_z`). A cache hit must still pass the existing `can_reuse` checks. If dynamic state signatures exceed the cap, evict/fallback; do not merge them to force hits.

Each cached family would need independently owned scheduler/galloc storage and Meta descriptor/owner-generation lifetime, or another proven disjoint allocation scheme. Keeping three `llm_graph_result` objects while reusing the current single scheduler is unsafe: a rebuild calls `ggml_backend_sched_reset()` and reuses scheduler allocation/descriptors (`src/llama-context.cpp:3258–3308`). Separate schedulers/buffers may consume substantial HBM; no V32 headroom ledger establishes that cost. For V35's 4-active target graph there is no demonstrated cache target (`0/32` target rebuilds); the `T16→T12` transition is a one-way lane-completion tail. The proposed cache is therefore a design boundary, not an implementation recommendation without per-reject diagnostics and a measured memory budget.

### Provenance

- V32 `graph-reuse-reject` count: zero; exact environment omits `LLAMA_GRAPH_RESULT_DEBUG`.
- V35 `graph-reuse-reject` count: zero; V80 runtime source hashes for the relevant current files match the saved build report: `llama-context.cpp` `a402784401300f4e333205e0ab592804e8073f1d2d718017f04391d2a1e05f57`, `llama-graph.cpp` `bedf928940be6ef6e8e5f78aaa0027ca52f17cb53d80827fd10b2e503ade3d68`, `llama-graph.h` `eb0fcc19fda61cab31b3e38f70d7f3019feb75b55cfbc15ecbdd6bc40da6bb94`, `common/speculative.cpp` `c8b04bb430ca86050c956d944580bc3909a4d7df42a701a541cad0b1e3882188`, and `llama-mtp-lane-policy.h` `7fa380bf4bddb28c1afd8a1864d958f6055e20a9309efafa39e131e8f6267786`.
- V35 log: `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v35-onebank4-combined-20261016/results/root-p4-v35-onebank4-v80-ep6-d3-16k-20261016t073508z/runtime.log`.
