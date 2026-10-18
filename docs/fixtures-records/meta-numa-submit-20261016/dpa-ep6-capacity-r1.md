# DPA2×attention-TP3 + EP6 capacity/design estimate

**Scope:** read-only capacity analysis from the integrated GGUF manifest, V49/V13 run artifacts, and the frozen runtime source. No model payload was read, no build or GPU process was run, and no core source was changed. This is a placement estimate, not an allocator fit result.

## Result

The proposed **two attention islands (each TP3, each serving two lanes) plus one six-GPU expert pool** has a reasonable memory rationale: under the stated placement assumptions—including an explicit hypothetical rebalance of the V13 GPU-resident PLE bytes across all six cards—its replicated/sharded static model footprint is about **10.00 GiB/GPU**, versus **12.34 GiB/GPU** for six full dense/attention replicas sharing EP6 experts. The DPA2 estimate saves about **2.34 GiB/GPU** in static weights by keeping dense/attention TP3 within each island.

That subtotal excludes KV, recurrent state, compute buffers, CUDA graphs, allocator slack, and routing scratch, so it **does not establish that either design fits**. More importantly, current code has two independent Meta3 devices; its EP split width is 3, not 6. DPA2 requires a new cross-island EP transport and reduction path. It would trade the present layer-pipeline boundary for routed expert exchanges in **all 48 MoE layers**. The hardware cost may therefore be synchronization and XBUS traffic rather than GPU memory.

The 4-stream result used as context is diagnostic only: the V49 artifact reports 963 emitted tokens in 5.858726 s (**164.370206 shared tok/s**), `semantic_status=unscored`, and `profiled=false`. It is not a quality-qualified production baseline.

## Source data

- Model manifest: `/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf.monpack.manifest.json`.
- GGUF: 78,615,622,080 bytes, 1,256 tensors; tensor payload source bytes total 78,604,595,712. The 48 target layers contain 36 GDN and 12 QSA layers. The separate MTP layer is `blk.48`.
- Current run: `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v13-prep-20261016/results/p4-d3-20261015T200425Z/` (V49 runtime, model and cache telemetry retained there).
- The PLE location is directly logged at `runtime.log:193-196,1925`: 8 GiB GPU-resident stripes are mapped to GPU logical 0/1/2 at NUMA 255/254/253; the remaining 18.82 GiB is host DDR on nodes 0/8. The log does not state each card’s exact PLE-byte count.
- Reproducible metadata-only calculator: `dpa-ep6-capacity.py`; its output is `dpa-ep6-capacity-ledger-r1.json`. The calculator validates the 1,256-tensor manifest, 144 target expert tensors, 3 MTP expert tensors, and all seven target expert formats.

### Model bytes from the manifest

| Weight group | Bytes | GiB | Shape in proposed placement |
|---|---:|---:|---|
| Target routed experts, 48 layers | 42,912,972,800 | 39.9658 | EP6 average 6.6610 GiB/GPU |
| Target GDN experts | 32,505,856,000 | 30.2734 | EP6 average 5.0456 GiB/GPU |
| Target QSA experts | 10,407,116,800 | 9.6924 | EP6 average 1.6154 GiB/GPU |
| Target dense/attention/other | 3,205,701,120 | 2.9855 | TP3 average 0.9952 GiB/GPU in DPA2; replicated for DP6 |
| Target shared experts | 121,077,760 | 0.1128 | TP3 average 0.0376 GiB/GPU in DPA2; replicated for DP6 |
| Global tensors (output, embedding, HC) | 723,343,360 | 0.6737 | DPA2 assumes output head split TP3 and remaining global tensors mirrored |
| PLE table `per_layer_token_embd.weight` | 28,800,138,240 | 26.8222 | V13 hybrid placement: 8.00 GiB on GPU logical 0/1/2 (NUMA 255/254/253) plus 18.82 GiB host DDR; candidate bill rebalances the 8 GiB across six |
| PLE auxiliary layer tensors | 65,740,800 | 0.0612 | Assumed mirrored |
| MTP routed experts, `blk.48` (`Q8_0`) | 2,673,868,800 | 2.4902 | EP6 average 0.4150 GiB/GPU, if EP6 includes MTP |
| MTP remaining tensors | 101,752,832 | 0.0948 | Assumed replicated |

By layer family, GDN is much heavier in routed experts while QSA has the larger attention/indexer path:

| Layer family | Count | Routed experts GiB | Dense/attention/other GiB | Shared experts GiB | PLE aux GiB |
|---|---:|---:|---:|---:|---:|
| GDN | 36 | 30.2734 | 2.2439 | 0.0840 | 0.0612 |
| QSA | 12 | 9.6924 | 0.7416 | 0.0288 | 0 |

The target expert tensors use all seven formats; these are original tensor bytes and would remain unchanged:

| Format | Bytes across target model | GiB |
|---|---:|---:|
| IQ3_S | 9,371,648,000 | 8.7280 |
| IQ4_NL | 8,493,465,600 | 7.9102 |
| Q2_0 | 7,077,888,000 | 6.5918 |
| IQ2_S | 5,373,952,000 | 5.0049 |
| IQ2_XS | 4,849,664,000 | 4.5166 |
| IQ2_XXS | 3,892,838,400 | 3.6255 |
| IQ3_XXS | 3,853,516,800 | 3.5889 |

Uniform EP6 boundaries for 512 experts are `[85, 85, 86, 85, 85, 86]`. Summing each tensor’s exact contiguous expert slice gives target expert bytes of **6.635 GiB** on ranks 0/1/3/4 and **6.713 GiB** on ranks 2/5; the MTP expert slices are **0.413 GiB** or **0.418 GiB**. The average values in the table should not be mistaken for identical rank usage.

## Per-GPU static-weight proxy

The subtotal uses the *V13 observed hybrid PLE totals* (8 GiB GPU residency and 18.82 GiB host residency), but **assumes the 8 GiB GPU component is rebalanced evenly across six cards** for the candidate layout. V13 itself placed that 8 GiB only on GPU logical devices 0/1/2, mapped to NUMA nodes 255/254/253. The per-device byte split within those three is not logged. The subtotal also assumes uniform TP3 dense weights, output-head sharding, mirrored small tensors, and EP6 for both target and MTP experts.

| Per-GPU component | DPA2: TP3 attention ×2 + EP6 | DP6: dense/attention replica ×6 + EP6 |
|---|---:|---:|
| Target routed experts, EP6 | 6.661 | 6.661 |
| MTP routed experts, EP6 | 0.415 | 0.415 |
| Target dense + shared weights | 1.033 | 3.098 |
| Global output/embedding/HC | 0.402 | 0.674 |
| MTP non-expert weights | 0.095 | 0.095 |
| Rebalanced PLE GPU residency assumption (8 GiB / 6) | 1.333 | 1.333 |
| PLE auxiliary tensors | 0.061 | 0.061 |
| **Static subtotal** | **10.000** | **12.337** |

If MTP experts are replicated instead of EP6-sharded, add **2.075 GiB/GPU**: DPA2 becomes 12.076 GiB and DP6 14.412 GiB before dynamic allocations. If the entire 26.82-GiB PLE table is row-sharded over six GPUs in VRAM, that table alone costs **4.470 GiB/GPU**, 3.137 GiB more than the candidate’s rebalanced 1.333-GiB GPU budget. Relative to V13’s actual observed placement, it would be 1.803 GiB above the three-GPU island’s **2.667-GiB mean**, and would add 4.470 GiB to each GPU on the other island if both islands owned full row shards. Those are different layouts; the V13 log does not provide each card’s exact PLE byte count. Any full GPU row-shard also needs a route for each token’s hashed PLE rows to all owners.

The V13 runtime logged island weight buffers of 20.24 GiB and 25.89 GiB across three members (averages 6.75 and 8.63 GiB/member), plus Meta model-buffer lines of 7,438.72 and 9,341.12 MiB. Those logger values use different accounting layers and must not be added together. The PLE NUMA map records GPU logical 0/1/2 at NUMA 255/254/253; GPU logical 3/4/5 had no PLE GPU-resident stripe allocation in that map. The run’s 68-sample GPU telemetry recorded minimum free memory of 2,772/2,724/2,726 MiB on device indices 0/1/2 and 2,263/2,220/2,210 MiB on 3/4/5. That is actual headroom from the existing layout, not a measurement of either proposed topology.

## KV, recurrent state, and workspaces

V13 allocated **132,096 cells per bank** with `n_seq_max=2`, `kv_unified`, so the KV tensors use one stream shared by both sequences. The two independent banks do not combine into a single 264K request context. The cache line reports target QSA K+V 1,644.75 MiB and indexer K 205.59 MiB at 132,096 cells; MTP’s single-layer K+V is 137.06 MiB and indexer K 17.13 MiB.

| Cache estimate | Full logical bank | DPA2 per GPU (TP3 K/V, mirrored indexer) |
|---|---:|---:|
| Target QSA KV + indexer at 132,096 reserved cells | 1.807 GiB | 0.736 GiB |
| MTP KV + indexer at 132,096 reserved cells | 0.151 GiB | 0.061 GiB |
| Target QSA KV + indexer for two active 50K lanes (100K cells) | 1.368 GiB | 0.557 GiB |
| MTP KV + indexer for two active 50K lanes | 0.114 GiB | 0.046 GiB |

The 50K rows are active-token arithmetic; the V13 allocator reserves the configured 132,096-cell pool, so its resident cache cost is the first two rows even when the current prompts are shorter. For comparison, a deliberately pessimistic design that replicates four complete 50K sequence caches on every GPU uses **2.736 GiB/GPU** of target QSA KV+indexer; a single DP6 owner at 50K uses **0.684 GiB/GPU**. These figures describe cache duplication only, not total model memory.

GDN recurrent state is fixed-size rather than proportional to token-cache length. The runtime reports 1,125.70 MiB at two sequences and four recurrent-state rollback slots. A uniform TP3 estimate is 0.366 GiB/GPU for a two-lane DPA2 bank; a one-sequence DP6 estimate is 0.550 GiB/GPU. These are shape-based projections; the new ownership layouts have not been allocated or measured.

The same run reports target Meta compute buffers of 552.55 and 554.37 MiB, target CUDA-host compute of 106.05 MiB, MTP Meta compute of 272.89 MiB, and MTP CUDA-host compute of 9.89 MiB. These are Meta/backend buffer figures, **not additive per-GPU allocations**. Workspace, graph capture, temporary routing buffers and allocator fragmentation remain outside the static subtotal.

## What the current code supports—and what is missing

Relevant frozen-source locations under `/home/debian/ac922-monster/experiments/standard-opt-20261014/src`:

- `src/llama.cpp:274,331-340` sets the split-state `n_devices` to the first island’s size (3) and constructs two separate Meta devices, each with its own three-device child list. There is no current six-member Meta expert group.
- `src/llama-model.cpp:130-135,330-340,386-390,696-716` detects Qwen4Exp `_exps.weight` tensors and selects axis 2 for expert placement. Its split callback divides that axis over the current Meta group, so with `3,3` it is EP3 within a group, not EP6 across both islands.
- `ggml/src/ggml-backend-meta.cpp:1308-1343,2466-2485` handles `MUL_MAT_ID` with axis-2 expert weights, mirrored activation/route metadata, rank-local expert ranges, and final output reduction inside that Meta group. This does not route a bank’s activation into a second Meta group or combine six-rank expert partials.
- `src/llama-model.cpp:121-128,296-321` maps target `cache_k/v` to the attention output split (axis 0) and explicitly mirrors the one-head QSA indexer cache. `src/llama-kv-cache.cpp:86-88,214-240` creates `n_stream=1` for unified KV and allocates one K/V tensor set for the configured cell pool.
- `src/llama-model.cpp:316-321` maps recurrent R/S cache by the SSM output split. `src/llama-memory-recurrent.cpp:35-39,80-98,127-139` allocates per-layer recurrent buffers and reports the whole logical state size; no DPA sequence-owner allocator exists.
- PLE hashing runs on the host and produces row IDs (`src/models/qwen4exp.cpp:3214-3216`); the graph performs `GET_ROWS` on the shared table (`3375-3385`). `src/llama.cpp:319-327` has a separate PLE Meta6 device mode for the existing two-island topology, but DPA2 still needs a clear row-result path into both island-local attention graphs.
- The companion [nested Meta audit](nested-meta-audit-r1.md) checks whether an outer Meta2 can wrap two inner Meta3 groups. It finds a deterministic registry-lookup assertion plus a non-composing expert-offset contract; the 10/12-GiB subtotals above remain assumptions, not a tested recursive layout.

For DPA2, each of 48 MoE blocks needs: (1) the attention island’s hidden rows and route IDs/weights dispatched to the EP6 ranks; (2) only the selected expert work run by its owner; and (3) weighted expert partials returned and summed with that island’s TP3 shared-expert result. If activations are all-gathered, the **minimum full-row payload** for two decode tokens is 10 KiB per island per direction per MoE layer in FP16 (20 KiB in FP32); repeated over 48 layers that is 480 KiB one-way per bank-step, before return traffic, router metadata, TP collectives, or remote-owner fan-out. For a 1,024-token prefill ubatch, the same lower-bound row matrix is 5 MiB per layer and 240 MiB one-way across 48 layers. Top-10 routes may require copies to several expert owners, increasing that figure. These are payload arithmetic, not a measured traffic trace.

The existing layer pipeline has one island-to-island hidden-state boundary per model pass. DPA2 would remove that particular handoff but introduce EP dispatch/return at each of 48 MoE layers. Do not describe that as “communication removed”; it is a different, potentially more frequent communication pattern. DP6 removes attention TP3 synchronization and gives six independent attention owners, but replicates about 2.34 GiB/GPU more dense/global static weights than DPA2 under this bill and still requires each token’s MoE work to reach EP6 experts.

## Decision and next proof

- **Capacity:** DPA2 has the lower static-weight estimate. Its exact per-rank target+MTP EP payload is 7.048 GiB on four ranks and 7.131 GiB on two ranks; dense/global and PLE placement are still approximated. Both static totals leave the dynamic-memory question open.
- **Architecture:** the necessary implementation is a hierarchical scheduler with two persistent attention/sequence-owner Meta3 contexts plus a distinct six-rank expert group, explicit per-layer dispatch/return, and a defined final sum that includes TP3 shared experts. Current Meta3 `MUL_MAT_ID` support alone does not implement this.
- **KV:** DPA2 keeps each bank’s two-lane KV and recurrent state within its attention island; it does not need to copy both banks’ complete KV caches to all six GPUs. In contrast, the counterfactual “four full sequence caches on every GPU” wastes about 2.74 GiB/GPU at 50K/sequence.
- **Quality:** this design changes collective grouping and floating-point reduction order even without changing quantization. Require exact-logit/semantic and slot-reuse parity on a small real-weight path before full-model throughput claims; the current V49 4-stream result is `unscored`.
- **Smallest useful next experiment:** first implement or fixture one real MoE layer with two independent Meta3 attention owners and a Meta6 expert consumer. Validate all route owners, no missing/duplicated expert contributions, one final sum including shared experts, and payload/wait counts. Only after that should a 48-layer model be loaded. A CPU-only topology simulation cannot establish XBUS/NVLink latency or GPU fit.

## Verification performed

- `python3 -m py_compile dpa-ep6-capacity.py` — passed.
- Calculator was executed on `ac922` against the real manifest; it returned 1,256 tensors, 144 target expert tensors, 3 MTP expert tensors, all seven expected formats, DPA2/DP6 static subtotals 10.0004/12.3373 GiB, and the cache estimates above. The ledger now records V13’s actual first-island PLE map separately from the candidate 8/6 rebalancing assumption.
- `python3 -m json.tool dpa-ep6-capacity-ledger-r1.json` — passed.
- No CUDA/GPU command, build, model load, or source edit was performed.
