# Prefill workspace and ubatch capacity audit

Date: 2026-10-15  
Scope: read-only source/log audit. No source edits, builds, model loads, or GPU tests.

## Findings

V12 reserves scheduler workspace for the full 132,096-cell cache and maximum configured ubatch at startup. This happens even though the first real prompts are 8K tokens. The physical KV/indexer/GDN backing stores are separate capacity allocations; the QSA graph workspace can be smaller for a short active KV view.

There is no existing environment variable that makes the workspace reserve start small and grow while retaining the full KV capacity. LLAMA_PREFILL_KV_PAD and LLAMA_DECODE_KV_PAD only change padding on runtime KV views.

## Actual memory headroom

- V12 target settings: n_ctx=132096, n_batch=2048, n_ubatch=1024, n_seq_max=4, unified KV.
- Startup target Meta compute buffers: 2180.98 and 2188.23 MiB; CUDA_Host compute buffer: 418.88 MiB. MTP draft has a separate 272.89 MiB Meta and 9.89 MiB CUDA_Host buffer at n_ubatch=32.
- Target backing: attention KV 1,644.75 MiB; indexer KV 205.59 MiB; recurrent state 2,251.41 MiB. These allocations preserve the configured cache/state capacity.
- The live V12 semantic-plus-benchmark capture reported 11,415 MiB maximum GPU use and 1,687 MiB free. This is the relevant steady headroom; a 5.6–6.5 GiB load-time free-memory snapshot is not representative.
- P4 V6’s completed two-bank run used two 66,048-cell banks pooled to 132,096 at ubatch 1024. Its telemetry shows minimum free VRAM 715 MiB and maximum 50°C; it passed 14 semantic checks and the 512-token benchmark. Both full-model contexts span all six GPUs. A full TP3 replica does not fit on one three-GPU island, so placing one context per island is not a viable layout.

V12 and P4 V6 both set LLAMA_QSA_INDEXER_IMPL=fused. The V12 scheduler sizes above include the fused route. The startup log traces actual fused calls at KV view 2,048 with vector dispatch and batch 32 with tile dispatch; those are runtime requests, not the full-cache reserve shape.

## Reserve mechanics

llama_context::sched_reserve() chooses n_tokens=min(n_ctx,n_ubatch), creates memory->init_full(), and calls graph_reserve() with that full-memory context. llama_kv_cache::init_full() sets n_kv to the allocated cache size. Normal inference instead computes its KV view from cells.used_max_p1() plus the configured padding, so short requests can have smaller graph shapes while the startup allocator still reserves the full shape.

The cold reserve context has no active ubatch, so ubatch_cells_follow() and blk_ok() are false. The QSA builder enters build_qsa_top_k(), but V12’s LLAMA_QSA_INDEXER_IMPL=fused gate still selects ggml_lightning_indexer inside that function. Thus the V12 reserve does not use the standard four-head score matrix.

The V12 and P4 V6 launch profiles both set LLAMA_QSA_INDEXER_IMPL=fused. The nested fused gate in build_qsa_top_k() is independent of blk_ok(); it emits ggml_lightning_indexer for the current 128-wide, four-head model. The V12 startup log confirms actual fused runtime execution at KV view 2,048 for batch 2 (vector) and batch 32 (tile); it does not directly trace the cold reserve execution at n_kv=132,096 and ubatch 1024. The source shape for that fused node is [n_blocks,n_tps,n_stream], rather than the standard four-head score matrix.

The standard fallback still materializes the four-head MUL_MAT result, then applies ReLU and sums head views. CUDA strided_sum can fuse the ReLU/view-ADD reduction and optional bias into one reduction kernel, but that kernel reads the full MUL_MAT score tensor; it does not remove that tensor. A streamed-head rewrite would target only the standard fallback and is not the immediate V12 memory route.

For fused QSA, the graph output and F16 mask are each sized by n_blocks × n_tps. At ratio 4 and one stream:

| Context / ubatch | F32 fused output | F16 mask | F32 block-bias input when enabled |
|---|---:|---:|---:|
| 132K / 1024 | 129 MiB | 64.5 MiB | 129 MiB |
| 132K / 2048 | 258 MiB | 129 MiB | 258 MiB |
| 132K / 4096 | 516 MiB | 258 MiB | 516 MiB |
| 66K / 1024, per context | 64.5 MiB | 32.25 MiB | 64.5 MiB |
| 66K / 2048, per context | 129 MiB | 64.5 MiB | 129 MiB |
| 66K / 4096, per context | 258 MiB | 129 MiB | 258 MiB |

These are tensor sizes, not total GPU-resident peaks. The bias is a graph input and may reside in CUDA_Host; allocator lifetime, operator scratch and transfers affect actual residency.

The standard fallback is still the source path that materializes the full four-head matrix. With n_blocks=33,024 and four heads, its F32 score tensor is 516 MiB at UB1024, 1,032 MiB at UB2048, and 2,064 MiB at UB4096; per-block bias adds 129/258/516 MiB when the causal block-bias predicate matches. This is not V12’s active indexer route.

GDN output also grows with ubatch. For S_v=128, H=48, K=5 and four sequences, one GDN result tensor has about 96 MiB of token rows at UB1024, 192 MiB at UB2048, and 384 MiB at UB4096. This is one tensor, not the whole scheduler buffer. The persistent recurrent cache remains 2,251.41 MiB.

## UB2048 / UB4096 assessment

There is no measured UB2048 or UB4096 memory capture, so neither is qualified. At V12’s 1,687 MiB steady-state free, UB2048 cannot be declared safe from the startup-buffer projection alone; its QSA and GDN tensors grow, and MLP/HC/MoE workspace also depends on ubatch. UB4096 has even less margin: fused QSA output, mask and bias shapes alone grow by about 968 MiB over UB1024 at full 132K, before other graph growth.

The two-bank V6 layout has only 715 MiB free at UB1024. Raising both banks’ ubatch to 2048 is not a safe candidate without first reducing measured scheduler peak. The contexts span all six GPUs; there is no valid one-context-per-island arrangement because each full model replica needs all six cards. UB4096 also requires raising V12’s n_batch from 2048 to at least 4096.

## Bounded active-workspace candidate

Preserve n_ctx=132096 and allocate the full KV, indexer and recurrent-state backing as today. Add an explicit reserve policy that keys graph workspace to the actual padded n_kv view and effective ubatch:

1. Compute a conservative expansion budget from steady telemetry, separately for each Meta device buffer and CUDA_Host. For example, keeping a 512 MiB guard leaves at most 1,175 MiB of expansion against V12’s 1,687 MiB free and only 203 MiB against V6’s 715 MiB minimum. These are ceilings above the captured high-water mark, not permission to allocate that much without checking each backend’s actual reserve size.
2. Derive a starting ubatch cap from the active KV view. With R=4, H=4, stream count S, and fused QSA, score-output bytes are approximately ceil(n_kv/R) × n_tps × S × 4; the F16 mask is half that. For standard fallback multiply the score term by four heads. Add the GDN token-output term and other graph allocations, then confirm the actual candidate against ggml_backend_sched_get_buffer_size() and per-GPU telemetry. For the two-context V6 case, the known per-GPU expansion budget is only about 203 MiB with the example guard, so UB2048 cannot be selected from shape estimates alone.
3. At startup, reserve for the current active KV view and selected ubatch cap, rather than memory->init_full(). Keep the physical KV cache full-sized.
4. Before a ubatch would exceed the reserved KV-view bucket or workspace budget, drain the scheduler and explicitly re-reserve for the next bucket. Use geometric view buckets such as 8K, 16K, 32K, 64K and 132K to avoid a reserve on each prompt growth step.
5. Choose effective ubatch from the remaining per-GPU workspace budget and active n_kv. Short-context requests may use a larger ubatch; as n_kv grows, reduce ubatch or split prefill into more ubatches. Never change cell positions, empty/foreign/future masks, causal limits or actual max context.

This requires explicit reserve/rebind logic. ggml_gallocr_alloc_graph() only automatically re-reserves a single-buffer allocator; V12 has multiple Meta/CUDA_Host buffers, so a graph shape increase currently needs an explicit reserve. The proposal preserves the full KV capacity, but it has not been implemented or tested.

## Decision

- UB2048 is unqualified for both the one-context V12 and two-context V6 configurations.
- UB4096 is over budget until a measured graph-memory reduction is available.
- The next safe route is budgeted active-KV workspace growth with a dynamic ubatch cap, retaining the full 132K backing cache.

## Source and evidence

- src/src/llama-context.cpp:1121-1168, 4041-4104 — full-memory, maximum-ubatch scheduler reserve.
- src/src/llama-kv-cache.cpp:1254-1304, 2713-2727, 2832-2840 — runtime used-prefix views, full-cache reserve context, and no-ubatch reserve behavior.
- src/src/models/qwen4exp.cpp:2710-2738, 1825-1873, 1942-1984 — QSA branch selection, standard/fused indexer gates and score shapes.
- src/ggml/src/ggml.c:6615-6632; src/src/models/delta-net-base.cpp:563-588 — GDN output sizing and snapshot count.
- src/ggml/src/ggml-alloc.c:1052-1065 — automatic re-reservation only for one-buffer allocators.
- src/ggml/src/ggml-backend-meta.cpp:2338-2360 — Meta buffers are allocated on each child backend; reported size is the maximum child size.
- V12 log: standard-semantic-20261014/results/20261015T060946Z/server.log lines 1955-1959, 2062-2087, 2212-2219, 2296-2298; 8K prompts at lines 44670 and 55688.
- V12 launch: standard-semantic-20261014/results/20261015T060946Z/launch.json sets LLAMA_QSA_INDEXER_IMPL=fused.
- P4 V6: standard-opt-20261014/results/p4-v6-pinned-pipeline-20261015T061650Z/results.json, profile.json and telemetry.jsonl; min free 715 MiB.
- UB512 standard-indexer log: standard-semantic-20261014/results/20261014T150334Z/launch.json and server.log. It is not a scaling comparison for V12 fused.

