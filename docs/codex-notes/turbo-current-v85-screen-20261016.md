# V85 Hierarchical EP6 下 TurboPrefill / frame-pipeline 屏蔽原因

日期：2026-10-16  
范围：只读审计 V85 当前模型/调度拓扑、V36 exact profile/runtime、V38/V42 历史验证和当前源码。未 build、未启动 GPU workload、未改 engine。

## 结论先行

V85 当前把 TurboPrefill 与 frame pipeline 设为 0 是 V36 组合 profile 的明确选择：V36 launcher 要求 ROOT_TURBOPREFILL=0，并把它导出成 LLAMA_TURBOPREFILL=0；配置文件同时显式设 LLAMA_PREFILL_FRAME_PIPELINE=0。V36 因而没有尝试 Turbo admission，也没有运行时拒绝原因。

此外，即使只把环境变量改成 1，当前 Hierarchical EP6 拓扑也无法通过旧 Turbo A/B pipeline 的结构门。层级 EP6 将两个 TP3 island 包入一个 outer Meta；target scheduler 是 outer Meta 加 CPU。旧 frame/Turbo 代码要求两个顶层 Meta3 加 CPU，也要求顶层 CPU→MetaA→MetaB 三段图。V85 不存在这两个独立顶层阶段。V42 的旧 3+3 layer-island 成功记录不能作为当前 Hierarchical EP6 适配证据。

## 当前 V85 现场

- Run：root-p4-v36-v85-onebank4-ep6-d3-16k-20261016t091525z
- Runtime：runtime-perf-v85，build ID 20261016T085939.361444Z
- Exact profile：fixtures/dual-context-v36-combo-20261016/v36-combo-exact-env.json
- Profile SHA-256：d703e4aca4775abdcb95bc0a7927010ea6860c3af1f87baa3f8efe42744d5221
- V36 launcher SHA-256：806e1e6303dbce27a6c3d19e64690ef7160d92062565d79006ed267811ce3e90
- Full remote runtime log SHA-256：67a452f4c93b2f53c4fa361213cde2732a578f8a7f3943809df259c7944452be
- Current private-source hashes matched Mac/server at audit time:
  - src/src/llama-context.cpp: a402784401300f4e333205e0ab592804e8073f1d2d718017f04391d2a1e05f57
  - src/src/llama-memory-hybrid-idx.cpp: 4056b18ef417ab1a70f30754355d8c414d45a868678b79d2803fd08686e0ff20
  - src/ggml/src/ggml-backend.cpp: 7aab7ecc95489eb53e0e3de92c964cd84707972d9c8f5d94bd4c37dcae47adb0
  - src/src/llama-target-fifo-policy.h: 4be347f04c1028308d36f4a47166a38c1a06de8977a068809101b97f18bf05d0

The V36 exact environment sets LLAMA_TURBOPREFILL=0, LLAMA_PREFILL_FRAME_PIPELINE=0 and ROOT_TURBOPREFILL=0. The V36 launcher hard-rejects ROOT_TURBOPREFILL other than 0 with the message “V36 keeps the established TurboPrefill-off profile.” The runner maps ROOT_TURBOPREFILL to LLAMA_TURBOPREFILL. Thus no Turbo/frame candidate was submitted in this run.

The current one-lane 8K probe still used a four-lane target context: n_seq_max=4, four independent 16K KV streams, non-unified KV, target UB=1024, eight target UBs for 8K, MTP D3 / UB512. The runner records one sequence per logical prompt call and outputs only the final prompt token. It reports target prefill 9.932 s + MTP prefill 1.338 s = 11.271 s wall. This is an uncached prompt; the metric is not from a cache hit.

Other prerequisites are present in the V36 profile: GGML_META_FIFO_PIPELINE=1, GGML_META_ASYNC_COPY=1, GGML_META_GPU_EVENT_WAIT=1, GGML_CUDA_META_COPY_GPU_WAIT=1, GGML_SCHED_CPU_GATHER_PIPELINE=1, GGML_META_REPLAY_GUARD=1, LLAMA_TARGET_FIFO_MULTI=1, GGML_SCHED_META_SINGLE_COPY_EVENTS=1, and a 512 MiB Turbo pinned-cap setting. GGML_SCHED_SYNC_INPUTS and GGML_META_SHARED_DEVICE_COPY are unset. V36 runtime logs confirm nested Meta FIFO activity. Since Turbo was disabled, these values do not establish that all Turbo runtime gates pass.

## Topology: the decisive blocker

Hierarchical EP6 builds two inner Meta3 islands, then wraps them in one outer Meta2xMeta3 device. It places only that outer device in model.devices. The V36 log shows:

- Lines 99–107: two island Meta devices of three GPUs each, then “hierarchical-ep6 enabled=1 topology=Meta2xMeta3 physical=6 dense=TP3-mirrored experts=EP6.”
- Line 108: llama_prepare_model_devices uses one device named Meta().
- Full raw runtime.log lines 2004, 2097, 2120 and later repeats: GGML_SCHED_META_SINGLE_COPY_EVENTS is disabled with “requires exactly two Meta backends plus CPU.”
- Full raw runtime.log lines 70617, 70623, 70635 show the target outer Meta context has two members and its nested children each have three members.

The source confirms the topology:

- src/src/llama.cpp:348–370: in hierarchical mode, create the two 3-GPU island devices, wrap them in one outer Meta2, then push only that outer device into model.devices.
- src/src/llama.cpp:162–166: without hierarchical EP6, LLAMA_ISLAND_SPLIT puts each island in model.devices and divides model layers across the islands. That is the older layer-pipeline layout used by the A/B Turbo stage design.
- src/src/llama-context.cpp:916–972: the context creates one backend per model.devices entry and then adds CPU. In this V36 profile no separate PLE or draft de-TP backend is enabled, so the target scheduler gets the one outer Meta plus CPU (two top-level backends), not CPU + MetaA3 + MetaB3 (three top-level backends).
- src/ggml/src/ggml-backend.cpp:1164–1175: single-copy Meta events require exactly three scheduler backends and exactly two 3-member Meta backends. The runtime warning is the observed rejection.
- src/src/llama-context.cpp:105–115: frame topology gate requires model.devices.size()==2 and each top-level Meta to contain three members. Hierarchical EP6 has one top-level Meta containing two Meta3 children, so this gate rejects it.
- src/src/llama-context.cpp:2995–3022: frame scheduler gate requires exactly three top-level backends and one copy.
- src/ggml/src/ggml-backend.cpp:2162–2198: Turbo stage classifier requires three splits/backends ordered CPU, MetaA, MetaB, with a mirrored MetaA→MetaB activation edge.

Therefore changing only the environment variable will not unlock the current topology. The frame gate and Turbo stage classifier have incompatible top-level backend assumptions.

## Why the old A/B pipeline is not a direct fit

The old two-stage Turbo route treats MetaA3 and MetaB3 as sequential layer ranges. For each UBatch it enqueues CPU prefix + MetaA, stages A’s mirrored activation through pinned host memory, then seals MetaB behind an A completion event. A bounded two-slot ring lets the next A enqueue while a prior B remains pending. It can overlap those independent A/B stage executions when the scheduler exposes two top-level islands.

Hierarchical EP6 is different. Its one outer Meta graph contains both inner islands in the same layer computation: dense weights are mirrored TP3 across the two islands while expert weights are partitioned EP6. The two inner Meta3 children are participant shards, not independent sequential A/B layer stages. Treating them as A and B would change the model’s placement/communication semantics. The V42 stage map with separate CPU, Meta(CUDA0–2), Meta(CUDA3–5) belongs to the older 3+3 layer-split topology.

A hierarchical-aware frame path could be a separate design: queue whole nested-EP6 frame graphs while preserving each frame’s recurrent/KV snapshot and ordered GPU tails. That would not be the existing A/B Turbo stage adapter, and no current full-model proof shows it safe or faster. A real nested-EP6 prefill pipeline would need an explicit graph/scheduler seam; loosening the existing topology if-condition would be incorrect.

## Requested gate checks

| Gate | V85 evidence | Audit result |
|---|---|---|
| Turbo/frame environment | Both LLAMA_TURBOPREFILL and LLAMA_PREFILL_FRAME_PIPELINE are 0, and V36 script rejects ROOT_TURBOPREFILL=1 | This is the direct reason the run never attempts the path. |
| n_seq_max=4 | Target prompt batches are one sequence at a time; LLAMA_TARGET_FIFO_MULTI=1. The FIFO row policy accepts n_seqs=1 when max_seqs=4. freeze_current checks the current applied sequence count, not n_seq_max==1. | Not a standalone blocker. It does not rescue the incompatible hierarchical scheduler topology. |
| embeddings_nextn | Runtime says nextn_hidden=required; output_reserve logs actual buft CUDA_Host. The frame gate allows embeddings_nextn only with CUDA_Host output. Final output-bearing UB is excluded by n_outputs==0 and should drain to ordinary compute. | Appears satisfiable for the tested context; no Turbo attempt means not runtime-confirmed. |
| Backend samplers | The V36 C++ runner sets speculative draft backend_sampling=false and has no llama_context::set_sampler/llama_set_sampler call. Turbo requires sampling.samplers.empty(). | Empty for this direct runner by source inspection. Server registration order is separate and must be checked if using the server path. |
| Scheduler shape | Source constructs one top-level outer Meta plus CPU; runtime logs Meta() with two nested members and the single-copy event warning. | Hard blocker: V85 has two scheduler backends, not three CPU/MetaA/MetaB stages. |
| Pooled QSA cache | n_seq_max=4, kv_unified=0, LLAMA_QSA_POOLED_MULTI=0. llama_memory_hybrid_idx allocates mem_blk only when n_seq_max==1 or the unified multi-pool is enabled. | Current get_blk()==nullptr frame check should pass. Changing to n_seq_max=1 would create the single-sequence pooled cache; current frame gate rejects it because frozen frames do not maintain those pooled rows. |
| Full UBatch / output shape | The clean 8K probe reports eight target UBs of 1024. A one-sequence 1024-token chunk is over the >64-token threshold; the final prompt-output UB is intentionally ineligible. | The intermediate prompt UBs are shape candidates. Actual per-UB n_outputs/sequence layout must be logged in a new run. |
| Meta graph readiness | Turbo requires the graph allocated and Meta replay-ready. V38 toy fixture logged initial warmup declines before selecting; V42 had 14 selected 512-token frames. | Warmup fallback is expected in the old topology. Not evaluated for V85 hierarchical EP6, and cannot solve its top-level stage mismatch. |
| Pinned-memory budget | V85 cap is 512 MiB; V42 older 512-token path logged about 156.8 MB pinned high-water. V85 uses UB1024. | Unknown for current UB1024; do not extrapolate. A correct future profile should report make-plan/FIFO combined bytes and fail closed if over cap. |

Relevant source conditions:

- Frame policy: src/src/llama-prefill-frame-policy.h:77–126.
- Frame environment, topology, batch and scheduler gates: src/src/llama-context.cpp:96–137, 2914–3044.
- Frozen memory owner and graph admission: src/src/llama-context.cpp:3112–3125, 3238–3244, 4201–4251.
- QSA pooled cache allocation and frame freeze rules: src/src/llama-memory-hybrid-idx.cpp:121–163, 1722–1775.
- Turbo graph classification, staging plan and pinned budget: src/ggml/src/ggml-backend.cpp:2150–2198, 2777–2909.
- Turbo split submit and A/B event sequence: src/ggml/src/ggml-backend.cpp:3187–3314, 3372–3435.
- Per-context model vs memory ownership: src/src/llama-context.h:416–425.

## Prior validation, kept in its actual scope

- V38 fixture, fixtures/turbo-backend-20261016/results/turbo-backend-v38.log: synthetic 18-node F32 graph, CPU GET_ROWS, MetaA3 and MetaB3. It exercises eight exact frames, two-slot reuse, canary, deferred-output seal, state exactness, and event-based owner lifetime. It is not the Hierarchical EP6 model.
- V42 full model, notes/V42-fullmodel-summary.json: an older top-level CPU + two Meta3 stage topology. The fenced and Turbo 8K responses had identical token IDs; 14 Turbo frames were selected. V42-CURRENT-STATE reports PP 1396.43, while its fenced reference was cold 653.89, so that ratio is not a valid clean speed comparison. It proves that the old topology route ran, not that current V85 Hierarchical EP6 can use it.

## Next safe decision

Do not launch a V85 full-model run with only ROOT_TURBOPREFILL=1: the V36 controller currently refuses that value, and even a new controller would hit the structural top-level backend mismatch. The old A/B Turbo route would require reverting to the old per-island layer-split layout, which changes the current hierarchical dense/EP6 placement.

If the target is specifically faster prefill while retaining V85 Hierarchical EP6, the next code design should be a separate nested-Meta full-graph frame queue, not an A/B split. It must keep the current memory owner and recurrent-state snapshots, use the outer Meta's nested event/tail ordering, and prove exact outputs before measuring speed. A possible no-code baseline reuse is the current V85 single-8K direct probe; a correct comparison must use the same prompt, no prompt/KV cache, and preserve its output/logit gate. Do not report the historical V42 rate as a current-topology gain.
