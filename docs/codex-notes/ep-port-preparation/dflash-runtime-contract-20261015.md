# DFlash first GPU integration: static runtime contract audit (2026-10-15)

Scope: read-only inspection of the active private tree `/home/debian/ac922-monster/experiments/standard-opt-20261014/src`, the V25 draft-only loader log, and PixelML's adapter source. No source/build/model/GPU/system state was modified. This note distinguishes source-level compatibility from an unrun target+draft graph.

## Short verdict

The source contracts for anchor-first block size 7, mask ID, `p_min=0`, native HC features, and target weight borrowing are internally aligned. The V25 loader result proves that the standalone F32 draft GGUF's 58 tensors load through a 3-device Meta model; it does **not** prove the actual speculative context can run against the live six-device target, because that test did not load the target or exercise `ctx_other` borrowed weights.

Two prerequisites must be explicit in the first combined launch:

1. Use `--spec-draft-device CUDA3,CUDA4,CUDA5` (or the exact equivalent device names in the current process). `LLAMA_DFLASH_ISLAND_SPLIT=3` groups the draft's already-selected three devices; it does not select the cards. With no draft-specific device list, `common_base_params_to_speculative()` inherits the target's six devices and the `3` split sum will not match.
2. Keep target `LLAMA_ISLAND_SPLIT=3,3` and draft `LLAMA_DFLASH_ISLAND_SPLIT=3` together. The arch-gated environment selection in `src/llama.cpp:249-261` lets the Qwen4Exp target keep 3+3 while DFlash's 3 selected devices form one Meta3 island.
3. In the combined process, use the draft-specific device option instead of process-wide `CUDA_VISIBLE_DEVICES=3,4,5`; the latter also hides the first target island.

The remaining first-run risk is one combined **graph** check: verify the draft scheduler can consume the target's borrowed CPU embedding and second-island LM head, and that its zero-width attention shard is handled through a complete draft forward. Static source indicates both layouts are supported, but only a combined context execution proves them on this branch.

## Verified model metadata and decoder indexing

`fixtures/dflash-loader-20261015/gpu-meta3-loader.log` reports:

- `dflash.block_count=5`, `dflash.embedding_length=2560`, `dflash.attention.head_count=24`, and `head_count_kv=2`;
- `dflash.block_size=7`, `dflash.target_layers=[4,16,24,36,44]`;
- `dflash.sample_from_anchor=true`, `dflash.has_confidence_head=false`;
- `tokenizer.ggml.mask_token_id=248077`;
- F32 weights, 6/6 layers offloaded, Meta buffer 760.13 MiB.

The converter's `DSparkModel` sets `_sample_from_anchor` and writes `dflash.sample_from_anchor` (`conversion/qwen.py:806-836`). For the PixelML schema, `target_layer_ids=[3,15,23,35,43]` becomes GGUF boundary IDs `[4,16,24,36,44]`, and the mask ID comes from `dflash_config`. The author adapter defines query zero as the first prediction for the anchor-first checkpoint: [PixelML exporter](https://raw.githubusercontent.com/PixelML/deepspec-qwen38-flash-next/main/scripts/data/export_features_vllm.py), [adapter overlay](https://raw.githubusercontent.com/PixelML/deepspec-qwen38-flash-next/main/serving/adapter/adapter-overlay.patch), [anchor-layout patch](https://raw.githubusercontent.com/PixelML/deepspec-qwen38-flash-next/main/serving/patches/02-dflash-speculator-anchor-layout.diff).

The private speculative implementation matches this layout when selected as **`draft-dspark`**:

- `common/speculative.cpp:951-1008`: reads `dflash.block_size`/`dflash.sample_from_anchor`, derives `n_draft_max=7`, and checks confidence gating. Because the model declares no confidence head, `--spec-draft-p-min 0` is required; any positive threshold throws at lines 991-998.
- `common/speculative.cpp:1211-1216`: with DSpark + anchor-first it constructs exactly 7 query rows: `[anchor, mask, mask, mask, mask, mask, mask]`; output flags are enabled for all rows for non-DFlash2.
- `common/speculative.cpp:1273-1299`: begins sampling at row 0 and returns rows 0..6. This is the expected 7 next-token proposals from the seven trained block positions; it does not return the anchor itself.
- `common/speculative.cpp:3349-3351` auto-detects only by presence of `markov_w1.weight`; because this checkpoint has no Markov head, it becomes DFlash without an explicit type. The fallback type must not be `draft-dflash`: the implementation only uses row zero for the anchor-first DSpark type. Choose `--spec-type draft-dspark` explicitly.

This agrees with the adapter's `query_zero_predicts_next=true` and the logged GGUF `sample_from_anchor=true`. No off-by-one or extra mask row is visible in the current code. A semantic target-only vs target+draft equality check is still required after the graph works.

## Target feature taps

The target metadata boundary IDs are already shifted by the converter. The speculative constructor calls `llama_set_embeddings_layer_inp()` on those IDs (`common/speculative.cpp:1052-1059`) and reads each F32 row at `n_embd_tgt=2560` (`:1150-1161`).

When `LLAMA_DFLASH_NATIVE_HC_TAPS=1`, the Qwen4Exp graph captures `cur` after `build_hc_mix()` at the beginning of each selected `il`, before that block's attention (`src/models/qwen4exp.cpp:536-556`). That is the contracted 2560-wide block input, matching PixelML's vLLM `GatedResidual.mix()[1]` / `combine_and_mix()[1]`. The `t_layer_inp` wide `res_hc` is replaced for selected layers; extraction asserts F32 and row width 2560 (`src/llama-context.cpp:4412-4441`). The capture indices `[4,16,24,36,44]` therefore map to the reference tap IDs `[3,15,23,35,43]` exactly. No slicing from the 10240-wide HC stream is used.

## Q-head/KV-head distribution across three draft cards

The loader has a dedicated GQA split granularity in `src/llama-model.cpp:504-530,555-575`:

- `n_gqa = 24 / 2 = 12` query heads per KV head;
- `n_embd_q = n_gqa * 256 = 3072`;
- `granularity_kv = granularity_q / n_gqa = 256`;
- with three equal device weights, 6144 query-output elements split `[0,3072,3072]`, and 512 key/value-output elements split `[0,256,256]`.

In head units, that is query heads `[0,12,12]` and KV heads `[0,1,1]` before rotation. `llama_meta_device_get_split_state()` rotates by effective layer index (`src/llama-model.cpp:197-232`), so the five draft layers rotate the empty shard across CUDA3, CUDA4, and CUDA5 (then CUDA3, CUDA4). There is no KV replication. `ggml-backend-meta.cpp:1891-1930` materializes local tensors and `:2068-2084` clears the compute flag when a source slice is empty, so zero-size members are explicitly represented. The full DFlash attention graph on this shape remains unverified; log each layer's rotated layout.

The F32 feed-forward split is separate and uses the usual tensor axis across all three cards. The load log's `offloaded 6/6 layers` does not itself certify Q/KV graph execution.

## Borrowed target embeddings/head and backend buffer compatibility

The PixelML draft omits its own token embedding and output tensor. `src/models/dflash.cpp:75,129-134` marks them optional; decoder graph branches use `ctx_other` if either pointer is null (`:592-600`, `:712-720`). `common_speculative_init_result` sets `cparams.ctx_other=ctx_tgt` (`common/speculative.cpp:3581-3588`), and `llama-context.cpp:729-735` retains it for a DFlash model with missing weights.

Static placement review:

- Model loading deliberately keeps the input embedding on CPU: `llama-model.cpp:1508` sets `dev_input={cpu_dev,...}`. The draft context includes a CPU backend (`llama-context.cpp:963`), so the shared token embedding is reachable through the ordinary CPU-to-draft scheduling/copy path.
- The target output layer is assigned by `get_layer_buft_list(n_layer_all)` (`llama-model.cpp:1516-1518`), i.e. the final target island. With target 3+3 and draft devices exactly CUDA3/4/5 in the same order, both use the same three underlying device handles.
- The meta backend's `supports_buft` checks that a Meta device and Meta buffer have the same ordered underlying device list (`ggml-backend-meta.cpp:244-263`). This supports a target output buffer owned by the target's second Meta island being consumed by the draft's separate Meta3 scheduler, even though the Meta wrapper objects are distinct.
- The Meta scheduler check is on the ordered physical device list, not CUDA ordinal text. When the whole process sees six GPUs, specify draft devices 3,4,5; in a standalone CVD-restricted loader test they may be renumbered 0,1,2.

This is a compatible source-level contract **only if** the draft device order exactly matches the target output island. If the draft card list is reordered, broadened to all six, or reduced via a process-wide `CUDA_VISIBLE_DEVICES` that also hides target cards, the buffer/device check can fail or target topology can change. The V25 draft-only loader log proves DFlash tensor placement, not borrowed target-head use; no current log shows the combined graph.

## Minimal first combined run evidence to capture

Before interpreting performance, require startup logs for (1) target islands `CUDA0,1,2` / `CUDA3,4,5`, (2) `DFlash draft topology: env=LLAMA_DFLASH_ISLAND_SPLIT groups=3 physical_devices=3 target_topology_unchanged=1`, (3) draft device list and ordered physical members, (4) GGUF `sample_from_anchor=true`, `block_size=7`, mask 248077 and no confidence head, (5) native tap logs exactly at `[4,16,24,36,44]`, (6) borrowed `ctx_other` target embedding/head accepted by scheduler, and (7) no empty-shard compute for GPU3 while GPUs4/5 each own one KV head.

Then one bounded greedy request must produce the same tokens as target-only. This is the first evidence that the target/draft graph, borrowed head, tap rows, and anchor row all work together. V25's loader pass is not that test.

## Source pointers

- Private DFlash consumer: `common/speculative.cpp:951-1059,1129-1187,1199-1328`.
- Private model loader/decoder: `src/models/dflash.cpp:7-50,129-155,175-205,486-535,592-723`.
- Private Qwen4Exp HC tap: `src/models/qwen4exp.cpp:536-556`; row policy/extraction: `src/llama-dflash-tap-policy.h:8-28`, `src/llama-context.cpp:4230-4232,4412-4441`.
- Draft device routing and island env: `common/speculative.cpp:3581-3605`, `common/speculative.cpp:3497-3526`, `src/llama.cpp:249-261`, `src/llama-model.cpp:1510-1518`.
- Meta zero shards and borrowed buffer device compatibility: `ggml/src/ggml-backend-meta.cpp:244-263,1891-1930,2068-2084`.
- V25 log: `fixtures/dflash-loader-20261015/gpu-meta3-loader.log`.
