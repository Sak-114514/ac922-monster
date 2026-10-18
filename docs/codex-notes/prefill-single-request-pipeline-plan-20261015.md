# Single-request, cross-ubatch 3+3 prefill pipeline plan

Date: 2026-10-15. Scope: read-only source/ownership plan for queueing successive outputless prefill ubatches through the existing six-V100 3+3 tensor-split scheduler. No engine source, build, model, or GPU was touched in this task.

## Decision

Try a two-frame queue on **one existing scheduler, one scheduler-owned graph, and one `llm_graph_result` for the no-output shape**. Reuse its two Meta backend objects/streams, graph allocator, and input tensors across ubatches. Use the already implemented concrete `llama_memory_hybrid_idx_context::freeze_current()` to snapshot per-ubatch metadata while retaining the single live KV/indexer/GDN tensor allocations.

Do not make a scheduler, full graph result, KV cache, or recurrent-state copy per frame. The V15 fixture already shows that a single-copy scheduler can queue the same fixed graph repeatedly and overlap the next island-A chain with the previous island-B chain, with two alternating bridge-copy slots.

## Current executor path and the minimum seam

`llama_context::sched_reserve()` builds one scheduler from `backend_ptrs/backend_buft` and allocates the PP/TG reservation (`src/src/llama-context.cpp:1212-1307`). With tensor split 3+3, the required runtime backend list is MetaA, MetaB, CPU; `pipeline_parallel` must remain false so the scheduler keeps `n_copies == 1` and the FIFO assertion remains valid.

The normal batch path is already asynchronous at the graph-call boundary:

1. `decode_internal()` prepares one `llama_memory_context` with all ubatches (`llama_context.cpp:3480-3521`).
2. For each ubatch it counts outputs, calls `process_ubatch()`, then advances the live context with `mctx->next()` (`:3554-3589`, `:3838-3848`).
3. `process_ubatch()` calls `mctx->apply()`, forms graph parameters, selects `gf_res_prev[n_outputs > 0]`, tests `res->can_reuse()`, sets inputs, enables FIFO snapshots, and invokes `graph_compute()` (`:2856-2907`, `:2910-3069`).
4. `graph_compute()` calls `ggml_backend_sched_graph_compute_async()` and returns without a final synchronize (`:4508-4538`); `decode_internal()`'s unconditional end-of-batch `synchronize()` is commented out (`:3905-3908`).

The two-island graph scheduler already handles A→B dependency copies asynchronously when the mirrored copy callback accepts the input. The V15 change only replaces a host barrier at a **no-input** cross-Meta split. It is not itself the A→B activation-copy mechanism.

The critical current blocker is graph/result reuse. When `res->can_reuse()` fails, `process_ubatch()` synchronizes all tensor-split backends before resetting/rebuilding the scheduler graph (`:2976-3025`). The stream pipeline can only run while the same no-output graph shape is reused. `llm_graph_result::can_reuse()` permits a new memory-context pointer if every input class confirms its required tensor shapes/cursor geometry (`src/src/llama-graph.cpp:1756-1831`). Qwen4Exp's QSA input checks `n_kv`, block-cell, bias, and ubatch shapes (`src/models/qwen4exp.cpp:1182-1216`); the hybrid input checks attention mask/KV dimensions and recurrent `n_rs`, `head`, and `rs_z` (`src/llama-graph.cpp:1212-1259`).

The existing `gf_res_prev` has only two entries, deliberately separated by output state (`src/src/llama-context.h:557-562`, `get_gf_res_prev()` in `llama-context.cpp:4323-4328`). All outputless interior ubatches share `gf_res_prev[0]`; the output-bearing boundary uses `gf_res_prev[1]`. Keep that structure. A new graph result per frame would force scheduler reallocation/rebinding, reuse the same single-copy scratch incorrectly, or require a broader multi-graph scheduler design.

## Memory snapshot contract already present

The live hybrid-indexed context is already the concrete class expected by graph downcasts. Its `freeze_current()` requires a successfully applied, non-empty one-sequence ubatch and rejects multi-sequence pooled-QSA mode (`src/src/llama-memory-hybrid-idx.cpp:1675-1687`). It creates:

- an owned ubatch descriptor (`llama-memory.cpp:92-154`);
- frozen attention and indexer child contexts with one copied slot-info/cursor and a copy of the selected sequence's `llama_kv_cells` metadata (`llama-kv-cache.cpp:2810-2839`);
- recurrent `{n_rs, head, rs_z, size, s_copy[]}` scalar/index metadata, captured once (`llama-memory-recurrent.cpp:1372-1405`);
- a cell-granular QSA plan pointing into the frozen indexer-cell snapshot, with pooled-block fast path disabled (`llama-memory-hybrid-idx.cpp:1695-1727`).

The child contexts retain the original cache owner pointers. `get_k/get_v`, `get_r_l/get_s_l/get_p_l` continue to refer to the existing cache tensors; there is no K/V, indexer tensor, or GDN payload copy. Frozen `apply()` is a no-op. The live context must perform `apply()` once, then be frozen immediately; do not call `prepare()` again or find new slots for a frame.

The minimum insertion point is **inside `process_ubatch()` immediately after its existing successful `mctx->apply()`**. For a preflight-approved candidate, dynamically cast to the concrete hybrid-indexed context, call `freeze_current()`, and use the frozen pointer for `graph_params()`/`can_reuse()`/`set_inputs()`. This avoids a second apply. Store the returned owner in the pipeline state before returning from `process_ubatch()`.

On reuse, `llm_graph_input_mem_hybrid::can_reuse()` and QSA's `can_reuse()` rebind their C++ input wrappers to the new frozen context; the subsequent `res->set_inputs()` materializes host input arrays for that ubatch. The scheduler FIFO snapshot path then synchronously copies each host input array into its two-slot pinned buffer before enqueueing H2D (`ggml-backend.cpp:1824-1857`). The graph's device input copies and the next same-island graph are ordered on the same Meta member streams.

`llm_graph_result::params.mctx` is a non-owning raw pointer. In standard reuse, `llm_graph_result::can_reuse()` rebinds input-wrapper pointers but does not update `params` as the MTP-unrolled path does (`llama-context.cpp:2734-2737`). The first frozen context used to build a cached graph therefore needs an owner retained until that graph result is reset. A narrow safe ownership rule is: keep one `graph_owner` tied to the active result, plus a two-slot ring of current frame owners; on graph reuse update the active result's owner/signature to the current frame only after `can_reuse()` succeeds, and release the prior slot only after its B-completion event. Do not leave `params.mctx` dangling or discard an in-flight frame owner.

## Queue order and shared state

For each ubatch `i`, one scheduler call queues its normal graph splits in layer order:

```text
A_i computes its assigned early-layer work and shared KV/GDN writes
  -> mirrored async activation copy A_i -> B_i
  -> B_i computes its assigned later-layer work and shared KV/GDN writes
next call queues A_(i+1) behind A_i and its source copy on A's same streams
  -> B_(i+1) queues behind B_i on B's same streams
```

This permits A_(i+1) to overlap B_i because each island's recurrent/KV state is ordered by its own streams and the islands own distinct layer state. The activation copy reads A_i's canonical source on A's stream before A_(i+1) reuses that scratch. The existing CUDA peer-fanout path records `source_ready`, makes B streams wait before writing the B input, and on two-slot reuse waits for `dst_done` before overwriting the staging slot (`ggml-cuda.cu:3789-3841`). If the mirrored async-copy callback refuses the actual tensor, the scheduler synchronizes source/destination and does a synchronous copy; correctness is retained, but that run is pipeline-ineligible.

Use **one** scheduler and the same two Meta backend objects for all frames. The prior fixture's separate Meta contexts per frame cannot provide this stream-order proof. Preserve the graph's causal KV dependency, and do not let a new request, sequence reset, memory update, cancellation, rollback, or failure mutate shared memory while prior frames are queued.

## Bounded eligibility

Start with a narrow opt-in for ordinary Qwen4Exp prompt processing. Enable only when all conditions are true:

- default target graph (not MTP, unrolled MTP, DFlash, embedding, or encoder path);
- exactly one active sequence and one KV stream; `freeze_current()` must succeed;
- full fixed-size prompt ubatch (`n_tokens == cparams.n_ubatch`, preferably 1536), more than 64 tokens, `n_outputs == 0`, no embeddings, no sampling/output readback, no eval callback;
- two Meta3 backends plus CPU, tensor split, `parallel=false`, `pipeline_parallel=false`, no MoE expert cache;
- `GGML_META_FIFO_PIPELINE=1`, `GGML_META_GPU_EVENT_WAIT=1`, `GGML_SCHED_META_SINGLE_COPY_EVENTS=1`, `GGML_META_ASYNC_COPY=1`, `GGML_CUDA_META_COPY_GPU_WAIT=1`, peer-copy topology enabled, and `GGML_META_SHARED_DEVICE_COPY=0` before the relevant backend initializers;
- graph result reuse succeeds, the graph has reached exact Meta replay-ready state, and the target A→B activation copy is logged as async mirrored copy.

For a server context configured with `n_seq_max > 1`, `LLAMA_TARGET_FIFO_MULTI=1` and the existing target-row eligibility check are required for host-input FIFO; one active sequence must remain the only unique sequence. The first batch at position 0 should be serialized. Any frame with `n_outputs > 0` is a drain boundary. The existing multi-sequence FIFO path also drains on a start position, short ubatch, or output (`llama-context.cpp:2797-2807`).

The 8192 prefill KV pad keeps `n_kv` and QSA tensor dimensions constant within an 8K cache-view bucket (`llama-kv-cache.cpp:1254-1304`, `llama-kv-cache-decode-pad.h:56-63`). With UB1536, that permits several same-shape chunks, then a graph-shape boundary roughly every five to six chunks; drain before the bucket-changing ubatch, rebuild/warm the new shape serially, and restart only after replay-ready. The exact live gate is `res->can_reuse()`; if QSA pooling changes, `n_kv` changes, recurrent `head/rs_z/n_rs` changes, or any input shape differs, fall back to the current synchronize/rebuild path.

CPU PLE work is a separate gate. `GGML_SCHED_CPU_GATHER_PIPELINE` only skips the prior-device barrier for a CPU split that the scheduler proves is an independent host-weight `GET_ROWS` with CPU input IDs. If the actual PLE split is not classified independent, the no-input Meta event does not remove a Meta→CPU host wait. Keep the existing PLE prefetcher, verify the actual CPU-split marker, and retain its request/buffer ownership until the matching frame completes.

## Object count and capacity ledger

| Resource | Pipeline allocation | Bound / evidence |
|---|---:|---|
| scheduler | 1 | Existing context scheduler; `n_copies=1`, unchanged graph arena |
| no-output graph result | 1 | Existing `gf_res_prev[0]`, reused for every interior frame |
| output graph result | 1 existing | `gf_res_prev[1]`; use only after draining |
| per-frame memory metadata | at most 2 in-flight | Frozen cell/slot/ubatch/recurrent snapshots; underlying KV/indexer/GDN tensors shared |
| host input snapshots | 2 slots per `(input tensor, Meta backend)` | Existing FIFO allocates two pinned buffers per source; pinned host RAM, not GPU VRAM |
| island-copy staging | 2 bridge slots | Existing peer path caps each slot at 64 MiB and checks at least `nbytes + 512 MiB` free on its leader when allocating; at 64 MiB, worst case is 128 MiB on the leader GPU |
| scheduler events | 2 Meta events | Small event metadata; each wraps six CUDA events total |

At 132,096 cells, one `llama_kv_cells` snapshot has about 6.6 MiB of raw arrays (`pos`, 2D ext, shift, sequence bitset) plus `std::set` node/index overhead. Each frame snapshots attention and indexer cell metadata, so two frames are roughly 26 MiB raw host arrays plus set overhead, not a duplicate 132K KV tensor. Each frame also copies the ubatch/slot vectors (tens of KiB at 1536 tokens) and a small recurrent `s_copy` vector. These are estimates from the actual types; allocator/tree overhead is additional.

The pipeline must not double the 132K KV cache or recurrent tensors. It adds no second compute-workspace arena if the same scheduler/result/graph is reused. The only bounded VRAM scratch beyond the existing scheduler allocation is the already-existing two-slot peer bridge (up to 128 MiB on its chosen leader); FIFO input staging is pinned host memory. If graph reuse fails, do not solve that by making two complete contexts or two graph arenas.

## First hook and acceptance sequence

1. Record actual `ubatch.n_tokens`, `n_outputs`, `n_kv`, QSA bucket, recurrent `n_rs/head/rs_z`, graph/result pointer, replay-ready result, active sequence ID, and copy sizes. The pipeline flag must log selected, rejected, or drain reasons.
2. Apply the live context once; freeze it. If freeze returns null, fail closed. In particular, `freeze_current()` of recurrent state captures `s_copy[]` by calling the ordinary one-shot getter that consumes `rs_idx`; do not try the live path again after a failed/partial capture.
3. Build the first frame normally. It may warm/capture the graph, but it has no frame to overlap. Start pairing only after the exact root and split graphs are replay-ready.
4. On each eligible outputless frame, require graph reuse, set its per-frame host inputs, call the existing asynchronous scheduler, record a completion event on MetaB, and retain the frozen owner in a two-slot ring. Submit the next frame on the same scheduler/graph. On slot reuse, wait only for the corresponding B completion and FIFO host-slot event.
5. Drain both islands before the output-bearing ubatch, cache bucket change/rebuild, start of a different sequence, `memory_update`, `seq_rm`, cancellation, sampler/logits consumption, PLE request-owner release, exception, or any fallback. A rejected shape must run through the existing serial/rebuild path.

First runtime test should use real text and a long request with at least two consecutive `n_outputs == 0` full ubatches plus one output boundary, within one 8192 KV bucket. Keep UB1536; batch3072 only provides two ubatches, so it is useful only if both are outputless. If the second ubatch produces logits, there is only one outputless frame and no two-frame overlap; use a larger batch that contains two interior ubatches before the output boundary (for example 4608/6144) if request semantics allow. `n_ubatch` remains 1536, so this increases the number of queued chunks without increasing per-graph scratch. Compare semantic canaries and output/state against the serial standard-indexer path before timing.

## Benefit ceiling and stop conditions

The existing V15 PP point supplied by the parent is 859 tok/s. Perfect two-stage overlap is bounded by `(t_A + t_B)/max(t_A, t_B) <= 2x`, so it cannot by itself reach 6000 tok/s; a two-ubatch call has only one overlap window and is at most 1.33x for balanced stages. This step tests whether GPU idle time between islands is recoverable without copying KV/GDN state; MoE/prefill kernels, CPU PLE, graph rebuilds, and host submission remain separate bottlenecks.

Keep the path opt-in and off on any semantic mismatch, failed `freeze_current`, changed recurrent/QSA shape, non-mirrored/unsupported activation copy, CPU-gather host barrier, graph rebuild, unexpected output count, or frame/slot-order error. Do not claim full-model benefit from the V15 synthetic proof; it establishes scheduler and bridge ordering only.
