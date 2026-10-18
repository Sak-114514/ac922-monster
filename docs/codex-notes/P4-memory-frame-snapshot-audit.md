# P4 memory-frame snapshot adapter: source audit and minimum seam

Date: 2026-10-15. Scope: read-only audit of memory/KV/GDN/QSA/graph-input contracts for two private outputless single-sequence prefill frames. No engine code, build, GPU, model, or system state was touched.

## Finding

A snapshot adapter cannot be wired in new files alone. `llm_graph_params::mctx` is a `llama_memory_context_i *`, but the generic graph builder downcasts it to `llama_memory_hybrid_context` in `llama-graph.cpp:3796-3804`. Qwen4Exp then downcasts to `llama_memory_hybrid_idx_context` in `models/qwen4exp.cpp:501-509`, and its QSA graph inputs repeatedly downcast the same context at reuse/build time. A wrapper that is not an actual subclass is unsafe; a subclass still cannot replace the state accessors used through base pointers.

The concrete context internals that need independent frame state are private:

- `llama_memory_hybrid_context` owns `ubatches`, `ctx_attn`, `ctx_recr`, and its cursor (`llama-memory-hybrid.h:130-139`).
- `llama_memory_hybrid_idx_context` owns its `mem`, `ctx_idx`, stream counts, and mirrored cursor (`llama-memory-hybrid-idx.h:274-285`).
- `llama_kv_cache_context` owns the `llama_kv_cache *`, `sinfos`, `ubatches`, `i_cur`, and mutable `n_kv` (`llama-kv-cache.h:446-478`). There is a public constructor that accepts explicit `sinfos` and `ubatches`, but no read/export method for the active slot info or the underlying cache pointer from an existing context.
- `llama_memory_recurrent_context::get_head/get_rs_z/s_copy` read live recurrent memory. `s_copy` consumes `rs_idx[seq]` by resetting it (`llama-memory-recurrent.cpp:1312-1353`), so concurrent reads through existing contexts are not immutable snapshots.

The current Qwen4Exp graph input path also depends on live QSA state: `llm_graph_input_qsa::set_input` updates per-sequence pooled bookkeeping and calls `set_input_qsa` on the owner context. A frame copy must own its selector arrays and cursor values while sharing only the underlying K/V/indexer/GDN storage.

## Narrow core seam needed

Keep existing graph context classes as the dispatch type, but add a capture/factory seam in the memory layer, then let the existing graph owner build from the captured concrete frame context:

Minimum memory-side files and suggested APIs:

1. `src/llama-kv-cache.h/.cpp`: `std::unique_ptr<llama_kv_cache_context> freeze_current() const`. Inside the class, copy its current private `kv`, `sinfo[i_cur]`, `ubatches[i_cur]`, and `n_kv` into a one-frame child context. Mark it committed/frozen so `apply()` is a no-op and all getters/input setters use the copied cursor/slot.
2. `src/llama-memory-recurrent.h/.cpp`: add `llama_recurrent_frame_state { head, rs_z, n_rs, size, s_copy[] }` plus `std::unique_ptr<llama_memory_recurrent_context> freeze_current() const`. Capture `s_copy[]` once, serially, while the owner state is stable; getters use the frozen copy and `get_r_l/s_l/p_l` return the exact same underlying tensor pointers.
3. `src/llama-memory-hybrid.h/.cpp`: add a constructor accepting one owned ubatch plus frozen attn/recurrent child contexts, and a `freeze_current()` wrapper that preserves the existing concrete `llama_memory_hybrid_context` type.
4. `src/llama-memory-hybrid-idx.h/.cpp`: add `std::unique_ptr<llama_memory_hybrid_idx_context> freeze_current() const`; copy its private memory owner and current QSA/index slot state, attach the frozen index child context and frame-owned QSA selector input plan, and preserve the concrete indexed-context type. For this single-sequence experiment, any pooled QSA registration/commit must be materialized during the serialized freeze; no frame graph-input call may mutate shared tables.

The integration caller (owned by the context/graph lane) should call live `apply()` once, then `freeze_current()`, and pass that resulting concrete indexed context to `graph_params`. Do not create a child by calling `prepare()` again or searching for new cells. The coordinator commits frames in sequence order only after the owning graph succeeds.

The frame factory must store each frame's `n_kv` view explicitly. Current `apply()` both commits metadata and refreshes `n_kv` (`llama-kv-cache.cpp:2740-2753`); slot capture and metadata commit therefore remain separate operations.

The recurrent frame state must contain `{head, rs_z, n_rs, size, s_copy[]}` plus shared per-layer `r_l/s_l/p_l` storage references. Capture `s_copy[]` once, serially, while the owning recurrent metadata is stable; never recompute it from shared `rs_idx` during graph input. A GDN state write/read dependency between adjacent chunks must be represented by an explicit event or ordered stream edge; immutable host metadata alone does not make shared recurrent rows safe for concurrent writes.

Capture QSA input plans/selector arrays per frame (including indexer cache cell mappings and any pooled-plan vectors). Do not share mutable `qsa_multi_*` tables or graph-input host buffers. Serialize the corresponding global cache commit/registration operations.

This preserves all current graph concrete casts without edits to `llama-graph.cpp` or Qwen4Exp. If using a non-concrete snapshot object instead, the alternative is a graph-input interface conversion across those callers, which is broader and belongs to the context/graph lane.

## Write/dependency boundary

The graph executor calls `mctx->apply()` before graph input setup (`llama-context.cpp:2160-2173`; batch path at `4573-4591`). `apply()` mutates attention and recurrent cell metadata. For two adjacent chunks of one sequence, frame B's GDN state and valid attention prefix depend on frame A. Thus the useful overlap is stage/layer pipelining with explicit A→B dependencies, not two independently submitted whole graphs that both mutate the same cache cursor.

Safe sharing boundary: model weights and underlying per-layer KV/indexer/GDN tensors remain shared. Frame-local data includes ubatch pointers/storage ownership, slot maps, `n_kv`, K/V/indexer destination indices, masks, QSA selectors, recurrent head/zero-state and s-copy rows, scratch tensors, and graph result/input buffers. Disjoint KV destination cells are necessary but not sufficient: GDN state rows and QSA pooled-table commits need their own ordering guarantees.

## Verification status

## Memory-layer implementation

The memory-side frozen path is now implemented in the four file pairs above, with the private helper `src/llama-memory-frame-snapshot.{h,cpp}`. The adapter is an actual `llama_memory_hybrid_idx_context`, so the existing graph downcasts remain valid. `freeze_current()` is opt-in, requires that the live context has successfully applied its current ubatch, and returns null for unsupported ubatch shapes or active multi-sequence pooled-QSA mode.

- Attention and indexer child contexts copy one slot, the committed `n_kv`, an owned ubatch, and the current sequence's KV-cell metadata. KQ masks, position buckets, predecessor-token inputs, and QSA selectors read those copied cells. K/V tensors remain the original shared cache allocations.
- The recurrent child captures `n_rs`, head, zero-state index, cache size, and every `s_copy` value once. That capture consumes each live rollback index in the same order as the original graph input; frozen getters never read or reset `rs_idx` again. The per-layer R/S/P tensors remain the original allocations.
- Frozen graph input calls build KQ masks, predecessor-token rows, position buckets, and QSA cell/block selectors from copied cell metadata. The mutable pooled-block fast path is disabled for a frozen frame, and the multi-sequence pooled path fails closed until its table commits have their own snapshot transaction.
- Frozen `apply()` is a no-op; the owner must apply/commit the live ubatch first, freeze immediately afterward, then pass the frozen concrete context to the graph. The context/graph lane owns that call-site integration and any A-to-B GPU event dependency.

CPU immutability fixture PASS for both raw-pointer and allocr-owned shared ubatch storage: cloned tokens, positions, sequence pointers and output markers survive live-source mutation; recurrent cursor/s_copy value state is one-shot and stable; copied QSA cell metadata, including the position-to-token index, stays unchanged after the live cell vector is mutated; output-producing ubatches fail closed. `-fsyntax-only` PASS for `llama-memory-frame-snapshot.cpp`, `llama-kv-cache.cpp`, `llama-memory-recurrent.cpp`, `llama-memory-hybrid.cpp`, and `llama-memory-hybrid-idx.cpp` against the private source/include tree. No engine build, GPU initialization, model load, or inference ran.

Remaining gate: the context/graph owner must call `live.apply()` exactly once before `freeze_current()`, then consume frozen frames with explicit dependencies for adjacent GDN/KV writes. This memory-layer API alone is not evidence that two full graphs can overlap or that full-model semantics/throughput are correct.
