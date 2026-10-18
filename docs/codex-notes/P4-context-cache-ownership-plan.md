# P4 context cache ownership: safe path toward removing CPU submit guard

Date: 2026-10-15
Status: read-only source audit and bounded reproducer design; no code/build/GPU changes.

## Conclusion

The runner's process mutex is broader than a per-context lock, but removing it is not yet justified by source alone. It covers `llama_decode()` host graph construction/submission (and the speculative draft/process calls) and currently prevents two contexts from concurrently mutating descriptor state attached to their shared Meta model-weight buffer.

The primary concrete hazard is not a global `ggml_backend_meta_context` object. Each Meta backend context owns its graph state and its `tensor_arenas`. The shared model buffer separately owns the mutable `stc_compute[2]`, its active/next indices, and `split_state_cache`; graph views of shared weights can be stored there. The per-context shared_ptr generations preserve a container after a rebuild, but they do not make the shared active-index selection or split-cache map context-local or synchronized.

The V3 unguarded CUDA launch failure is compatible with a graph/lifetime race, but does not prove this Meta cache is its cause. A separate lazy CUDA Graph or OneCat first-use failure remains possible. Keep those diagnoses distinct.

## Guard scope observed in P4 V6

The active harness is `/tmp/dual-context-v6.cpp`, with its runner-only `dual_context_submit_guard` in `/tmp/dual-context-v4-submit-guard.h`. The guard is one process-static `std::mutex`, not a mutex inside llama.cpp.

- Target prefill: guard around `llama_decode` at V6 lines 213-215; a separate guard around `common_speculative_process` at 218-220.
- Draft: guard around `common_speculative_draft` at 284-288.
- Target verify: guard around `llama_decode` at 329-337.
- Verify processing: guard around `common_speculative_process` at 391-394.
- The target and draft guards explicitly unlock before stream synchronization (for target, synchronization is at 358). Therefore their hold time measures CPU submission/build work, while queued GPU work drains outside the process mutex.
- The recorded V6 launch says `ROOT_SERIAL_CPU_SUBMIT=1`. The root reports about 20 ms hold for one lane and 28 ms for two lanes. This audit does not attribute those milliseconds to any single engine function.

## Meta buffer state and cross-context call chain

1. Static model tensors are materialized into `ggml_backend_meta_buffer_context::stc_static` by `ggml_backend_meta_alloc_ctx_tensors_from_buft` (source lines 2363-2405). The buffer also creates two compute descriptor containers at allocation time (lines 2338-2360, 2377-2384).

2. The same buffer context owns `stc_compute[2]`, `stc_compute_index`, `stc_compute_index_next`, and `split_state_cache` (lines 488-502). `get_simple_tensor_container(tensor)` checks only whether the tensor address is in `stc_static.simple_tensors`; every other tensor resolves through the buffer's current shared compute index (lines 521-525). There is no Meta-backend-context key.

3. During dynamic tensor initialization, `ggml_backend_meta_buffer_init_tensor` assigns the buffer-wide current index from the buffer-wide next index, then builds descriptors in that selected container (lines 1949-1953). `ggml_backend_meta_buffer_init_tensor_impl` creates simple tensors, copies op/layout fields, resolves sources and views, and inserts `stc.simple_tensors[tensor]` (lines 1746-1946). Its view path resolves the source member and makes the child view inherit the source member's concrete buffer (lines 1850-1884). Thus a compute graph view of a shared model weight can live in the model weight's Meta buffer context even though its descriptor is transient.

4. Split-state metadata is also buffer-owned. `ggml_backend_meta_get_split_state_impl` searches/inserts in the buffer's `std::map`, snapshots the tensor descriptor, and aborts on a stale same-address descriptor (lines 1533-1540, 1595); the iterative cache lookup can clear the whole map when a pointer's tensor snapshot changes (lines 1671-1687). No lock protects these operations. Buffer `get/set` paths call this resolver and retrieve simple descriptors (e.g. get at 2191-2194 and 2217-2243; set at 2060 and subsequent member copies). `ggml_backend_meta_buffer_reset` also clears the map (lines 2298-2307).

5. Per-backend-context state does already exist: `ggml_backend_meta_context::tensor_arenas` retains shared_ptr references to active descriptor generations (lines 2556-2557). On graph rebuild, `ggml_backend_meta_graph_compute` selects buffers used by that context, keeps the current generation alive, flips the *shared buffer* next index, and resets or replaces the other generation based on `shared_ptr::unique()` (lines 2998-3031). Later it resolves every graph node through the buffer's current container (beginning at line 3040; node loop starts around 3045). The retained shared_ptr protects memory lifetime; it does not protect the buffer-wide index/map from concurrent graph allocation or mutation.

Interleaving two graph allocations against the same weight buffer can therefore race in at least three places: current-index assignment, compute-container map/context mutation, and split-state cache insertion/clear. A mutex around only the split-state lookup would not fix wrong-arena selection or the tensor-descriptor map race.

## Other mutable process/model state found

- `ggml_backend_meta_device` has process-static device/context registries (line 289 onward), and `ggml_backend_meta_device_get_buffer_type` has a process-static buffer-type map (line 420 onward); neither has a mutex. These are lazy initialization paths, not per-token decode work, but concurrent context/model initialization should not be assumed safe without a separate check.
- `ggml_backend_meta_init_time_record` has a static mutex/map (lines 1729-1744), but is only called when `GGML_META_INIT_TIMING=1`. V6's recorded environment does not set this variable. `GGML_META_TIMING` accounting is thread-local (lines 2978-2982).
- `qwen4exp_hc_owner` uses a process-static device-to-owner map under a static mutex (qwen4exp.cpp lines 313-350). V6 sets `LLAMA_QWEN4EXP_HC_FUSION=1`; the owner helper is called from each HC graph build at line 396, and its API is used at lines 398 and 462. Existing-owner hits still take the map mutex on each helper call. This is an actual process-level graph-build lock candidate, though the source does not show that it explains the reported 20-28 ms.
- The `qwen4exp_qsa_path_trace` and `qwen4exp_qsa_mblk_trace` static sets/mutexes are guarded by `LLAMA_QSA_PATH_TRACE` (qwen4exp.cpp lines 918-1002); V6 enables `LLAMA_QSA_INDEXER_TRACE`, not that variable. `qwen4exp_qsa_mblk_demote` and `llama_memory_hybrid_idx::qsa_multi_fast_ok` use static lock-protected counters only when `LLAMA_QSA_BLK_DEBUG=1`. They are not established as active in V6.
- Environment-derived local-static booleans/limits are initialized once by C++ and then read-only. They are not mutable per-request state.

## Preferred ownership design

1. Make the model-weight buffer's `stc_static` and static split descriptors a frozen base after load. Static model tensor set/get continues to use that base. Do not put transient graph views into the base descriptor map.

2. Put every dynamic graph tensor/view descriptor and its split-state memo in a generation owned by the calling `ggml_backend_meta_context`. The existing `tensor_arenas` shared_ptr lifetime is the right starting point: one context can replace/reset its own next generation without changing another context's active view descriptors.

3. Bind the owning context/generation during scheduler graph allocation, not only during `ggml_backend_meta_graph_compute`: the buffer `init_tensor` callback is where dynamic view descriptors are materialized, and today it has only `buffer,tensor` parameters. The graph path needs an explicit owner-aware resolver (or equivalent internal owner token) so view-source lookup can use the immutable base for static weights and the calling context's overlay for dynamic graph nodes. A TLS owner is a smaller bridge but must be scoped to synchronous CPU allocation/callbacks and must not leak into Meta worker/CUDA execution threads.

4. Keep any buffer-level overlay index bounded by live graph generations and carry a generation identity in addition to the tensor pointer. Allocator tensor addresses are reused; the existing descriptor-byte snapshot is evidence that pointer-only keys are insufficient. Context destruction/rebuild must unregister overlays only after that context's retained graph descriptors are no longer reachable.

5. Audit host `ggml_backend_meta_buffer_get_tensor/set_tensor` callers before moving overlays. Model weight I/O can use the immutable base cache. Any dynamic graph view passed directly through a generic buffer callback must either carry an owner token or take a controlled fallback path; it must not silently resolve through whichever context most recently changed the shared index.

6. After this ownership split, narrow serialization to the rare shared initialization path if needed (new buffer/base metadata materialization), while normal graph execution and per-context graph rebuilds use context-owned descriptor state. Do not simply remove the harness mutex first.

## Bounded reproducer before any implementation

Use a CPU-only Meta fixture, not a full model and not CUDA:

- Create one shared two-member Meta weight buffer with a small deterministic F32 base tensor and confirm both contexts see the same parent buffer and same immutable member weights.
- Create two separate Meta backend instances/schedulers from that shared Meta device. In each context, construct graphs containing different 2-D views into the shared base weight plus a small deterministic compute op; vary view offsets/shape across A/B/A so stale address/layout reuse is exercised.
- Start two host threads at a barrier. In each thread, repeatedly allocate/rebuild/reset/compute its own scheduler graph while retaining the other context's previous graph generation until its next rebuild. Verify per-device view source, offsets, split shares, and exact outputs after both threads complete.
- Include a same-shape replay after the other context's rebuild; this catches lifetime corruption even when each individual graph build appears valid.
- First run under a race detector if a small CPU build is available. If CPU-only passes, a separate one-minute-or-less GPU microfixture is required to validate CUDA graph capture/replay; the CPU fixture alone cannot prove CUDA graph safety.
- Required before/after instrumentation: buffer pointer, Meta backend-context pointer, selected compute-generation identity, tensor address + descriptor snapshot/hash, split-state shares, and per-context graph UID/rebuild decision. Do not use a random stress loop as a substitute for the targeted A/B/A lifetime sequence.

## Decision gate

- Safe to remove the broad submit guard only after the two-context CPU fixture proves descriptors/caches are context-owned and stable, and a bounded GPU graph-replay fixture passes on both island paths.
- Until then, retaining guard through graph creation/capture is the conservative route. If optimization is needed sooner, prebuild a fixed, bounded set of graph shapes serially and release the guard only for graph executions whose context-owned descriptors and graph UID are already stable.
