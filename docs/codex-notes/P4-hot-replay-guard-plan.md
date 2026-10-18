# P4 selective hot-replay guard implementation plan

Date: 2026-10-15
Status: implementation plan written before source edits. Scope limited to the private Meta backend and llama-context integration below.

## Goal and hard safety rule

Add an opt-in process-shared recursive guard for Meta shared-buffer descriptor allocation/reset and Meta graph preparation. When enabled with `GGML_META_REPLAY_GUARD=1`, allow the target API call to skip the broad outer runner mutex only when every Meta backend on that context has a stable UID, committed merge decision, and all child CUDA graphs are ready for replay. If any condition is unknown, missing, unsupported, or false, take the guard around the host graph submission path. Do not hold it while draining device work.

The user-facing runner remains guarded until the CPU fixture and root-granted GPU island fixture pass. Default behavior remains off.

## Ownership / files

Only write:

- `src/ggml/src/ggml-backend-meta.cpp`
- new private `src/ggml/src/ggml-meta-replay-guard.h`
- `src/src/llama-context.cpp`
- new isolated fixture files under `fixtures/meta-replay-guard-20261015/`
- `notes/P4-hot-replay-guard-plan.md` and result notes

No public ggml/llama headers, CUDA kernels, backend scheduler files, other agents' files, engine build, model launch, or GPU use.

## Source contract

- A graph allocation/rebuild can mutate Meta shared-buffer descriptor state: `ggml_backend_meta_buffer_init_tensor` chooses the shared compute arena; `ggml_backend_meta_get_split_state` may insert/clear the buffer-owned cache; `ggml_backend_meta_graph_compute` rotates/reset-replaces generations and performs graph rewrites.
- `llama_context::process_ubatch`'s plain `graph_reused` flag is insufficient. Draft-cache restoration can reset/allocate while setting `graph_reused=true`; Meta first-use can defer P2P merge and merge on a subsequent reused graph call.
- A safe hot predicate must fail closed unless: the graph UID matches the Meta backend context's prepared UID; P2P merge decision is committed (or a supported non-P2P fallback is explicitly known ready); merged/chunk graph lists are complete; and every child cgraph returns true from the CUDA `graph_ready` callback. With P2P enabled but no committed merged graph, or with no readiness callback, the predicate is false.
- CUDA `graph_ready` is a per-child read of that child backend's graph cache; V6 uses separate owner threads/Meta contexts. Preserve runner one-time device/OneCat warmup and do not assume a warmed graph survives eviction or a >5 s cache freshness expiry.
- Input mutation after a ready check comes from the current `llm_graph_result::set_inputs` objects and its memory context. Keep graph-input buffer identity context-local; if runtime evidence contradicts this, fail closed and retain the lock.
- Public reserve/reset and initialization paths must share the lock. Drain a context before entering the shared lock for reserve/reset/destruction; never wait on pending GPU work while holding the global guard.

## Implementation steps

1. Add a private C-linkage guard API and C++ RAII scope in the new private header. Define one `std::recursive_mutex` and cached opt-in env check in `ggml-backend-meta.cpp`. Recursive behavior is required because llama-context cold scopes call Meta callbacks that also take the guard.
2. Protect Meta buffer generation selection/init, reset/free, simple-tensor map lookup, split-state cache resolution, and buffer get/set metadata lookup with the same guard when opt-in. Keep all CUDA execution/drain outside this guard.
3. Add a private `ggml_backend_meta_graph_replay_ready(backend, graph)` query:
   - false for a graph UID mismatch or missing graph/backend context;
   - false if P2P is present but merge hasn't been attempted/committed, merged graphs/chunks are missing, or any chunk is not CUDA-graph-ready;
   - false if readiness cannot be queried;
   - true only after replay-ready per-child graphs exist.
   Do not mutate `seen_shapes`, `merge_tried`, tensor arenas, or graph cache state from this query.
4. In `llama_context::process_ubatch`, acquire the guard after any required per-context scheduler drain and before scheduler reset, graph build, graph allocation, and the first Meta graph dispatch. Hold through host enqueue only and explicitly release after `graph_compute` returns.
   - Include the draft-cache restore branch even though it later sets `graph_reused=true`.
   - For an otherwise reused graph, acquire only if any Meta backend's replay-ready query fails. This catches the deferred second P2P merge/capture call.
   - Leave context-owned input assignment and graph dispatch unguarded only when the readiness gate passes.
5. Guard `sched_reserve` and `graph_reserve` around scheduler reset/build/reserve after the context has been synchronized. This also covers public `llama_graph_reserve`. Ensure destructor/clear paths drain the current context before releasing graph generations; do not add a cross-context GPU wait.
6. Keep the environment opt-in and preserve existing behavior when unset. Do not change MTP chain state, argmax output, pinned staging, or graph layout.

## Verification gates before root removes the runner guard

1. CPU-only fixture: two Meta contexts sharing one static weight buffer, A/B/A graph allocation/reset, concurrent host scheduling, output exactness, descriptor/generation identity, split-cache counters. Confirm after each warmed ready graph that shared weight-buffer compute index/cache do not change on replay. The fixture must fail if cache pointers or simple-view sources cross contexts.
2. CUDA fixture: root-controlled 3+3 island run, same two-context A/B/A sequence; assert each context's Meta UID/merge readiness, all child graph-ready callbacks true before unguarded replay, exact outputs, and no shared-buffer generation/cache mutation on hot dispatch.
3. Negative cases: missing CUDA graph callback, cold shape, missing merged graph, graph UID mismatch, draft-cache restore, graph reserve/reset all take the guard. Log exactly why eligibility failed.
4. No engine/model performance run here. Root separately owns final throughput comparison and runner-guard removal.

## Expected result / limitation

This reduces the broad guard on stable, already-merged, replay-ready shapes. New signatures require serialized warmup and at least the deferred Meta merge/capture path; graph-ready cache eviction or unknown backend state falls back to guarded submission. If the new private readiness query cannot prove the complete child path without dereferencing mutable graph-cache state, keep the outer guard for that case.


## Design correction from the CPU fixture

The original root cgraph and scheduler Meta split cgraphs have separate UIDs. The fixture observed root UID 1 while the Meta callback dispatched split UID 2 and stored context UID 2. Calling the strict Meta graph query with the root graph therefore fails closed after every dispatch.

The final caller-side gate is scheduler-bound: ggml_backend_sched_meta_replay_ready(sched, root_graph) requires the scheduler to still own the exact allocated root graph pointer and UID, then checks strict readiness for each actual scheduler split assigned to a backend. Scheduler reset clears the binding; successful graph allocation records it. The strict Meta graph API remains unchanged for internal dispatch. This also prevents a cached ready bit for graph A from authorizing graph B after a scheduler switch.
