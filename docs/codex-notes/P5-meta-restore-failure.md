# P5 Meta-backend A/B/A restore failure

## Follow-up (2026-10-15)

The initial recommendation below to categorically exclude Meta devices is superseded by the bounded restore fix in `P5-meta-binding-reset.md`. A CPU-only two-member Meta A/B/A fixture with views in both families reproduces stale descriptors for A's Meta-backed view (`2` missing member mappings) after switching to B. It confirms that capture must include a view whose data pointer was already set before scheduler allocation and whose inherited source buffer has WEIGHTS usage. Clearing and reinitializing that graph view together with COMPUTE tensors restores all mappings and the exact result. This is not full-model or CUDA validation; P5 remains disabled until the root validates the actual MTP path.

## Evidence

The real-server log `standard-semantic-20261014/results/20261014T150334Z/server.log` shows this sequence at the MTP context:

1. `draft graph cache build tokens=1 outputs=1 entries=2`
2. `draft graph cache active reuse tokens=1 outputs=1 entries=2`
3. `draft graph cache restore tokens=1 outputs=1 entries=2`
4. Meta backend aborts at `GGML_ASSERT(bcj.nodes[i])` in `ggml-backend-meta.cpp:3039`.

The assertion's direct value comes from `ggml_backend_meta_buffer_simple_tensor(node, j)`, which returns null when the tensor is absent from the currently selected simple-tensor container (`ggml-backend-meta.cpp:545-554`).

## Root cause

P5 retains a cached `llm_graph_result` and its original `ggml_cgraph`, but it does not retain the Meta backend's per-graph simple-tensor descriptors or subgraph mapping.

On the A -> B transition, Meta rebuilds B's subgraphs. It clears the backend context's prior `tensor_arenas` owner list, rotates the two compute descriptor generations, and may reset the inactive generation's contexts and `simple_tensors` map (`ggml-backend-meta.cpp:2998-3022`). The Meta backend therefore keeps only the current graph generation's mapping.

P5 later restores A by resetting the scheduler and calling `ggml_backend_sched_alloc_graph` on A's old graph (`llama-context.cpp:2525-2545`). Its tensors still have their original non-null `buffer` and `data` fields. The graph allocator's `ggml_gallocr_init_tensor()` invokes backend buffer init only for tensors with null buffer/data (`ggml-alloc.c:970-995`); `ggml_gallocr_alloc_graph()` does not clear those fields before rebinding (`ggml-alloc.c:1048-1080`). Thus Meta's buffer-init callback does not switch to a new generation or rematerialize A's simple-tensor descriptors.

When Meta compute sees A's old UID differs from the active backend UID, it correctly enters the rebuild path (`ggml-backend-meta.cpp:2966-2968`). That rebuild asks the current container for A's descriptors and receives null, then asserts at lines 3038-3039. This is a mismatch between the lifetime of P5's outer graph objects and Meta's single-active-generation descriptor mapping. The CPU scheduler fixture passed because it exercised allocation signatures with the CPU backend, not Meta's descriptor-generation lifecycle.

## Minimum safe fix

Keep P5 disabled for any model whose device set contains a Meta device. In the P5 eligibility predicate, add a predicate equivalent to:

```cpp
std::none_of(model.devices.begin(), model.devices.end(),
        [](const auto & device) { return device.is_meta; })
```

Then retain the existing opt-in cache for non-Meta contexts. This avoids restoring graph objects whose backend-specific mapping cannot be restored. For the current AC922 island topology, the feature should remain off.

Do not try to fix this by forcing `cgraph->uid = 0`: Meta already rebuilds A because A's UID differs from the active UID. The missing simple-tensor mapping, not failure to request a rebuild, causes the assertion. A real Meta-enabled cache needs a Meta backend change that retains or reconstructs descriptor containers and subgraph maps keyed by graph identity/generation, with scheduler allocation lifecycle support. That exceeds the P5-only scope.

## Small Meta A/B/A fixture plan (not run)

Use a tiny CPU-only Meta backend made from two CPU devices, so the descriptor lifecycle can be tested without CUDA, a model, or full server startup.

1. Create a Meta device from two CPU device handles with a split callback that returns a valid mirrored split state for the fixture tensors.
2. Create one scheduler on its Meta backend and two persistent GGML graph families with distinct shapes/node identities: A and B. Allocate and compute A, then reset the scheduler, allocate and compute B.
3. Reset and allocate the original A graph object again, exactly matching P5's restore shape. Before graph compute, check every Meta-backed node with `ggml_backend_meta_tensor_get_simple(node, rank)`; current code is expected to return null for at least one A node, followed by the same `bcj.nodes[i]` assertion if compute is invoked.
4. Controls: A/A without switching should pass; A/B followed by a newly constructed A graph should pass. These distinguish stale per-graph metadata from unsupported operation shapes.
5. After a Meta-aware mapping-lifetime fix, rerun A/B/A and assert all simple tensors resolve, compute status succeeds, and A's output equals fresh-A.

This fixture is a proposed next validation only. No Meta fixture, source edit, build, GPU operation, or model request was run for this diagnosis.

## Run state

P5 remains disabled in the active runtime. Before enabling it, validate the binding-reset restore path in the actual MTP model runtime; retain Meta exclusion as fallback if that path fails.
