# Recursive Meta feasibility audit: `Meta2{Meta3A, Meta3B}`

**Scope:** static audit of the frozen `/home/debian/ac922-monster/experiments/standard-opt-20261014/src/ggml/src/ggml-backend-meta.cpp` and `ggml-backend.cpp`. No core edits, build, GPU execution, or model load.

## Verdict

The Meta device constructor itself does **not** forbid another Meta device as a child: it stores its immediate `devs[]` without checking their device type. Several buffer and `supports_*` paths are structurally recursive. However, an outer `Meta2{Meta3A, Meta3B}` cannot currently initialize its backend when it has two members. Backend init assumes each immediate member belongs to a registered leaf backend and looks up the outer collective API through that child’s backend registry. Meta devices set `reg=nullptr`; the registry lookup asserts on null. This is a deterministic source-level blocker before graph execution.

Even after making that lookup safe, nested Meta is not a ready-made DPA2/EP6 scheduler. The direct-P2P collective and `graph_ready` hooks are fetched from one backend registry and operate on the immediate members; they do not flatten and coordinate the six physical CUDA ranks. The present EP private op parameters also overwrite, rather than compose, nested expert offsets. So this is **not a drop-in topology switch**. A useful nested implementation would need explicit recursive collective ownership and hierarchical split/EP metadata, or an explicit outer Meta fallback that is tested as a distinct slower path.

## Creation and buffer path: recursion is partly present

- `ggml_backend_meta_device()` stores the supplied device pointers in `simple_devs`; it only asserts `n_devs <= GGML_BACKEND_META_MAX_DEVICES` (`ggml-backend-meta.cpp:390-421`). There is no `simple_dev must not be META` assertion.
- Meta device construction writes `reg=nullptr` into the device object (`ggml-backend-meta.cpp:413-417`). `ggml_backend_dev_backend_reg()` returns that field (`ggml-backend.cpp:623-626`).
- `supports_op` walks immediate members and calls `ggml_backend_dev_supports_op()` on each (`ggml-backend-meta.cpp:242-247`), which in turn calls the member’s device interface. For a Meta3 child this recurses through its own members. The Meta buffer support check also compares the immediate device list against the buffer type’s immediate list (`249-265`), so a child Meta3 buffer can match its exact child Meta3 device.
- Nested buffer allocation is also recursive in shape: the parent collects each immediate child’s buffer type (`522-540`), allocates each child buffer (`3102-3124`), constructs its own per-member tensor descriptors, and calls `ggml_backend_buffer_init_tensor()` on a child buffer (`2441-2445`, `2571-2593`). When that child is a Meta buffer, its init callback can build the next descriptor layer. This suggests the dense `outer=MIRRORED, inner=TP3` weight-copy layout is representable at the buffer layer, subject to ordinary op/shape support.
- Split-state is attached per Meta device instance (`ggml_backend_meta_device_context` stores one callback and userdata at `142-152`). Each layer uses its own immediate member count and descriptor array. Therefore the abstraction can describe two hierarchical levels if callers create distinct callbacks/userdata and operations preserve the level boundary. It does not automatically flatten 2×3 into a six-rank split.

This is evidence that recursive Meta **objects and buffers are partly composable**, not proof that full graph execution, cache ownership, or collectives compose correctly.

## Deterministic backend-init blocker

`ggml_backend_meta_device_init_backend()` constructs `ggml_backend_meta_context`. For an outer Meta2, `n_devs=2`, so it enters the collective setup unconditionally:

1. At `ggml-backend-meta.cpp:3445-3450`, it asks `ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0]))` for `ggml_backend_comm_init`.
2. With Meta3A as the first child, `ggml_backend_get_device(simple_backends[0])` is Meta3A and its registry is null (`reg=nullptr` above).
3. `ggml_backend_reg_get_proc_address()` contains `GGML_ASSERT(reg)` before it can return null (`ggml-backend.cpp:688-694`).

The next setup block repeats the same assumption for `ggml_backend_p2p_ar_init` (`ggml-backend-meta.cpp:3452-3474`). If `GGML_META_GPU_SPANS=1`, the earlier CUDA span lookup repeats the null-registry issue at `3427-3433`.

This blocks `Meta2{Meta3A,Meta3B}` before it can fall back to any graph implementation. It is a narrow implementation assumption, not a mathematical limitation of nested groups.

## Collective, AllReduce, and GraphReady boundary

- A normal Meta3 obtains the CUDA registry from its first physical child, then creates its P2P-AR/comm context from the 3 child backends (`3445-3481`). An outer Meta2 cannot use that API as written: its immediate child backends are Meta3 wrappers, not the six CUDA backends expected by the CUDA plugin.
- Meta graph replay/capture asks the backend-provided `graph_ready` callback for each immediate backend’s child graph (`ggml-backend-meta.cpp:4683-4689` and `3660-3693`). The outer context has no registry from which to obtain this callback. A nested readiness query could recursively ask each Meta3 child, but no such composite registration path is wired here.
- When no backend collective succeeds, Meta has a generic butterfly fallback that copies partial tensors between child backends and adds them (`4787-4888`). With outer children Meta3, this would copy Meta3-backed mirrored values and execute the add through Meta3. It is a plausible correctness fallback after fixing init, but it is not evidence of XBUS P2P, overlap, or fast six-rank collectives. The current constructor’s null-reg assertion prevents even reaching it.
- `GGML_META_ASYNC_COPY` specifically recognizes a pair of mirrored three-member Meta devices and retrieves a physical CUDA copy helper only after checking each child backend registry (`ggml-backend-meta.cpp:3864-3979`). This is useful for an island-to-island copy between Meta3 backends; it does not supply a general outer Meta2 collective.
- Meta event wait’s fast CUDA path resolves registry names on event devices (`3997-4027`). With a nested Meta event, those immediate event devices are Meta3 wrappers with null registries, so recursive event-query/wait logic also needs a guard or leaf-recursive implementation.

## Nested EP offset composition is not implemented

An EP6 shape split as `outer EP axis2: 512 → 256 + 256`, then `inner EP axis2: 256 → 85/85/86` has a second correctness issue beyond collectives:

- Outer Meta’s `MUL_MAT_ID` descriptor computes `first` and `total` for its immediate two slices, then writes those through `ggml_ep_private::set()` (`ggml-backend-meta.cpp:2466-2479`).
- Inner Meta materializes another `MUL_MAT_ID` descriptor over each 256-expert slice and writes a local `first,total` again at the same code path.
- `ggml_ep_private::set()` assigns its `offset_param` and `global_experts_param` slots; it does not add an inherited base offset (`ggml-ep-private.h:20-26`). The inner 0/85/170 offset therefore replaces the outer 0/256 base, and the inner total becomes 256. CUDA MMQ/MMVQ consumers use that offset as the global route-ID interval (`mmq.cu:197-201`, `mmq-moe-grouped.cu:214-216`, `mmvq.cu:1824-1828`). Router IDs are global 0..511, so rank ownership would be wrong for the second outer shard unless the parent base is carried and composed or IDs are explicitly localized.

This exact nested EP6 route cannot be considered correct from the current fixture/API contract. Add a parent-offset field or an explicitly localized route-ID protocol, then test boundaries and all 512 owners before measuring it.

## Dense, token ownership, and KV state

- `outer MIRRORED` dense weights followed by an inner Meta3 TP split is the easiest case: the outer buffer can copy a full tensor into the child Meta buffer, and the child callback can TP-split it. This is a placement observation only; graph collectives for the inner/outer groups remain separate.
- DPA2’s two islands process distinct request lanes. Merely mirroring dense weights does not assign different sequence IDs or route two independent input/KV states to the correct child. That needs either separate context/graph inputs per island or explicit outer token/sequence split states carried through every op.
- The current unified KV cache allocates K/V shaped by embedding dimension, pool-cell count, and one stream (`llama-kv-cache.cpp:86-88, 239-240`). Cells use sequence membership in the cache metadata, rather than a contiguous per-sequence dimension. A simple outer tensor split cannot guarantee that every cell owned by a sequence follows that sequence when free cells are allocated/reused. A real DPA owner map must be connected to cell allocation, K/V writes, rollback/reuse, and attention reads.
- GDN R/S cache allocation follows layer backend placement and its split callback (`llama-memory-recurrent.cpp:80-98`, `llama-model.cpp:316-321`). It likewise needs bank-local state ownership; a nested Meta buffer alone does not create separate sequence-state pools.

## Minimal interface if the team wants to keep the nested approach

1. **Make init recursive-safe.** When a member is Meta, do not call `ggml_backend_reg_get_proc_address()` on its null registry. Define a composite collective interface that can flatten the six CUDA leaves for EP6, or a deliberately generic outer reduce path for the two Meta3 children. Keep each inner Meta3’s TP3 collective independent.
2. **Compose graph readiness and events.** Outer readiness must be the conjunction of the two inner Meta3 graph/capture readiness states plus outer copy/collective dependencies. Event support/query must recurse to leaf events rather than resolving a registry on a Meta wrapper.
3. **Carry hierarchical split coordinates.** For nested EP, propagate `(outer_base, inner_base, global_expert_count)` and derive final offsets, rather than letting each `ggml_ep_private::set()` overwrite the prior level. Define whether local partials are reduced inside TP3, EP6, or both, and make the final-output flag express that order.
4. **Give requests and state owners to islands.** Provide separate bank-owned token inputs and KV/GDN state pools, or a tested token-owner split contract. Do not let an outer `MIRRORED` default duplicate one sequence’s cache into the other bank.
5. **Prove with one real expert block before a model run.** Exercise two Meta3 children, six physical expert slices, outer expert ranges 0–255/256–511, inner ranges, global router IDs, one final reduction including TP3 shared experts, graph capture/replay, and KV/state isolation.

**Recommendation:** keep nested Meta as a possible implementation route, but do not describe it as immediately usable. The constructor and tensor-buffer plumbing are encouraging; the unconditional null-registry collective lookup and non-compositional EP offsets are concrete blockers. A tiny fixture after those interface changes is required before selecting it over an explicit two-island executor plus a separate EP6 group.
