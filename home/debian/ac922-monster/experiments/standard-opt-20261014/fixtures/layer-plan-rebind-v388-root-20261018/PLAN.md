# V388 private layer-plan rebind

## Goal

Avoid rebuilding the scheduler context and split graph on every cached ordinary layer-prefill reuse. Keep the current cache-rebind gate and restore step, then try a private scheduler plan rebind before calling `ggml_backend_sched_reset`.

## Candidate contract

- `LLAMA_LAYER_PREFILL_PLAN_REBIND1=1` opts into the private API inside the existing `LLAMA_LAYER_PREFILL_CACHE_REBIND` path. The existing caller gate still requires pipeline layer split, exactly six plain CUDA backends plus CPU, no Meta backend, callbacks, outputs, MTP, draft, FIFO, frame, or other excluded state.
- The private API requires exactly four scheduler copies and declines `layer_single_device_inputs`, so this candidate stays off any eight-slot or physically aliased copy ring.
- `ggml_backend_sched_try_alloc_layer_plan_rebind(sched, graph)` returns `1` after a successful rebind, `0` before any scheduler/tensor mutation when the plan is unsupported or validation declines, and `-1` after mutation if gallocr allocation fails.
- The caller restores root bindings once before trying the plan. On `0`, it resets the scheduler, replays owners, and performs the existing full split allocation without restoring twice. On `-1`, it stops.

## Plan lifetime and validation

The scheduler owns a separately `new`-allocated plan because its struct is `calloc`-created. The plan is first captured after root restore and before reset, while the old scheduler context, four-copy table, and split arrays still exist. It is deleted before scheduler reset, split reconstruction, and destruction; its destructor releases vectors without following tensor pointers.

Before mutation the candidate checks root and scheduler graph identity, tensor type/layout/op/op-params/source/owner snapshots, scheduler graph-copy node and leaf arrays, backend ID arrays, copy-table entries, split ranges/input arrays/graph views, context identity, four-slot state, and host CPU graph inputs. Unsupported GPU inputs, Meta state, callbacks, pending Turbo state, or enabled FIFO decline to the existing full-split path.

The plan records root cross-backend source sites and their four distinct destination copies, scheduler graph-copy node slots, and scheduler-owned helper-view `src[]`/`view_src` sites that point to those destination copies. Helper discovery uses non-root scheduler nodes with non-null `view_src`, regardless of opcode, because `ggml_view_tensor` allocation dependencies use `GGML_OP_NONE`. It excludes CPU-owned graph input roots from rotation, preserves the CPU input root and graph-copy leaf order, and retargets only recorded sites. It restores `INPUT|OUTPUT` flags on scheduler graph inputs and clears `buffer`, `data`, and `extra` only on captured scheduler-owned helper views.

After validation it advances the existing four-slot ring, assigns fresh root and split UIDs, and calls the unchanged `ggml_backend_sched_alloc_splits` routine. The existing gallocr allocation check, backend drain, reserve, and retry behavior remains in that routine. Copy events, peer fences, and R389 pinned-upload reader events are not modified.

## CPU fixture

`scheduler-plan-rebind-v388.cpp` uses three mock CPU backends with distinct host buffer types, the actual ggml scheduler and gallocr from an isolated candidate library build, and the real CPU graph executor. Mock backend graph optimizers add scheduler allocation-dependency views that reference cross-backend copies.

The prepared fixture checks initial full-split parity, A/B/A input replay, all four destination slots and wrap, `GGML_OP_NONE` dependency-view retargeting and data reinitialization, fresh root/split UIDs, a layout decline before mutation, full-split fallback after decline, and allocation failure after exactly one root restore. The fixture and scripts have not been built or run. CPU mocks do not prove CUDA event safety.

## Source pins and review status

The patch is based on the read-only AC922 source snapshot and hashes in `parent-hashes.txt`. The pre-existing `parent/` snapshots are older R381 material and are kept for history; they are not the patch base. Candidate files live under `candidate/`; no active source tree was edited and no build, GPU, model, or process action was performed.

The patch is a staged review candidate. Before applying it, review the scheduler field relocation assumptions and compile/run the CPU fixture against the isolated patched ggml libraries. CUDA reader-event and peer-fence correctness remains pending a later bounded CUDA integration check.
