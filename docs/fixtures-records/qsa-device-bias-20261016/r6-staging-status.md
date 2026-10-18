# QSA device-bias r6 staging status

Date: 2026-10-16. This is a private fixture/staging result only. No core source was changed,
no engine build or model load was run, and this turn did not access the GPU.

## Placement correction

The r4 Meta3 failure is consistent with pinning metadata-only views to a backend different from
their source:

- `ggml/src/ggml-backend.cpp:775-777` defines scheduler view ops as VIEW, RESHAPE, PERMUTE, and
  TRANSPOSE.
- `ggml/src/ggml-backend.cpp:3869-3875` shows `ggml_backend_sched_set_tensor_backend` only writes
  the scheduler backend ID; it does not move/allocate tensor storage.
- `ggml/src/ggml-backend.cpp:1306-1334` assigns graph INPUT leaves with no preexisting buffer to
  the last scheduler backend (assumed CPU). `:1651-1672` propagates a view's source backend only
  when the view has no assigned backend. Thus the old explicit Meta assignment on a reshape could
  conflict with the CPU input that owns its aliased storage.
- `ggml/src/ggml-alloc.c:623-628` does not independently allocate view tensors. `ggml/src/ggml.c`
  `ggml_reshape_4d` constructs a view over its source (`:3987-4004`).
- `ggml/src/ggml-backend.cpp:1784-1804` creates a backend-local copy when a split consumes a
  source from a different, unsupported buffer type; `:2335-2394` copies host inputs to that split
  backend before compute.
- `ggml/src/ggml-backend-meta.cpp:249-265` accepts only Meta buffer types wrapping the same
  ordered member devices, so a CPU buffer is not a compatible Meta buffer.

r6's `pin()` therefore skips only scheduler metadata view ops; it still pins casts, arithmetic,
FILL, SET_ROWS, CONT, and other compute nodes. A cast/compute node remains explicitly assigned to
the target Meta3 backend. Host metadata views retain the input's CPU assignment, allowing the
scheduler's normal split-copy path to create Meta-owned inputs before the Meta graph evaluates its
dependencies. This is source-based reasoning; Meta3 behavior still needs the authorized GPU fixture
to confirm it.

The fixture now prints `TENSOR_BIND` lines for each metadata input and graph node before/after
scheduler allocation. Each line includes type, op, input flag, shape, buffer name, Meta-buffer bit,
backend hint, view source, and up to three source tensors with type/op/buffer. It also checks that
views inherit their source backend, while every compute op stays on the requested backend and is
reported supported.

## Verification

- r6 patch `qsa-device-bias-v47-r6.patch` dry-run/applied to the preserved clean V47 tree; applied
  staged files match the patch output. SHA-256:
  `61c0ea58db9ef29bab8e0d17f52fbaaa0b6615ad887b024941b300106031b599`.
- `cpu` GGML graph and independent Python contract passed, including exact bias bytes, stable
  top-k IDs, final masks, compact SET_ROWS indices `[B,1,S,1]`, and 65,536-block case.
- `cpu-sched` mode passed the same scenarios and emitted 88 pre-allocation and 88 post-allocation
  `TENSOR_BIND` lines. Its CPU allocation shows reshape views sharing the source's CPU buffer.
- `-fsyntax-only` passed for the staged `llama-memory-hybrid-idx.cpp`, `models/qwen4exp.cpp`, and
  `qsa-device-bias-ggml-fixture.cpp` in `syntax-only-v47-r6.log`.
- Tiny CUDA and mirrored Meta3 runs passed all five fixture gates each: one-stream hole/foreign/
  spare/tail, two-stream sequence membership, 65,536-block positions, same-shape sequence reuse,
  and shape-cache key separation. Every scenario reported exact bias bytes, stable top-k IDs, and
  final masks. Both modes returned 0; no `supports_target=0` nodes or Meta assertions appeared.
  Meta3 used ordered members CUDA0/CUDA1/CUDA2. The logs show metadata leaves and reshape views
  staying on CPU while CPY and compute/scatter nodes are placed on the requested CUDA/Meta backend.
- The two GPU modes completed in 11.4 seconds total. Postflight reported 38-39 C and 1-2 MiB per
  GPU with no fixture process or compute app remaining. The r4 GPU failure logs remain unchanged.
- This proves the shared tiny GGML graph helper and Meta3 allocation/placement path only; it does
  not prove a full-model integration or speedup. Keep `LLAMA_QSA_BIAS_DEVICE` disabled until the
  actual model graph has its own semantic and runtime validation.

Artifacts are isolated under `fixtures/qsa-device-bias-20261016/`; primary r6 files are the patch,
`apply-v47-r6/`, `run-qsa-device-bias-ggml-fixture-r6.sh`, the two r6 CPU logs, and
`syntax-only-v47-r6.log`, `gpu-cuda-single-v47-r6.log`, and `gpu-meta3-v47-r6.log`.
