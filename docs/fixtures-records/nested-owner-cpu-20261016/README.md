# Nested Meta child-weight owner regression (CPU only)

**Status:** source fixture prepared; public-API syntax check passes. Awaiting root's V63 freeze before host-link/runtime execution. No core source was changed, no model was loaded, and no GPU was initialized.

## What it covers

The fixture creates `Meta2{Meta3CPU, Meta3CPU}`. Each Meta3 wraps the same CPU backend device three times. It allocates one static F32 `blk.1.ple_conv1d.weight` buffer on the outer Meta2 WEIGHTS buffer, then creates two independent root Meta2 backend contexts and two schedulers that share that exact tensor/buffer.

Each scheduler uses its nested Meta backend followed by the required CPU fallback backend, and builds a small graph with four `ggml_view_2d` tap views into the static root followed by `ggml_cont`. It calls public `ggml_backend_sched_alloc_graph()` only. It never calls graph compute, CUDA, private Meta leaf accessors, or owner registry functions. It reads each non-contiguous tap view row-by-row through public `ggml_backend_tensor_get()`, using the view's actual row stride, and checks exact F32 values and tap offsets.

The lifetime sequence is:

1. Allocate both schedulers while A and B remain alive; verify A and B reads against the same static buffer.
2. Reset scheduler A, reset and rebuild A's graph in the same `ggml_context` at the same view addresses/offsets, and allocate it again.
3. Verify rebuilt A and still-live B, then destroy A's scheduler/context and verify B once more.

This exercises scheduler owner registration for real nested generated child WEIGHTS views, graph-context address reuse, owner unbind/rebind, and isolation between two live root Meta backend contexts. V62 should reproduce its missing child-view owner failure during scheduler allocation or public `tensor_get`; V63 should pass every row read and the post-A-teardown B reads.

## Safety and environment

`run-cpu-only.sh` starts with a clean environment and requires:

- `GGML_META_NESTED_EXPERIMENTAL=1`
- `GGML_META_CONTEXT_VIEWS=1`
- `GGML_META_NESTED_GPU_AR=0`
- Meta FIFO, async copy/input, CPU async-copy, GPU event wait, CUDA copy wait, and Meta single-copy events all disabled

The binary links only `libggml-cpu` and `libggml-base`; the build script rejects any dependency on `libggml-cuda`, CUDA runtime/driver, or NCCL. No `ggml_backend_graph_compute*()` call is present.

## Run after V63 is frozen

```bash
AC922_MONSTER_SRC=/home/debian/ac922-monster/experiments/standard-opt-20261014/src \
NESTED_OWNER_RUNTIME_LIB=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v63/lib \
NESTED_OWNER_BINARY=/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nested-owner-cpu-20261016/nested-owner-cpu-v63 \
bash /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nested-owner-cpu-20261016/build-cpu-only.sh

NESTED_OWNER_RUNTIME_LIB=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v63/lib \
NESTED_OWNER_BINARY=/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nested-owner-cpu-20261016/nested-owner-cpu-v63 \
bash /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nested-owner-cpu-20261016/run-cpu-only.sh
```

The source-only check is independent of a frozen runtime:

```bash
AC922_MONSTER_SRC=/home/debian/ac922-monster/experiments/standard-opt-20261014/src \
bash /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nested-owner-cpu-20261016/run-static-checks.sh
```

The source uses only public headers (`ggml.h`, `ggml-alloc.h`, `ggml-backend.h`, `ggml-cpu.h`) and APIs. It does not include `ggml-backend-impl.h` or call `ggml_backend_meta_tensor_get_simple()` / `ggml_backend_meta_alloc_ctx_tensors_from_buft()`.
