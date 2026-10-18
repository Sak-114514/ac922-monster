# Stateful two-frame island pipeline fixture

This is a separate extension fixture. It does not modify llama.cpp core or the existing `island-frame-p4.cpp` fixture.

## Fixture state model

Each island owns one shared KV marker tensor and one shared recurrent-state tensor. Both frame graphs on that island reference the same tensors. Frame input, activation handoff, graph result, and scheduler allocation are distinct per slot.

For each layer and token, the fixture applies this recurrence in float32:

```text
kv[l, p, i] = 0.5 * hidden[i] + gdn[l] + 0.25 * kv[l, p - 1, i]
gdn[l]      = 0.5 * gdn[l] + 0.25 * hidden[0]
hidden[i]   = kv[l, p, i] + gdn[l]
```

The first position has no previous KV row. All values remain binary fractions of small integers, so the CPU reference and GGML graph must match exactly. The stateful graph writes the KV row and GDN cell in place. The next token in a frame depends on the preceding token's GDN write; the next frame depends on the prior frame through an event wait and reads its last KV row and GDN state. This checks state carry, not just activation copying.

The submission sequence is `A0, B0, A1, B1`. `A1` waits for `A0` because they share lower-layer KV/GDN state. `B1` waits for both `B0` state and `A1` activation. This leaves the intended overlap window `A1` versus `B0` while retaining per-layer serial state updates.

The fixture runs two request epochs. Each epoch drains both island events before releasing its two frame slots. It tests ring backpressure at capacity, rejects stale generation tickets after slot reuse, clears the shared layer state only after a full drain, then starts the next request at position zero. Four stage outputs and both islands' entire KV/GDN state are compared with a serial CPU reference.

## Build and CPU policy check

The header-only frame ownership helper is checked without CUDA by:

```sh
c++ -std=c++17 -Wall -Wextra -Werror -pedantic island-prefill-frame-ring-test.cpp -o island-prefill-frame-ring-test
./island-prefill-frame-ring-test
```

The stateful CUDA fixture can be compiled and linked without running it:

```sh
ROOT=/home/debian/ac922-monster
EXP="$ROOT/experiments/hardware-next-20261014"
CUDA=/usr/local/cuda-12.4/targets/ppc64le-linux
g++ -O2 -std=c++17 -D_GNU_SOURCE -Wall -Wextra -Werror \
  -I"$ROOT/src/ggml/include" -I"$CUDA/include" \
  "$EXP/fixtures/island-frame-p4/island-frame-p4-state.cpp" \
  -L"$EXP/runtime-before" -Wl,-rpath,"$EXP/runtime-before" \
  -L"$CUDA/lib" -Wl,-rpath,"$CUDA/lib" \
  -L/usr/local/cuda/lib -Wl,-rpath,/usr/local/cuda/lib \
  -Wl,--no-as-needed -lggml -lggml-base -lggml-cuda -lggml-cpu \
  -lcudart -lcublas -lnccl -lnvToolsExt -ldl -pthread \
  -o "$EXP/fixtures/island-frame-p4/island-frame-p4-state"
```

The executable initializes CUDA and must only be run when the parent grants the fixture an exclusive GPU window. A build or `ldd` check is not runtime evidence. The eventual GPU run should include an Nsight CUDA+NVTX trace and require the A1 kernels on CUDA 0-2 to overlap B0 kernels on CUDA 3-5. The fixture does not load the Qwen model, KV cache, PLE table, MTP, or real semantic prompts.

## Core integration plan (not applied here)

The present model already assigns whole layers to each island: layers `0..24` to A and `25..47` plus output to B by default. Reuse the existing layer ownership and shared KV/GDN allocations. Do not clone model state per frame.

The current `llama_context::decode_internal` owns one mutable `llama_memory_context_i` cursor. `init_batch()` precomputes the ubatches and KV slot-info vectors, but `process_ubatch()` calls `mctx->apply()` for the current cursor. `llama_memory_hybrid_idx_context::apply()` applies attention and recurrent state and advances the QSA bookkeeping; `next()` advances both child cursors. `llama_kv_cache_context::apply()` calls `kv->apply_ubatch()` and snapshots `n_kv`; `llama_memory_recurrent_context::apply()` calls `find_slot()`. Graph input objects retain pointers to the current memory context: the recurrent input reads `n_rs`, `s_copy`, `head`, and `rs_z`; the QSA input reads the indexer and pooled-cache plans. Reusing that mutable cursor for two queued frame graphs would let the later `apply()` change the metadata used by the earlier graph.

Before integration, add a target-frame memory snapshot that copies the selected ubatch, KV/indexer `slot_info`, `n_kv`, K/V write indices and attention masks, recurrent head/`rs_z`/`s_copy`, and any QSA per-frame selectors. It must retain non-owning views of the existing single KV and recurrent state buffers. Build and populate each frame graph's inputs from this immutable snapshot before applying the next ubatch. Keep graph input tensors, graph result metadata, boundary activation, and scheduler scratch owned by the frame slot.

The stage scheduler must expose the existing per-backend graph splits instead of executing all splits in one `ggml_backend_sched_compute_splits()` call. Run PLE/CPU preparation and island A for frame `k`; record A's completion event. Island B waits for that event, receives the boundary activation, then runs frame `k`. Queue A for `k+1` after A's prior completion, independently of B for `k`. Queue B for `k+1` after both B's prior completion and A's matching activation event. A and B each own their existing layers, so their KV/GDN buffer regions are disjoint; per-island event ordering preserves same-layer position order across frames.

Start behind a default-off switch and limit eligibility to ordinary Qwen4Exp prompt ubatches with one sequence, equal fixed-size interior chunks, `n_outputs == 0`, no slot reuse, and no MTP/rollback activity. Drain both stage events before the output-producing final ubatch, sampling/logit readback, request cancellation, sequence removal, memory update, or graph/frame reset. On any stage error, drain already queued work before rollback. Keep pooled QSA mutation disabled in the first implementation unless its commit state is also frame-snapshotted.

The first full-model gate is the existing clean semantic route, with the same prompt split serially and through two frames. Compare output and the final KV/GDN state against serial execution before measuring throughput. The low-level fixture cannot establish these model-level conditions by itself.
