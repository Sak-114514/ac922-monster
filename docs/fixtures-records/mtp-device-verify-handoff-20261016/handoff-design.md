# MTP target-to-draft hidden handoff

Status: source-grounded design and CPU row/lifetime contract only. No core files
were changed and no GPU/model work was run.

The source audit used the AC922 tree; SHA-256s for the core files match the
local mirror: `common/speculative.cpp`
`be7bbf090e88160c7d444cc44089dc2fe0e64959311676cd9c0077164a6e4abc`,
`src/llama-context.cpp`
`a402784401300f4e333205e0ab592804e8073f1d2d718017f04391d2a1e05f57`,
`src/llama-context.h`
`575bf93c2aae5e91f8de6a22178453b8f90a14cd9c54ea31324c5b5f1ce21d78`, and
`src/llama-ext.h`
`80221e526f6e51f75708267c547f464a25d43d8d9834a43bcbcc37cc97463ca9`.
The actual-source-only `common/speculative-mtp-layout.h` is
`89cbf61dc47c9288f57126fe323aa1ff6bfe1d0a9cb8da43e1f0c89bd79fb42d`.

## Finding

The round trip is present in the current path. `llama_context::process_ubatch()`
gets the target graph's `t_h_nextn` and queues its rows into the CPU-side
`embd_nextn` buffer. `common_speculative_impl_draft_mtp::process()` calls
`llama_get_embeddings_nextn(ctx_tgt)`, shifts rows on the CPU, and stores the
folded verify rows in `seq_deferred.hidden`. Later `draft()` copies those rows
into `batch.embd`; the draft context uploads them again.

The existing per-sequence GPU carry is a useful graph seam, but it cannot
represent these inputs as-is. It allocates `mtp_carry_ps_hidden` as one F32 row
per sequence. Its eligibility gate rejects duplicate sequence IDs and batches
with more input rows than sequences. Folded catch-up commonly has several rows
for one sequence. The graph already has `ggml_get_rows()` and
`ggml_set_rows()` for the hidden input and feedback, so row selection itself
does not need a new CUDA kernel.

**Conclusion:** a GPU-resident handoff is feasible for a bounded, contiguous
target output on a compatible peer route. It needs a narrow private staging
seam and a row table sized for catch-up inputs; simply turning on
`LLAMA_MTP_GPU_CHAIN_MULTI` or reusing the one-row carry does not remove the
target-to-draft host round trip. The existing target getter is host-only and
synchronizes. There is no current private API that hands `common/speculative`
the target `t_h_nextn` tensor/backend or stages it into draft-owned storage.
The existing Meta cross-backend path can also bounce through pinned host
memory when peer fanout is unavailable, so an async API return alone is not a
device-residency proof.

## Source flow and row contract

The relevant production path is:

1. `llama_context::process_ubatch()` obtains `t_h_nextn` and queues a device to
   host read into `embd_nextn` for unmasked target outputs.
2. `llama_get_embeddings_nextn()` synchronizes before returning the host
   `embd_nextn` pointer. The per-sequence layout helper keeps global batch row
   IDs in `layout.rows[seq]` and maps every global row to the previous row of
   the same sequence in `layout.previous`.
3. `common_speculative_mtp_shift_rows()` defines the exact hidden pairing:
   `shifted[g] = pending_h[seq]` for the first row of a sequence; otherwise
   `shifted[g] = target_h[layout.previous[g]]`.
4. On a target verify batch, `seq_deferred.keep` becomes
   `min(n_accepted + 1, sequence_row_count)`. The selected next seed is target
   hidden row `layout.rows[seq][min(n_accepted, row_count - 1)]`.

For a sequence with global verify rows `[g0, g1, ...]`, the device row plan is:

- Catch-up input `j == 0`: the previous accepted target hidden row, preserved
  in that sequence's reserved carry slot.
- Catch-up input `j > 0`: target hidden row `layout.previous[g_j]`.
- Next seed after acceptance: target hidden row
  `g[min(n_accepted, row_count - 1)]`.
- Number of deferred catch-up rows: `min(n_accepted + 1, row_count)`.

These formulas preserve the current FP32 bytes and sequence-local order. The
CPU fixture below checks them against the existing `common_speculative_mtp_*`
helpers.

## Small patch proposal (not applied)

Keep this private to the existing MTP extension surface; do not add a public
llama API or a kernel.

1. **Capture target output validity in `llama-context.cpp`.** At the point where
   `t_h_nextn` and `backend_h` are available, retain an ephemeral output view
   plus a monotonically increasing output generation for the current target
   decode. The current unmasked target branch queues D2H immediately, so the
   opt-in route must defer that transfer until the staging helper reports its
   route. If the helper declines, queue/flush the exact existing CPU readback
   before using the legacy path. The view is valid only until the next target
   graph reuse, so the caller must stage it before another target decode.

2. **Allocate draft-owned row storage.** In the existing per-sequence carry
   setup, add a bounded input table of shape
   `[n_embd_out, n_ubatch + n_seq_max]` plus its I32 row map. Reserve the first
   `n_seq_max` rows for each sequence's last accepted seed and the remaining
   rows for the current target verify output. Keep draft feedback in the
   existing `[n_embd_out, n_seq_max]` table, separate from this input table.
   On accept, copy the selected target row into the sequence's reserved slot;
   this keeps the prior seed alive when the next target output is staged. The
   initial step then maps catch-up row 0 to the reserved slot and later rows to
   their `layout.previous` source rows. Buffers belong to each `llama_context`;
   do not use a static or process-wide bank.

3. **Stage before target output reuse.** Add one private helper that copies the
   current target output view into the draft-owned table. Qwen4Exp exports the
   wide hidden as a contiguous `[n_embd, hc, rows]` tensor but reads it as
   `[n_embd_out, rows]`; stage through a same-byte 2D view, with no arithmetic.
   The Meta async mirror-copy seam accepts contiguous, same-layout, mirrored
   Meta tensors when `GGML_META_ASYNC_COPY=1`. Its CUDA copy implementation
   uses peer GPU staging only when `GGML_CUDA_META_COPY_TOPOLOGY=1` and peer
   fanout is available. Otherwise it uses a pinned-host staging area. The
   generic `ggml_backend_tensor_copy_async()` is `void` and synchronizes then
   falls back when the async callback declines, so the proposed handoff must
   use a route-confirmed private helper or explicit eligibility result. Only
   the peer-fanout result counts as GPU-resident; the other cases select the
   current CPU route. A shape/view or unsupported topology also declines to
   CPU.

4. **Store row references in `common/speculative.cpp`.** In the handoff branch,
   replace the F32 payloads in `seq_deferred.hidden` and `verify_h` with the
   current target generation plus global row IDs. `process()` builds those IDs
   from `layout.previous`; `accept()` applies the existing
   `min(n_accepted, n_rows - 1)` and `min(n_accepted + 1, rows)` formulas, then
   asks the draft context to copy the selected seed into its reserved per-seq
   slot. Keep token/position arrays on CPU. This avoids serializing hidden rows
   as floats while preserving the exact sequence order.

5. **Pass per-input row IDs to the draft graph.** Extend the current private
   handoff/carry state so each draft input row maps to a staged table row. Use
   `ggml_get_rows()` for inputs and retain the current per-sequence
   `ggml_set_rows()` feedback table for the draft's next hidden state. The
   current `llm_graph_input_embd_h::set_input()` generates a row map from
   `seq_id`; that behavior stays intact for existing per-sequence carry. The
   handoff mode supplies a checked, per-input-row map instead, and uploads the
   CPU row only when its stage slot is not marked ready.

6. **Keep `GPU_CHAIN_MULTI` compatible at step zero.** Its current graph path
   selects the chain slot or host input and disables generic per-sequence
   carry while the chain is active. The first catch-up/seed graph therefore
   needs the same staged row map; later draft steps continue using the existing
   lane hidden slots. The unrolled graph builder currently clears row-map
   parameters for every step, so only step zero may accept the handoff map and
   all later steps must retain the existing previous-step hidden input.

7. **Commit, fallback, and reset rules.** On a route decline, split ubatches,
   row-map validation failure, or active capture without a proven copy
   dependency, materialize `t_h_nextn` through the current CPU getter and run
   the existing path. If the staged draft decode itself fails, keep the stage
   table untouched, invalidate its handles, and materialize those rows only
   for the error fallback/rollback path. Do not consume a row past `keep_rows`.
   A fresh position-zero sequence clears its reserved seed and deferred
   generation. Drain any recorded copy before freeing either context's storage.
   Snapshot serialization keeps CPU metadata/values only; never serialize a
   device pointer or generation handle. If a snapshot is requested while rows
   exist only in a device generation, either materialize those rows for the
   snapshot or decline that snapshot.

The narrow backend route report is a required seam: `ggml_backend_meta_cpy_tensor_async`
returns a `bool` internally, but the public generic async-copy wrapper hides
whether it selected that callback, and the CUDA copy context can choose either
GPU peer staging or pinned-host staging. Do not mark a target row GPU-resident
from a generic copy API return alone. The first implementation should require
the frozen source/draft topology to report the peer-fanout route; otherwise it
must take the already-correct CPU path. The opt-in route should publish one
transactional result such as `peer_device_copy`, `host_staged`, or `declined`;
only `peer_device_copy` may suppress the existing CPU readback.

### Acceptance and exceptional cases

- `n_accepted == 0` keeps only catch-up row 0, and the next seed uses target
  row 0. `is_other` still follows the existing `accept()` behavior; it does not
  authorize a stale or out-of-range row.
- If a sequence has no output rows, `accept()` leaves its current carry alone;
  the handoff must mark that sequence unavailable for this target generation.
- EOS/suppression or a non-drafting lane must clear rejected row IDs and must
  not write another lane's slot.
- Ragged sequences use `layout.rows[seq]` and global source rows from
  `layout.previous`, never a token-count stride or `seq_id` as a source row.
- A failed draft/chain attempt must invalidate staged handles before rollback
  and preserve the existing CPU batch snapshot until the fallback/rollback
  decision is complete.
- The target source view must be staged before the next target graph can reuse
  its output storage. If the backend route cannot provide a completion fence,
  synchronize before returning from the handoff helper or decline to CPU.
- Two independent owners use two independent staging buffers, row maps,
  events, and generations. Interleaving owner A/B/A must not allow a row handle
  from one owner or generation to resolve against another.
- Graph capture/replay needs separate proof: keep the staging buffer pointer
  stable, update row-map contents before replay, and keep target-to-draft copy
  outside capture until the exact backend route is confirmed capture-safe.

## Verification prepared

`row-layout-lifetime-test.cpp` uses the actual
`common/speculative-mtp-layout.h` helper. It exercises interleaved/ragged row
IDs, accepted-prefix clamping, `n_accepted == 0`, `is_other`, no-row lanes,
source overwrite after staging, two independent owners, CPU fallback on
unsupported/split/capture cases, and generation invalidation on new target,
reset, rollback, and teardown.

Run on AC922 without GPU access:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
bash fixtures/mtp-device-verify-handoff-20261016/run-cpu-only.sh
```

The fixture is a row-layout/lifetime contract only. It does not prove backend
copy dispatch, CUDA capture safety, target/draft graph equivalence, or a model
result. Those require a later bounded runtime test after the private handoff
seam is implemented.
