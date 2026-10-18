# V30b CPU-only target/MTP row trace

## Observed failure

Run `root-p4-v30b-onebank4d3-hierarchical-ep6-shortctx16k-20261016t041327z`
used V71 with `GGML_META_FRAGMENT_PARENT_SIG=1`, one bank, four 16K streams,
D3, verify capacity 16. Preclean F32 comparison passed (40/40 serial/P4 pairs;
42/42 same-flow before/after-MTP pairs), and native semantic checks passed.
The generation token gate did not pass: V30b's own serial outputs match the
V27 serial/P4 outputs for `real8k-0` and `real8k-1`, while both V30b diagnostic
and clean P4 outputs first diverge from V30b serial at token indices 2 and 13.
Flows 2 and 3 match. The original cold/warm 32-token prefix checks also fail.
The first-80 token IDs and full output texts are in
`results/root-p4-v30b-onebank4d3-hierarchical-ep6-shortctx16k-20261016t041327z/v30b-real8k-token-review.json`.

This proves an exact serial/P4 generation mismatch in the four-lane candidate.
It does not identify whether target recurrent computation or MTP hidden-state
selection/rollback first causes it. No source invariant violation was observed
in the reviewed row mapping, and the saved full-model log does not contain a
per-lane row-index trace.

## Row and state mapping in the current source

- The fixture appends each lane's current token then its draft tokens. It saves
  `batch.n_tokens` into that lane's `rows` vector immediately before each
  `common_batch_add`; after target `llama_decode` and synchronization,
  `collect()` reads logits from those saved batch-row IDs. It computes
  `accepted = outputs.size() - 1`, passes that count to
  `common_speculative_accept`, then rewinds target and draft memory to the
  accepted prefix. See `dual-context-v30-onebank4d3.cpp:638-648` and
  `:777-804`.
- With multi-chain D3 and four independent sequences, MTP enables per-sequence
  layout. `common_speculative_mtp_make_layout` groups original batch-row IDs by
  `seq_id`; `process()` stores target `h_nextn` rows in that order. `accept()`
  selects row `min(n_accepted, n_rows - 1)` for the next `pending_h`. See
  `common/speculative.cpp:1571-1584`, `:1786-1798`, `:1987-1999`, and
  `:2722-2733`, plus `common/speculative-mtp-layout.h:289-336`.
- The target context requests unmasked next-token embeddings. When recurrent
  batch splitting reorders execution rows, the context records original
  `row_ids` and restores them before `llama_get_embeddings_nextn_ith` returns
  raw batch-row `i`. Target logits use `output_ids` to translate that same raw
  batch index. See `src/llama-context.cpp:1631-1670`, `:2291-2325`,
  `:3740-3747`, and `:4275-4327`.
- Recurrent memory calls `split_equal(..., sequential=true,
  n_rs_seq + 1)` to keep rollback history together; `find_slot()` asserts
  equal row counts and addresses each sequence block at `s * n_seq_tokens`.
  `qwen4exp` GDN asserts the same shape and reshapes its inputs/states as
  `[n_seq_tokens, n_seqs]`. Its convolution history writes separate rows for
  each rollback slot. See `src/llama-memory-recurrent.cpp:453-490`, `:530-545`,
  `:680-700`, `:1353-1369`; `src/llama-batch.cpp:520-688`, `:759-815`; and
  `src/models/qwen4exp.cpp:3031-3035`, `:3062-3076`, `:3307-3352`.
- The target context reported `n_rs_seq=4` and the recurrent cache reported
  four sequence cells/four rollback snapshots. The full-model run's clean
  `frame-submit` row-count histogram was `{16:67, 12:2, 8:6, 7:1, 5:1, 4:6,
  2:1}`; the diagnostic histogram was `{16:79, 12:6, 8:7, 7:1, 5:1, 4:10,
  2:1}`. Because only total rows and `max_draft_rows` were logged, those
  records do not reveal each lane's exact row count or the recurrent
  sub-ubatch partition.

## Follow-up full-trunk localization (only if the operator probe is exact)

If the isolated CUDA operator probe below passes, prepare a fixture-only target
probe with no speculative draft/accept loop and no core edits. Load the model
once, create two target contexts with the V30b
four-sequence non-unified recurrent layout, Q8 KV, `n_rs_seq=4`, and unmasked
`h_nextn` enabled. Prefill the same short real prompt prefix into each of four
sequence IDs in both contexts. For each sequence, choose the same four real
token IDs and positions in both arms:

1. **Serial arm:** decode each sequence's four verify rows separately.
2. **Batched arm:** append the same rows in the fixture's lane-major order and
   decode all 16 rows in one `llama_decode` call.

Compare every F32 logits row and every `h_nextn` row exactly by `(seq_id,
position)`. Run one equal shape `[4,4,4,4]` and one ragged shape such as
`[4,4,2,2]` (12 total rows, observed in the full-model row-count histogram).
The recurrent allocator's `n_rs_seq+1` split behavior must be logged, together
with the original batch row ID and output row ID for each `(seq_id, position)`.

If logits or `h_nextn` differ, the target/GDN recurrent row path is sufficient
to reproduce the mismatch. If both are exact for both shapes, add one narrow
MTP step next and log candidate token rows, `verify_h` row IDs, accepted count,
rollback position, and selected hidden row; do not start another full-model run
from this artifact alone.

This target-level probe remains a design only; it was not compiled or executed.

## First diagnostic gate: GDN X4 batch versus serial

The V71 CUDA dispatch review does not show an X4-only-on-four-sequences branch.
In `ggml/src/ggml-cuda/gated_delta_net.cu:339-425`, the X4 kernel is selected
when the opt-in is nonzero, KDA is false, state width is 128, the device is
Volta/Turing, and `n_tokens > 1`. The predicate does not test `n_seqs`. Thus
T=4 batch (`n_seqs=4`) and four independent T=4 serial calls (`n_seqs=1`) use
the same X4 kernel on V100. This isolates sequence/state addressing and
snapshot scatter; it does not compare X4 against the generic CUDA kernel.

Static kernel addressing is sequence-local: `blockIdx.y` is the sequence;
the gathered initial-state address is
`state_rows[sequence] * state_row_stride + head * S_v * S_v`; and output
snapshots use `(sequence * H + head) * S_v * S_v` plus the caller's slot
stride. The X4 kernel groups four state columns within that sequence. I found
no static cross-sequence reduction or obvious `n_seqs=4` indexing violation.
The remaining useful check is execution against independent serial sequences,
including a non-monotonic initial-row map and the real cache-copy strides.

The fixture `gdn-x4-probe/gdn-x4-batch-serial.cpp` performs that comparison
with one batched F32 GDN op at width 128, four heads, T=4, four sequences and
four serial T=4 ops. It gathers initial states with IDs `18,6,13,1` from a
20-row state pool. V30b's runtime log reports recurrent `mem_size=4` and
`n_rs_seq=4`, so these IDs exercise the actual five-block
`mem_size * (1+n_rs_seq)` state pool. Its cache copy uses the same production view geometry as
`src/models/delta-net-base.cpp:562-592`: `[D,n_seqs,n_written]`, per-sequence
row stride, and rollback slot stride `mem_size * row_size` with
`mem_size=4`. It checks all
attention values, every written snapshot, whole-cache canaries, and the
accepted-length mapping `accepted=0..3 -> slot=3..0`. The GDN-to-cache fusion
match criteria are in `ggml/src/ggml-cuda/ggml-cuda.cu:4706-4755`; deferred
initial-state row gather eligibility is in `:5435-5485`.

CPU-only result (`runtime-perf-v71`, direct CPU backend, no CUDA backend
initialized) is a fixture-mapping pass:

- All four lanes: attention `2048/2048` bitwise exact and all rollback-state
  values `262144/262144` bitwise exact.
- All 16 accepted-length slot selections exact (`accepted=0..3`, slots `3..0`).
- Whole cache `1310720/1310720` exact, including untouched canary rows.
- Result log: `gdn-x4-probe/results/cpu-oracle-prodstride-20261016T044837Z.log`,
  SHA256 `f7d42ad50206875488ac7c3dcd625f8de3bda0866c1e5baae5b01ccab06408b4`.
- The prior eight-row-stride CPU result is retained but superseded; it is not a
  V30 layout proof.

One bounded V71 CUDA run on GPU0 passed this operator comparison. Actual
dispatch summaries reported `volta_gdn_x4 requested=5 selected=5` and
`gdn_state_gather requested=5 selected=5`; all attention, snapshots, all 20
cache rows/canaries, and rollback selections were bitwise exact. The log is
`gdn-x4-probe/results/gdn-x4-v71-gpu0-20261016T045900Z.log`, SHA256
`4bfcaae541986ab7c1eba836190f54ace698526f0e25e2ac7ae71e848b8bead1`. GPU0 was
confirmed released at 2 MiB used, 39 C, 0% utilization, with no process. This
means the isolated same-input GDN batch-vs-serial test does not reproduce the
V30b token mismatch; it does not rule out full-graph GDN input/row mapping,
Qwen convolution history, or other trunk/MTP behavior. For provenance, the V71 runtime's
`libggml-cuda.so.0` SHA256 is
`3b55d98d385d54dce5ba177050218eb7e26f15b3c424498e7cfc98bb7d44a6da`, and the
V71-source `gated_delta_net.cu` SHA256 is
`a50a0671dff2485ed7ef68d921136ee39f6901510b2f2967f6f5906652871295`. The
fixture C++ SHA256 is
`18c08e486027d96a7f464ac4b62a3f5a9c39e1eef6397d01235242b4becfdc5c`. This
probe validates only the recurrent GDN op and its SSM snapshot scatter; it
excludes Qwen's separate convolution-history cache, GDN input
projections/convolution, target logits, MTP acceptance, and full-trunk
semantics. The onebank4/D3 token mismatch remains unresolved.
