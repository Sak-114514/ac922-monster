# P7: second outputless frame freeze on recurrent `rs_z=-1`

Date: 2026-10-15  
Scope: private memory snapshot code and an independent CPU-only real-metadata fixture. No shared source tree, running service, GPU, model weights, or system settings were changed.

## Cause

The V18 server log at `standard-semantic-20261014/results/20261015T104634Z/server.log` shows the first 3072-token outputless `shape-warmup` frame completing, then the second 3072-token frame applying live memory successfully and failing before graph build with `freeze failed after live apply`.

The CPU-only reproduction isolates the returning `nullptr` to the recurrent snapshot, while indexer and attention snapshots still pass. On frame 1, the real metadata is `n_rs=1, head=0, rs_z=-1`. `llama_memory_recurrent_context::freeze_current()` rejected any negative `rs_z` when `n_rs` was nonzero, and the single-assignment helper `llama_memory_frame_recurrent_state::capture()` repeated the same check.

`rs_z == -1` is a valid sentinel when the current recurrent range has no spare zero-state cell. `llm_graph_context::build_rs()` already handles it by constructing a zero-sized no-op `state_zero` view; it does not require `rs_z >= 0`. The two snapshot guards were stricter than the graph contract.

## Minimal fix

The private memory guards now accept exactly `-1` as that sentinel and still reject values below `-1` or nonnegative indices outside the cache. Bounds on `head`, `n_rs`, `head + n_rs`, ubatch ownership, cache cells, and single-capture `s_copy` behavior remain unchanged.

Changed private source files:

- `src/src/llama-memory-recurrent.cpp`
- `src/src/llama-memory.cpp`

## CPU-only regression

The independent fixture is `fixtures/memory-frame-multiubatch-20261015/`. It loads the actual Qwen GGUF with `vocab_only=1, no_alloc=0`; the loader reports `vocab only - skipping tensors`. Attention, recurrent, and indexer cache layer filters are all false, so the real cache-cell and slot metadata code runs without allocating layer tensors. `CUDA_VISIBLE_DEVICES=''` forces CPU-only execution; CUDA initialization reports no visible device.

The fixture prepares a single sequence as two consecutive 3072-token outputless ubatches. For every arm, frame 0 uses the full hybrid-indexed freeze. Frame 1 independently checks the indexer, attention, recurrent, and full hybrid-indexed freeze paths. The full path also freezes the first frame, advances and applies the live context, and then compares the first snapshot's K-cell indices, 3072×3072 KQ mask, copied ubatch arrays, QSA cell metadata and position-to-token samples, recurrent cursor values, and captured `s_copy` values.

Before fix, using runtime V19:

- Frame 0 full freeze passed.
- Frame 1 indexer and attention freezes passed.
- Frame 1 recurrent freeze returned null; the full hybrid-indexed freeze also returned null.
- Both frame-1 lines reported `n_rs=1, head=0, rs_z=-1`.

After fix, using root-built runtime V20 (`20261015T110206.473788Z`, 2 C++ sources / 0 CUDA sources):

- Indexer, attention, recurrent, and full hybrid-indexed freezes all passed on both frames.
- Frame 1 with `rs_z=-1` passed the complete freeze path.
- After the live context advanced and applied frame 1, the frame-0 snapshot still matched exactly for mask, attention/indexer indices, ubatch, QSA cell view, recurrent cursor and `s_copy`.
- The helper accepted `rs_z=-1`, rejected `rs_z=-2`, and retained its one-shot `s_copy` value.
- Final log: `PASS multiubatch_freeze_cpu_only`.

The failure and success logs are preserved separately:

- `fixtures/memory-frame-multiubatch-20261015/cpu-repro-pre-fix-v19.log`
- `fixtures/memory-frame-multiubatch-20261015/cpu-repro-post-fix-v20.log`

SHA-256:

- Pre-fix log: `e5f9e207ca7eed56e2f09dc711f8ea52249ffd8eea3d32280cf93de2033ec163`
- Post-fix log: `5e6550db26beb15cf7da846955f697b5661c13f444a148e44fb6b3f32544ac3f`

## Remaining scope

This validates CPU metadata ownership and the exact freeze failure path. No CUDA graph, GPU cache tensor, full-model inference, or production throughput check was run. The memory source edit is now handed back to root for integration decisions.
