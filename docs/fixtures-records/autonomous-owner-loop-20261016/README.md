# V21 → V23 autonomous bank-owner loop staging

Status: fixture-only design/CPU simulation and transformation proposal. No core source was modified and no GPU/model run was made.

## V21 bottleneck traced in source

V21 already gives each `bank` one long-lived `dual_context_owner` worker (`fixtures/dual-context-v21-prep-20261016/dual-context-v15-owner.h`). The target phase is submitted to that owner in `bank::submit_async()` (`dual-context-v21.cpp:655`); the owner executes `run_target_frame()` (`:538`), including draft submission, target verify, and `llama_synchronize(target)`.

The future then becomes ready. The only existing ready loop (`dual-context-v21-prep-20261016/dual-context-v14-ready-loop.h:14-45`) runs on the main coordinator: `complete_collect(bank)` is called synchronously, and only after it returns does the loop resubmit that bank. In V21 the callback at `dual-context-v21.cpp:1126-1135` performs `join_target_frame()` and `collect()` before `submit_async()`.

`collect()` (`dual-context-v21.cpp:682`) runs `common_speculative_process`, greedy target selection, `common_speculative_accept`, per-lane prompt/generated/timestamp updates, and target/draft KV rollback via `llama_memory_seq_rm`. If this host work blocks on one bank, the main loop cannot service another bank that has already finished its target frame, so that bank's owner is idle until the coordinator returns. V21's ready loop is bank-aware but the **collection work itself is still main-thread serialized**.

## V23 proposal

Use the existing one-worker-per-bank queue for one long task per active bank per parallel wave:

```text
bank owner: while any lane active:
    run_target_frame()
    mark target frame complete
    collect()      # same bank owner: process, greedy, accept, rollback, timestamps
    immediately start next frame
main coordinator:
    submit every active bank owner loop
    join every submitted owner loop
    if any owner failed, drain all banks and propagate first failure
```

The candidate keeps `run_target_frame()` and `collect()` byte-for-byte unchanged. It adds a self-driving wrapper that calls those existing methods on the same owner thread, plus a `join_all_owner_loops()` helper that waits every bank future before rethrowing the first exception. `drain_owner_contexts()` is separate from `drain_after_failure()` so an owner task never calls `future.get()` on its own future. V21 serialized warmup, prefill tasks, lane state fields, `common_speculative_process` guard, target synchronization, per-lane accepted counts/timestamps/limits, and target/draft rollback behavior stay in place.

The main loop still joins all banks at the end of the wave. It has no callback between per-bank frames. Each bank has one active owner task, so no two threads access its target, draft, speculative state, batch, or lane state concurrently.

## CPU simulation gates

`tests/autonomous-owner-loop-test.cpp` covers:

- **No per-step bank barrier:** bank A deliberately blocks in its first collect; bank B must submit its second target frame before A is released.
- **Thread ownership:** every phase for a bank stays on one thread, and banks use different threads.
- **Lane accounting:** a mock target/draft sequence checks accepted counts, prompt rollback positions, monotonic emission timestamps, and per-lane generation limits.
- **Failure drain:** one owner throws after a partial wave; the helper drains the failing bank, joins the healthy bank to completion, and the coordinator drain is invoked for every bank before propagating the failure.

Run `bash run-cpu-tests.sh` from this fixture directory. It regenerates `proposal.patch`, runs the CPU simulation, checks source invariants, and confirms the patch applies to the captured V21 source and reconstructs the V23 candidate.

On AC922, `bash run-syntax-v23.sh` runs a syntax-only parse of the staged runner against V21 fixture dependencies and current source headers. It does not link or run the engine and does not initialize CUDA.

## Transformation artifact and ownership risks

- `base/dual-context-v21.cpp` is a read-only copy of V21, SHA-256 `e529f45f3ee4716badfdc57a8fb7515efa57f8dbe93e8e298d0e3e4247e7aa0e`.
- `proposal.patch` adds `autonomous-owner-loop-v23.h` and transforms the captured V21 fixture source into `candidate/dual-context-v23.cpp`. It is a historical V21-based artifact, not applied to the live runner.
- V22 remains Turbo-owned and is only read as a source snapshot here; no V22 runner or core file is modified.
- The test only proves scheduling semantics in a CPU model. Before integration, retain the real V21 semantic gate and check V23 accepted tokens, per-lane emission times, generated limits, per-sequence rollback, draft hidden-state updates, failure drains, and thread IDs on V22/V23. Do not infer model correctness or throughput from the CPU simulation.
- `common_speculative_process()` and `common_speculative_accept()` move from the coordinator to the same per-bank owner already used for draft and target submission. That is intended to preserve bank-local context/TLS affinity and eliminate coordinator work from the other bank's submission path. Check any hidden main-thread-only hooks, CUDA TLS assumptions, or process-global sampling state during integration. The existing `dual_context_submit_guard` remains around its current operations, so an explicitly enabled global CPU submission lock can still serialize those guarded calls.
- `dual_context_owner` only accepts one queued task while busy. V23 submits exactly one complete loop per bank, and the coordinator joins it before reset or destruction. The owner loop itself queues no nested task.
- On a frame/collect exception, the owner drains its target and draft contexts without joining its own future. The coordinator then waits every bank future and performs the outer failure drain. A healthy bank is allowed to finish its bounded lane wave before the first failure is propagated; a cancellation protocol is intentionally not introduced in this patch.

## Potential payoff

V23 removes host-coordinator gaps between a bank's target completion and its own MTP processing/next frame. It does not reduce GPU work, target verify rows, draft depth, or cross-bank submit-guard contention. The gain is limited to periods where coordinator-side `collect()` delays a bank whose owner is ready to submit. Measure with the same V21 scenario and retain per-bank target/collect timestamps; no speedup is claimed here.

## V22 integration transform (host-ready snapshot)

The Turbo-owned V22 runner passed its host-only tests/link before this snapshot was taken. This fixture contains a separate, read-only copy at `base/dual-context-v22.cpp` and a V22-derived candidate at `candidate-v22/dual-context-v23.cpp`. The applyable `proposal-v22-to-v23.patch` replaces only the parallel decode wave with one self-driving task per active bank; it preserves the V22 diagnostic/clean context-family sequence, preclean semantic/F32 gate, and existing scenario/output layout. The frame and collection logic are unchanged; one stale coordinator comment in `run_target_frame()` is corrected.

Run `bash run-v22-transform-check.sh` to regenerate the candidate and patch, verify source invariants, and apply the patch to a temporary V22 copy. This check does not build/link or run the engine. Root owns V23 hostlink and runtime validation. V23 uses the V22 CLI and run-script environment, with no new owner-loop environment switch.
