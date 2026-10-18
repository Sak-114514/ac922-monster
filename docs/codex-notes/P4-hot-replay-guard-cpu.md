# P4 selective Meta replay guard: CPU fixture evidence

Date: 2026-10-15
Status: bounded CPU fixture passed; no engine runtime build or model/GPU run.

## What the fixture covers

The fixture builds the changed production Meta backend and scheduler translation units directly, links them to the frozen CPU/base libraries, and creates two distinct Meta backend contexts over the same two CPU device entries. Both contexts use the same static WEIGHTS Meta buffer. It runs:

- cold A graph, cold B graph, then exact A/B/A reuse;
- a graph switch to a shorter view on B's existing scheduler;
- a barrier-controlled attempt to replay hot A while B holds the process guard through graph allocation and dispatch;
- scheduler reset invalidation and a second B rebuild.

The root-graph gate must deny the old graph immediately after B switches/reset, deny the new graph before its first Meta dispatch, and permit it only after the exact scheduler split is dispatched and ready. The test compares scalar results and checks shared weight-buffer cache/generation fields around a hot replay.

## Result

Command:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/meta-replay-guard-20261015
./build-fixture.sh
OMP_NUM_THREADS=2 GGML_META_REPLAY_GUARD=1 GGML_META_REPLAY_GUARD_TRACE=1 timeout 30s ./meta-replay-guard-fixture
```

Exit status: 0.

Final result line from fixtures/meta-replay-guard-20261015/run-008.log:

```
Meta selective replay CPU fixture: PASS (A/B/A exact=1, A hot vs B rebuild contention=1, readyA=1, readyB4=1, reset_denied=1, static_descriptor_stable=1, hot_generation_invariant=1, hot_epoch=13->13, hot_split_entries=3->3, hot_simple_queries=14->15)
```

The initial diagnostic found the root-vs-split UID mismatch: the original graph had UID 1 while Meta dispatched split UID 2 and cached context UID 2. The strict Meta graph query on the original graph is therefore false; the scheduler gate succeeds by binding the original graph pointer+UID to the exact current allocated split graphs, then applying strict readiness to each split. This is why the implementation does not use a cached-ready bit alone.

The original graph UID 1 is not a scheduler Meta split UID; strict readiness on the original graph is false while the scheduler-bound gate is true for its actual ready split. The fixture also confirmed that switching/resetting the scheduler denies the old graph and that the shared static descriptor pointer/data remain unchanged after rebuilding B.

The scheduler emitted its normal "failed to allocate graph, reserving" diagnostic for the tiny fixture, recovered through its reserve fallback, and the test completed successfully.

## Runtime boundary

`ldd` for the fixture resolves only `libggml-cpu.so.0` and `libggml-base.so.0` from `build-reuse/bin`, with no CUDA library. `ggml-backend-meta.cpp` and `ggml-backend.cpp` are linked as the fixture's own objects with `-rdynamic`, so callbacks and the new private scheduler gate resolve to the experiment sources. This is CPU Meta ownership evidence only; it is not a CUDA, full llama_context, multi-request server, or performance result.

`llama-context.cpp` syntax-only compilation passed after switching its reused-graph predicate to the scheduler-bound API. The syntax-only command and its zero-diagnostic exit are recorded in notes/P4-llama-context-syntax.log.

The user-facing runner guard has not been removed here. The root owns the next actual 3+3 GPU gate and integration decision.
