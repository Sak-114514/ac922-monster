# V47 pooled-QSA / Turbo phase-ledger staging

Date: 2026-10-16

This is a fixture-only rebase against the current private V47 source. No core source was
modified, no engine was built, and no GPU program was run.

## Base and artifact

Source root: `/home/debian/ac922-monster/experiments/standard-opt-20261014/src`

V47 source hashes captured before applying the patch:

| File | SHA-256 |
|---|---|
| `src/llama-context.cpp` | `b5d5577fa12e8c80a272c3e7df6eacfa894bf2a5fdc82742aaf8ce5709d5f8e1` |
| `src/llama-memory-hybrid-idx.cpp` | `8a09c00826ca7e749e8b13607a822a146a46a23f20fd90c144450ea415d6b7f6` |
| `src/llama-memory-hybrid-idx.h` | `101bc4ba17ffde3d7e8e855142ec7b837c7909e2f9fa4a6cbe69d161f01266e6` |
| `src/models/qwen4exp.cpp` | `67adf511d7490a6a1f278a56c0d7fbfe813c5dce2d18e8bd703b86b3ad03b89f` |

Patch: `qsa-pool-turbo-phase-v47.patch` (768 lines, 5 files; 329 additions / 74 deletions).

## Connected behavior

- The staged physical-ID CELLSEL graph is fully connected to host inputs: query cell destinations
  and tail bias are built by the production planner, uploaded only on CELLSEL graphs, and shape
  checked on graph reuse. The selector F32 base is initialized with `ggml_fill(-INFINITY)` before
  `ggml_set_rows`; only the physical `[0,n_kv)` view reaches stable `TOP_K`. The gate requires
  `indexer_top_k % ratio == 0`, `GGML_CUDA_TOPK_STABLE=1`, and canonical ordering disabled.
- Turbo frame admission preserves the previous memory gate by default. `LLAMA_QSA_TURBO_MULTI_PHASE`
  must equal the literal `1`, and existing `LLAMA_QSA_POOLED_MULTI` must have constructed a
  mutable per-sequence pool. Single/global pooled mode remains rejected.
- A frozen frame continues to use the ordinary selector over its immutable cell snapshot; it does
  not read or update mutable pooled rows. After the entire snapshot is allocated, only that frame's
  sequence ledger is invalidated. A later mutable slow graph rescans the cells and leaves all pool
  rows dirty; its first eligible mblk graph schedules the refresh before scoring.
- Successful scheduler submission commits only pending pool ranges in the graph-success path.
  Failed submission invalidates each sequence in that ubatch, cancelling pending rows and leaving
  unrelated sequence ledgers unchanged. The old unconditional commit from the next `apply()` was
  removed.
- With all new variables unset, the prior standard route and prior Turbo/no-pool gate remain
  unchanged. `LLAMA_QSA_MULTI_SPARSE` remains off.

## Verification

- Patch dry-run and apply against the clean V47 staging copy: passed for all five files. Logs:
  `patch-dry-run-v47-r2.log`, `patch-apply-v47-r2.log`.
- CPU contract runner: passed. It covers default-off/literal opt-in, single-vs-multi frame gate,
  graph-success commit vs abort, freeze sequence-local invalidation, 4→1→4 reuse, foreign-ledger
  isolation, rollback boundary, holes, physical-ID ties, tail bias, and causal candidate mapping.
  Output: `cpu-test-output-v47.txt`.
- PowerPC64 syntax-only C++17 compile of the patched `llama-context.cpp`,
  `llama-memory-hybrid-idx.cpp`, and `qwen4exp.cpp`: all passed. Log:
  `syntax-only-v47-final.log`. Existing signedness warnings remain in unrelated context/MTP and
  qwen gather-limit comparisons; no compile errors.

## Qualification boundary

The host CELLSEL contract injects identical per-block scores. It proves physical-ID scatter, unique
scratch destinations, tail priority, stable tie order, and fallback rules; it does not establish that
the pooled mean/RMS-normalized score equals the ordinary per-cell indexer score. The latter is not
generally mathematically identical and needs the smallest real-model GPU gate: compare selected
physical IDs, masks, then full logits/semantic outputs against the clean unpooled selector with
`LLAMA_QSA_MULTI_SPARSE=0`. Any ID drift must be reviewed as a quality change, not accepted from
the host fixture.

No full engine build, GPU replay, model logit parity, or performance claim is included here.
