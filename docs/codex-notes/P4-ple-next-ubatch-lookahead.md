# P4 bounded next-ubatch PLE lookahead

Date: 2026-10-15. Private standard-opt source only. No direct edits to `llama-context.cpp` or `.h`; the context hook is prepared as a review/apply patch for the context owner/root.

## Implementation

Owned source changes:

- `src/src/llama-ple-prefetch-internal.h`
- `src/src/llama-ple-prefetch.cpp`
- `src/src/llama-ple-prefetch.h`

The existing PLE prefetcher remains default-off. Initialization returns null unless `LLAMA_PLE_PREFETCH=1` and `GGML_CPU_PLE_ROW_CACHE=1` are present and the real IQ4_NL PLE table matches its supported geometry. Normal GET_ROWS remains the source of truth.

A production job describes one physical text ubatch only. The helper gets `n_gram - 1` predecessors through the same KV `get_prev_tokens()` method used by qwen4exp, groups tokens by sequence ID, preserves lane token order, and requires contiguous positions. It skips shared-sequence rows, embeddings, noncontiguous positions, incomplete inputs, and oversize jobs instead of guessing or truncating. The first row in each lane carries the exact KV history (including token-null values; the Qwen hash path converts those to EOS); later lane history comes from earlier tokens in that ubatch.

The hard ceilings are 1,024 total tokens, 16,384 raw rows, at most 64 sequence lanes, and 128 MiB of unique-row aligned spans. Effective row/token limits are reduced to the acquired row-cache capacity at initialization (`max_rows <= row_capacity`, `max_tokens <= max_rows / n_heads`); with Qwen4Exp's 16 heads and capacity of at least 16,384 entries, the full 1,024-token bound is available. The row list is sorted/deduplicated only inside that one ubatch. Two job slots remain: one active and one pending. A full pending slot rejects the newer job rather than replacing the older next ubatch. Consumer epochs cancel jobs after the consumer has advanced; cancellation is checked before every 16-row `read_many` call. Existing row-cache `read_many()` is reused, so reads can merge nearby spans; no duplicate cache implementation or row-cache API change was made. The row-cache mutex is not held across I/O.

The 128 MiB figure bounds unique requested row spans assuming up to 8 KiB per 90-byte row. Concurrent demand and prefetch may race on the same cache miss and duplicate a direct read; the existing API has no per-job I/O counter/single-flight hook. The queue and row-count bounds remain strict, and stale work stops between batches of at most 16 rows.

## Context hook for root review

Patch file: `notes/P4-context-ple-lookahead.patch`.

It adds an optional prefetcher member/lifecycle to `llama_context`, a request-scope cancel guard, and two call points in the `mctx` ubatch loop:

1. Advance the prefetch consumer epoch before processing the current ubatch.
2. After a successful current ubatch, advance `mctx` once. If this is a non-MTP Qwen4Exp prompt chunk (`n_tokens_all > n_outputs_all`, token input, normal decoder context), submit exactly the now-current next ubatch with the hybrid attention KV context. The lookahead is canceled when the logical decode call exits.

The helper snapshots the next ubatch's history before it is applied and rejects unsupported row/lane shapes. The patch is not applied in this task because the context source is owned by the other agent/root.

## CPU fixture evidence

Fixture source and output:

- `fixtures/P4-ple-lookahead/ple-lookahead-rows-fixture.cpp`
- `fixtures/P4-ple-lookahead/fixture.log`

The fixture uses real Qwen4Exp PLE hash metadata and real prompt token IDs, then sends row IDs through an actual CPU backend scheduler `ggml_get_rows` using a tiny synthetic marker table. Every reconstructed marker matched the original PLE row ID. This validates hash/lane/chunk ordering and GET_ROWS index consumption; it does not load the model or read the real PLE weights.

Passed cases:

- 1-lane fresh prompt and next-chunk history.
- 2-lane and 4-lane interleaved physical ubatches with per-lane history.
- Maximum boundary for 1×1,024, 2×512, and 4×256 tokens: each produced 16,384 raw IDs and exact GET_ROWS output.
- 1,025-token ubatch rejected whole.
- Noncontiguous per-lane positions rejected.

The implementation translation unit also passed `g++ -fsyntax-only` using the private source/build include set. Only the standalone CPU fixture was compiled; the engine was not rebuilt. No model or GPU was used.

## Limitations / next integration gate

The feature is not live until root applies/reviews the context patch. The CPU fixture proves exact row planning and GET_ROWS index ordering with synthetic markers, not the real IQ4_NL model output or end-to-end disk/compute overlap. No model load, runtime benchmark, or GPU work was run.
