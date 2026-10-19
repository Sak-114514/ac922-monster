# Same-bank prefix checkpoint contract

This is a narrow same-bank cache entry, not a cross-slot prefix cache. One entry belongs to an existing bank and is invalid when that bank's context epoch or profile changes. Target and draft attention/QSA KV stay resident in the same contexts. The target stores only `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` recurrent state. The draft stores `LLAMA_STATE_SEQ_FLAGS_NONE` for its active sink64 + window2048 cells, plus one CPU MTP `pending_h` row. Draft state is bounded at64MiB and the aggregate checkpoint budget remains512MiB. No full target KV or device chain buffer is stored. Retaining later live draft KV alone is insufficient: subsequent window pruning deletes rows belonging to the earlier checkpoint.

## Stable capture boundary

For each selected prompt prefix ending at token count `P`:

1. Submit all target rows through `[0, P)` and finish the matching `common_speculative_process()` MTP hook.
2. Call `common_speculative_begin(spec, 0, prefix_tokens)` to clear fold/deferred proposal transients while retaining the final `pending_h` row.
3. Capture target `PARTIAL_ONLY` state and bounded draft `FLAGS_NONE` active-window state, then call `common_speculative_prefix_capture()` from `prefix-mtp-sidecar.patch`.
4. Record the exact token IDs `[0, P)`, current bank/context epoch and context-profile key. Require target and draft positions to agree at `P - 1`.
5. Continue prefill with the remaining request tokens. Keep one checkpoint per bank; account vector capacities against the 512 MiB total limit.

The frozen C12 worker's prompt loop currently calls `common_speculative_begin()` after prefill. The integrator must split the prompt at the chosen checkpoint boundary and make the capture at that point; do not capture a request-end state after generation.

## Restore boundary

1. After exact token LCP and epoch/profile checks, only hit if the request has at least one token after checkpoint position `P`.
2. Restore target recurrent state with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` and draft active-window full state with `LLAMA_STATE_SEQ_FLAGS_NONE`; trim target and draft sequence tails to `P`, including QSA indexer cells through the same context memory API.
3. Call `common_speculative_prefix_restore()`. It requires the same live single-lane C12 MTP context shape, an inactive GPU chain, matching target/draft positions, and a one-row pending hidden state. It clears only post-checkpoint MTP deferred/verify state and invalidates GPU carry reuse; next GPU_CHAIN draft starts from the restored host `pending_h` row.
4. Replay request tokens beginning at `P`, then call normal `common_speculative_begin()` before generation.

If exact LCP ends before `P`, target/draft contexts were repartitioned, profile/epoch changed, MTP state is dirty, or partial payload budget is exceeded, treat the entry as a cache miss and use the existing cold path. Do not move checkpoint state between banks.

## Source compatibility and proof boundary

The patch targets the read-only remote active source:

- `/home/debian/ac922-monster/experiments/standard-opt-20261014/src/common/speculative.h` SHA256 `8350f87a87d4007c54ddb4ff0378427215b15eb121654714746f997e37e5aee1`
- `/home/debian/ac922-monster/experiments/standard-opt-20261014/src/common/speculative.cpp` SHA256 `27660efcf0b4bed63c16d2290e4f6833e03fb59f726997be77d1b705d90c24ba`
- `/home/debian/ac922-monster/experiments/standard-opt-20261014/src/src/llama-memory-hybrid-idx.cpp` SHA256 `fb2020d969209e5fde40d0e9d7c88ed5c6ab724903cf56e66da42787b671ad12`

`common_speculative_impl_draft_mtp::snapshot_supported()` rejects the C12 one-lane fold/GPU_CHAIN mode; these helpers deliberately do not call the generic snapshot API. `llama_begin_mtp_gpu_chain()` resets chain count and carry reuse before the first proposal, and every C12 single-chain draft is guarded by `llama_abort_mtp_gpu_chain()` on exit. The pending hidden row is therefore the CPU seed to preserve; the GPU carry buffer remains transient and is invalidated before reuse.

This source audit proves the proposed interface predicates and cleanup sequence, not end-to-end numerical equivalence or performance. The added `test_prefix_checkpoint.cpp` covers pure policy: exact token LCP, same-bank/epoch/profile gates, clean-state gates, minimum replay token, and bounded memory accounting. A same-slot A/B/A semantic test and local `libllama-common` relink remain required before enabling the cache.

The two local worker declarations are:

```cpp
bool common_speculative_prefix_capture(common_speculative *, int, std::vector<float> &);
bool common_speculative_prefix_restore(common_speculative *, int, const std::vector<float> &);
```

Capture succeeds only after the worker called `common_speculative_begin()` at the selected prompt checkpoint. If `def_tok`/`def_pos` are empty, `def_keep` is unset and no draft is pending, the hook clears only the unowned residual `def_embd` storage and resets stale `def_seq`; source `begin()` clears the token/position bookkeeping but leaves these fields behind. Restore is called after recurrent payload restore and target/draft memory truncation to the same absolute position.
