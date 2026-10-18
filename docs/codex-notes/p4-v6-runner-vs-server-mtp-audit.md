# P4 V6 C API runner versus server MTP path

Date: 2026-10-15  
Scope: read-only comparison of `fixtures/dual-context-v6.cpp`, the P4 V6 artifacts, common MTP implementation, and server scheduling. No source edits, builds, GPU work, or model requests were made.

## Findings

### 1. `catchup_rows=0, fused=0` is expected for this runner's memory policy

The runner enables `LLAMA_MTP_FUSE_CATCHUP_SEED=1`, and the chain is active. Its log reports `MTP catch-up/seed fusion fallback lanes=2 catchup_rows=0 rows=2` followed by `multi-lane GPU MTP chain completed lanes=2 depth=3 catchup_rows=0 fused=0`. This is a no-op fusion decision: the fusion code only attempts a fused input when `n_catchup_rows > 0` (`src/common/speculative.cpp:2113` onward).

The C API runner preserves accepted MTP KV. In `dual-context-v6.cpp`, it processes the verified batch, calls `common_speculative_accept()`, updates the prompt, and removes target and draft memory only from the new prompt end (`:393-415`). Thus KV for accepted proposed tokens remains available in the draft context. `append_deferred()` discards deferred rows whose first position is already at or below the draft memory tail (`src/common/speculative.cpp:1673-1690`); the next `draft()` therefore has no catch-up rows.

The server intentionally follows a different policy. Immediately after producing proposals, `update_slots()` trims the draft context back to `ckpt.pos_max + 1` (`src/tools/server/server-context.cpp:3787-3803`), with a nearby TODO stating that it currently re-evaluates drafted tokens for simplicity (`:3793`). During target verification, `process()` stores accepted candidate rows and `accept()` records the accepted prefix (`src/common/speculative.cpp:1796-1822, 2571-2573`). The following draft call then appends those rows because they are absent from the draft KV (`:2014-2019`). This explains the server's observed eight catch-up rows and why it can report fusion eligible while the runner does not.

The runner's zero catch-up count is not evidence that the fusion flag failed. It reflects a more aggressive accepted-KV retention policy than the server's current conservative re-evaluation path.

### 2. The C API runner is not the server's four-lane execution topology

P4 V6 uses two banks, two lanes per bank, and a 66,048-token context per bank; each bank creates its own target context, MTP context, and speculative implementation. The configured pooled capacity is 132,096 tokens. See `results/p4-v6-pinned-pipeline-20261015T061650Z/{profile.json,results.json,runtime-proof.json}` and `fixtures/dual-context-v6.cpp:80-120, 450-485`.

The server initializes one speculative implementation with `params_base.n_parallel` lanes (`src/tools/server/server-context.cpp:1725-1734`) and the user-supplied server run used four lanes in that one context. The runner instead sets `n_seq_max=2` in each bank and creates two independent MTP chains. Its `ROOT_SERIAL_CPU_SUBMIT=1` guard is one process-wide mutex around each bank's `common_speculative_draft()` call (`dual-context-v6.cpp:260-291`, `fixtures/dual-context-v6-submit-guard.h`). Therefore, four runner lanes are drafted through two sequential two-lane MTP contexts, not one four-lane MTP context. This is a material topology difference for comparing its 104.68/106.76 tok/s result with the server's reported 131.5 tok/s.

### 3. The 30 ms draft guard hold is a synchronous API duration, not one head kernel

The guard surrounds the entire `common_speculative_draft()` call. That function performs one `llama_decode()` per MTP depth (`src/common/speculative.cpp:2280-2300`); with depth 3, it submits three separate graphs rather than one depth-unrolled graph. `begin_mtp_gpu_chain_multi()` synchronizes before starting (`src/src/llama-context.cpp:1869-1889`). While the chain is active, each `process_ubatch()` drains the draft scheduler before modifying/using chain inputs (`:2465-2468`). `end_mtp_gpu_chain_multi()` synchronizes and reads the token-history tensor to the host (`:1962-1978`).

So `draft_guard_hold_us` measures the whole chain submission, its internal waits, and final token-history readback. It cannot be compared directly with the approximately 2 ms MTP-head kernel estimate. The P4 guard comment says the explicit owner drain happens after the guard, but the MTP chain API already waits inside `common_speculative_draft()`.

This is consistent with the recorded ~30 ms per-bank draft hold. The runner also serializes two such calls across the banks, increasing the time needed to advance four lanes. The available artifacts do not split that 30 ms into GPU compute, per-step synchronization, host submission, and final readback, so the exact contribution of each remains unmeasured.

### 4. CUDA graphs were enabled and used

The P4 runtime environment has `GGML_CUDA_GRAPH_OPT=1`, `GGML_CUDA_GRAPH_STRUCTURAL=1`, and graph caching enabled. `runner.log` records CUDA graph captures and replays, including 160-node graphs. The 30 ms hold is therefore not explained by a global eager-mode setting. Per-call graph capture/replay churn is not excluded, but the observed capture/replay markers prove the graph path is active.

## Evidence limits

- `results.json` reports 14/14 semantic checks passing across the runner scenarios, not full service equivalence.
- `runtime-proof.json` confirms `LLAMA_MTP_FUSE_CATCHUP_SEED=1`, `LLAMA_MTP_GPU_CHAIN_MULTI=1`, and `LLAMA_MTP_GPU_CHAIN_SINGLE=0`.
- The server's eight-row catch-up marker and 131.5 rate are from the root's supplied server observation; this P4 artifact directory is the C API runner, not that server run.
- No disabled-MTP experiment or new baseline was run. The source and runtime markers explain the different paths without one.

## Concrete comparison boundary

The P4 runner demonstrates two independent two-lane MTP contexts with accepted draft KV retained. The server result demonstrates one four-lane MTP context with accepted draft rows re-evaluated and folded into the next MTP call. Those are different execution contracts. Compare them as such; do not attribute the throughput gap to a kernel defect or to catch-up fusion alone.
