# V57 three-arm route check

Run tag: `20261015T230509Z-69962`  
Runtime: `/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v57`  
Runtime `libggml-cuda.so.0` SHA-256: `e5a91189f18dce2eed73369a36088696542ee09ebdf94cde195ed3c6c226f983`  
Fixture binary SHA-256: `5f9ce8edab66469a68728d9d5151454618f4c58f1db9eb55c434ba1c4486b6f0`  
Fixture source SHA-256: `3563c74443c6bec1e2912b90d4ae8b639c145d82e82b35e27c555b395a0bcbf3`

## Gate results

- Host-link against V57 succeeded; the CPU-only geometry/oracle contract
  passed all 24 shape/grid checks and the zero-partial case. CUDA visibility
  was empty for that check.
- GPU0–2 UUID, idle-process, free-memory, and temperature guards passed before
  dispatch: each had 16,124 MiB free and was at 37–38°C.
- All three route processes used `GGML_META_CONTEXT_VIEWS=1` and had CUDA graph
  capture disabled.
- `p2p64`: fixed grid 64, 114 P2P rank launches, 54 segmented launches, and 2
  zero-slice launches; no NCCL F32 fallback selected.
- `nccl-f32`: zero P2P launches; the F32 wire path was selected for all 38
  Meta3 dispatches.
- `hybrid4m`: initialization selected the route with
  `reason=initialized_nccl_f32`. The three large payloads (4,211,716,
  16,797,700, and 33,579,012 bytes) each produced six actual NCCL F32 route
  markers with `threshold_bytes=4194304` and `graph_merged=0`. Summary and path
  counters both reported 18 large F32 calls. Smaller and zero-partial payloads
  stayed on P2P; total P2P rank launches were 60.
- All 21 output hashes in each arm match bitwise across the three arms.
  Per-rank CPU-oracle checks and canaries passed; the runner emitted
  `ROUTE_COMPARE_PASS`.
- After the run, GPU0–2 each returned to 0% utilization with 5 MiB reported
  memory use, at 37–38°C, and no compute processes.

## Limits

GPU3–5 were occupied by a separate small fixture during the run, so timing is
contended and is retained only as diagnostic data. This check establishes route
selection and synthetic Meta3 correctness for V57; it is not a performance
qualification or a full-model result.

One informational `AR_ROUTE_CALL_PROOF` line in the `nccl-f32` log was joined
to a concurrent optimization-path summary line by stdout/stderr interleaving.
The in-process route assertions executed on all dispatches, the NCCL summary
reported 38 selected F32 calls, all hash/oracle/canary checks passed, and the
runner completed. Preserve the raw log; a future fixture revision can serialize
log output if per-call text is needed for machine parsing.
