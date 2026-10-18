# V42 capacity review

## Prior full-model evidence

V41 used target ubatch 4096 and draft ubatch 512. Its run stopped under the existing GPU-free safety monitor at a minimum 505 MiB, below the 512 MiB floor. The run is incomplete and is not a full quality or performance result. The runner was stopped, then postflight confirmed all six GPUs were released. The record is `../dual-context-v41-ub4096-20261016/results/v41-partial-outcome.json`.

The retained V41 partial evidence includes a passing native semantic subset (12 scenarios / 32 cases), but the F32 pair, clean context, same-run S1/P4 checks, V37/V27 comparisons, and final activation validation did not complete. V42 must retain these gates; it cannot inherit a full-model quality pass from V41.

## V42 memory hypothesis and risk

V96 adds two AR-path controls in `allreduce-p2p.cu`: leader-only relay staging and unused-segment elision. Source review suggests that the prior relay staging reserve may be concentrated on the two leaders, with a potential reduction near the previously estimated ~478 MiB main plus ~60 MiB draft on non-leader ranks. Those are **unmeasured estimates**, not capacity evidence. The V96 fixture must record actual per-rank/per-GPU allocation and exact oracle/canary results before any model run.

The graph-variant cap of one may reduce retained CUDA graph memory, while increasing evictions or recapture. Its memory and timing effects remain unmeasured. The combination's ability to fit the full model is unproven.

## Hard guards

- Stop if any GPU has less than 512 MiB free; do not lower this floor.
- Stop at 58°C GPU or BMC temperature.
- One model load only; no automatic retry and no GPU reset.
- Maximum run time 900 seconds.
- Keep the existing fan/watchdog state unchanged.
- No model launch until the V96 six-rank local gate passes and root grants a separate bounded window.
