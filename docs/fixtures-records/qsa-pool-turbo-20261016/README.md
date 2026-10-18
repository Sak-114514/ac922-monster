# QSA pool + Turbo frame staging

Date: 2026-10-16. Staging only; production source was not edited. No engine build or GPU execution was performed.

## Decision

The existing per-sequence pooled QSA cache can be made safe across TurboPrefill frames by invalidating only the frozen sequence's host ledger before the immutable frame graph can write new indexer cells. The first following mutable QSA graph must use the ordinary path; its existing qsa_multi_register rescan rebuilds the host cell map and marks all pooled rows dirty. The next eligible multi-pool graph must re-pool all dirty rows through SET_ROWS before its score GET_ROWS read in that same graph. A graph failure must drop the affected sequence ledger and pending commit. Other sequences' ledgers stay untouched.

The live CELLSEL path is not reference-equivalent: ordinary QSA gives the visible incomplete tail a +1e9 score priority, while the old multi path lets that tail compete by score. The staged physical-scatter patch adds the same tail priority and maps candidates back to physical cell IDs before stable TOP_K. This preserves the baseline physical-ID tie rule without relying on canonical logical ordering. The patch is not GPU or logits qualified.

LLAMA_QSA_MULTI_SPARSE remains 0. It changes attention accumulation and needs a separate numerical gate after selector parity.

## Contents

- qsa-multi-phase-ledger.h and qsa-multi-phase-ledger-test.cpp: proposed private ledger helper and CPU transitions for freeze, rescan, successful update, abort, rollback, per-sequence isolation, gaps, overflow, and 4→1→4 reuse.
- memory-hybrid-phase-ledger.patch: staged lifecycle draft; it does not include the required graph-abort call-site or context gate changes, so it is not a production-ready patch by itself.
- qsa-physical-cellsel.patch: staged qwen4exp and memory-plan delta for tail bias, physical-id scatter, stable ties, and padded-n_kv graph reuse checks.
- qsa-cellsel-contract.py: CPU oracle for candidate budget, tail priority, physical cutoff ties, and hole demotion. It injects equal block scores and does not execute ggml score kernels.
- integration-proposal.md and source-audit.md: lifecycle integration points, context-gate snippet, SET_ROWS shape review, historical B/H limits, and GPU qualification plan.
- run-cpu-tests.sh and cpu-test-output.txt: reproducible local CPU checks.

## Verification boundary

The CPU tests prove the proposed host-ledger transitions and physical-scatter candidate/mask contract under identical injected block scores. They do not prove score bit parity between compact and full GEMM shapes, CUDA graph replay, end-to-end attention logits, or throughput. The staged qsa-physical-cellsel.patch applies after the lifecycle patch in a temporary source snapshot. Keep MULTI_SPARSE disabled until a separate GPU parity gate passes.

## Latest verification

The production host planner calls build_cell_scatter_plan from the private ledger header; the C++ CPU unit test calls that same function with query windows/row_cells fixtures and asserts the emitted physical IDs, unique scratch destinations, and exact tail bias. The remote CPU runner passes. The staged lifecycle and physical-scatter patches were applied in a temporary source snapshot, then the patched qwen4exp.cpp translation unit passed a C++17 fsyntax-only compile. No engine library was linked or rebuilt, and no GPU was used.
