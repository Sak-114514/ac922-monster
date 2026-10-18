# Streamed TC high-route-density follow-up plan

## Goal

Re-test the isolated Volta streamed-TC schedule at expert assignment densities that match larger prefill ubatches. The prior 512-token/258-active-expert probe averaged 19.845 assignments per active expert, which is representative of UB1024 only. It does not decide behavior near 40 or 80 assignments per expert.

## Scope and boundaries

- New fixture only: `fixtures/streamed-tc-iq-large-route-20261015/`; preserve the prior fixture and its negative result unchanged.
- No engine dispatch/core edits, full model, stored weight conversion, persistent F16 cache, or expert pruning.
- Test total tokens 1024, 2048, 4096, top-k 10, all 512 experts active.
- Balanced route gives exactly 20/40/80 assignments per expert.
- One-hot route sends one slot per token to expert 0 and distributes the other nine unique slots across the remaining 511 experts. All 512 stay active; mean remains 20/40/80, while expert 0 is hot at 1024/2048/4096 assignments.
- Keep seven real packed-format readers, RNE decode checks, frozen FP16/FP32 tolerances, A/B/A replay, canaries, and token-tile-64 tails.
- Use the engine's `ggml_cuda_launch_mm_ids_helper` and tile-offset scans to generate route maps on GPU, then measure the correct engine Q8_1 preparation variant: scatter for broadcast gate/up, ordinary indexed quantization for down. Include route-map/offset setup and activation preparation in the operator-level comparison; also report GEMM-only numbers.
- A single bounded GPU2 window was authorized by the parent agent after V15 stopped: keep the six GPUs otherwise idle, set a 10-minute wall-clock cap, monitor temperatures, and stop this probe if any GPU reaches 58 C. Do not touch the server/model/service scripts.

## Acceptance/reporting

For each token count/route, report per-format active-expert min/mean/max, hot-expert count, MMQ tile tails, engine route-map and offset-scan time, Q8 GPU-prep time, F32-to-F16 time, TC core, grouped-MMQ core, and core plus preparation ratios. Preserve exact seven-format packed decode/RNE checks, numeric tolerances, A/B/A replay, route-map equality, Q8/output canaries, and 64-token tile tails. Keep the prior 19.845-row negative result visible as the low-density point. No full-model result is implied.

This remains a kernel fixture, not full-model throughput evidence.
