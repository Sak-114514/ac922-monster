# V31 full-model run outcome

Run: `root-p4-v31-shortctx16k-hierarchical-ep6-d3-20261016t044546z`  
Runtime: V75 build `20261016T043346.193648Z`  
Runner: reused V29 binary, SHA-256 `8f25242b2c8bb8fe09eba2002386d5d98f5b8de08886e9e491a6499b76a5c44d`

## Outcome

The full-model process ended with SIGSEGV/139 while pre-filling `real8k-p4-2x2`. The fatal handler reports `si_addr=0`, TID `150662`, and a stack through `libcuda.so.1` → `cuGraphLaunch` → `cudaGraphLaunch` → V75 `libggml-cuda.so.0` → `ggml_backend_sched_graph_compute_async` → `llama_context::graph_compute` → `llama_decode` → `bank::prefill_full_logical_prompts`. The clean family never started; this is not a qualified performance result.

The failure occurred after native semantics completed and while the four-lane P4 real-prompt prefill was being scheduled. The log had already reported parent-manifest selection and nested composition replays. This timing correlation does not by itself identify parent signatures or composition as the fault; the null dereference is in CUDA graph launch, and needs source/runtime correlation before another full-model run.

## Quality evidence completed before the crash

- Native semantic gate: 12/12 scenarios, 32/32 cases passed.
- V27 generated-token comparison: 38/38 completed flows were exact. The four missing flows are exactly `real8k-0..3` from the unfinished `real8k-p4-2x2` scenario.
- Full-F32 preclean serial/P4 gate: not completed, because all four P4 outputs were unavailable.
- Clean P4 performance and clean-vs-serial checks: not run.
- The comparison report therefore remains failed/incomplete; no quality certification or speed claim is made.

## Runtime and safety observations

- Before the failure, the lowest sampled free HBM was 675 MiB/card, above the existing 512 MiB floor. Temperatures peaked at 45°C in the supervisor sample.
- After process exit, all six cards returned to approximately 16.1 GiB free and 39–40°C.
- Fan target remained 3000 RPM; observed main fan tach readings remained roughly 2840–2886 RPM.
- No BMC or fan settings were changed. The supervisor did not retry the failed model run.

## Artifacts

Full run directory:

`/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v31-compose-2bank2-20261016/results/root-p4-v31-shortctx16k-hierarchical-ep6-d3-20261016t044546z`

Key files: `runtime.log`, `gpu-telemetry.jsonl`, `unit-journal.txt`, `copy1/diagnostic/results.json`, `copy1/diagnostic/results.json.semantic-gate.json`, `v31-v27-token-array-comparison.json`, and `v31-direct-timing-analysis.json`.
