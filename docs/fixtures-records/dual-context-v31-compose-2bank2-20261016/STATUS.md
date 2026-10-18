# V31 run status

- State: one authorized full-model run ended with SIGSEGV/139 before the preclean F32 gate and clean family. No automatic retry was attempted.
- Runner: V29-qualified PPC64LE binary reused unchanged, SHA-256 `8f25242b2c8bb8fe09eba2002386d5d98f5b8de08886e9e491a6499b76a5c44d`, BuildID `8560fce4abedc573a22840fd81d9dfdf9dc34546`.
- Runtime closure: V75 build `20261016T043346.193648Z`; report SHA-256 `8dac3a5cb624684dfc7583c7bc1ea96804478f5bb326cfc43b41b844cd6a8100`; seven required application libraries resolve from `runtime-perf-v75/lib`.
- Geometry: two banks × two lanes, D3, 16K context per sequence, target UB1024, MTP UB512; existing V29 capacity/safety reserve policy is unchanged.
- Candidate delta: enable parent fragment signatures and nested graph composition (`GGML_META_FRAGMENT_PARENT_SIG=1`, `GGML_META_NESTED_COMPOSE=1`). Other V29 runtime flags are preserved.
- Quality outcome: native semantic checks passed 12/12 scenarios and 32/32 cases; the completed 38 V27 token-array flows were exact. Four P4 flows are missing because the run crashed during their prefill. Full-F32 serial/P4 preclean and clean P4 performance were not completed; this run is not a qualified performance result.
- Failure: `real8k-p4-2x2` prefill; SIGSEGV `si_addr=0` in CUDA `cuGraphLaunch`, through `ggml_backend_sched_graph_compute_async` and `llama_context::graph_compute`. See `V31-run-outcome.md` and the original run log below.
- Run tag: `root-p4-v31-shortctx16k-hierarchical-ep6-d3-20261016t044546z`.
- Original run log: `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v31-compose-2bank2-20261016/results/root-p4-v31-shortctx16k-hierarchical-ep6-d3-20261016t044546z/runtime.log`.
- Launch policy: no automatic retry. A follow-up run requires a new explicit root GPU-window grant after diagnosing the CUDA graph crash. The launch used the existing 30-minute cap and unchanged safety checks.
- No BMC or fan settings were changed.
