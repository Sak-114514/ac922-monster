# V27 preparation status

**V27 run completed with wrapper exit 1; supervisor stopped and all GPUs are released. No autorun or retry is configured.

- V65 runtime build ID `20261016T021234.138365Z`; report SHA-256 `d3210b011c29160e8cf2c2f0d3722356cbb7358b90db6841d27fc8b9977b85db`; V65 whole-closure ldd is verified.
- Runner is reused byte-for-byte from V26, SHA-256 `8f25242b2c8bb8fe09eba2002386d5d98f5b8de08886e9e491a6499b76a5c44d`; no engine or runner build.
- Same IQ3_XXS/all-experts/Q8_0 KV, hierarchical EP6, PLE 8 GiB/six nodes, 2 banks × 2 lanes, D3, 16K/request, UB1024/MTP512; quality, capacity, timing, and safety settings are otherwise unchanged from V26. `GGML_CUDA_GRAPH_STRUCTURAL_TRACE=1`; CUPTI/preload remain off.
- V65 root metadata: one C++ object/no CUDA object; clone merged physical graph descriptors and the internal view/src maps; keep temporary AllReduce rewrites in owned `ctx_m`, leaving original `bcj.nodes` untouched; fail closed for opaque extras; preserve raw graph UID.
- V26 failure and V25 history remain separately recorded in their STATUS files. V26 logged stale `allreduce_1` garbage descriptors.
- **Run authorization is limited to this one supervised attempt; no automatic retry.** The V65 tiny gate, merge immutability regression, and root GPU grant are confirmed in the provenance below. The V27 supervisor requires three explicit gate flags, rechecks V65 hashes/ldd and live GPU/BMC/fans, invokes the wrapper through `/bin/bash`, and has a 20-minute bound.


## V65 merge immutability qualification

The old/new V64/V65 regression logs are recorded in the plan manifest with SHA-256 values. V65 passed A-unmerged → A-merge → new-UID B rebuild → A return: 18 descriptors, exact outputs, max_abs 0, canary pass, and zero source-view mutations (V64 control reported six mutations). The earlier B failure was a fixture input-ordering issue; the engine oracle was not weakened.


## V27 run result

- Tag/unit: `root-p4-v27-shortctx16k-hierarchical-ep6-d3-20261016t022431z`; one model load; runtime V65. Result directory: `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v27-shortctx16k-hierarchical-20261016/results/root-p4-v27-shortctx16k-hierarchical-ep6-d3-20261016t022431z`. The unit ended with exit 1 because the wrapper's diagnostic comparator reported only `single8k-pp-probe-prefix` and `pp-warm-prefix` mismatches. The runner itself returned 0. No automatic retry occurred.
- Native semantic gate: **12/12 scenarios passed**, 32 native cases. Preclean full-F32 gate: **40/40 serial/P4 logit pairs exact**, **42/42 before/after-MTP pairs exact**, no preserved token-parity failures.
- The clean P4 comparator passed; all four 256-token outputs exactly matched their serial references. The record is `clean_performance_eligible=true`, with 847 shared tokens / 79.783215 s (**10.616 tok/s**), 32,768 input tokens, prefill wall 62.189831 s, and scenario wall 181.557128 s. These numbers belong to the V27 structural-trace/sparse-timing profile and are not a production performance claim.
- CUDA structural tracing recorded 48 fail-closed rejected-graph entries, zero `opaque external` records, and no SIGSEGV.
- Supervisor telemetry: 62 samples from 02:24:33Z–02:39:27Z; GPU free minimum 830 MiB, maximum GPU/BMC temperature 46C; fans stayed at 3000 RPM targets with no alarm/fault samples. DDR node0+8 MemFree minimum 174.6 MiB was record-only. After stop, all cards returned to 16,124–16,126 MiB free and 0% utilization.
- Overall verdict: native/F32/main-P4 parity and clean comparator passed, but the total wrapper gate is **FAIL** because the two probe-prefix checks remain failed. Keep those failures visible.
