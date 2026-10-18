# V30 one-bank four-lane D3 host preparation

Status: The corrected V30b one-shot completed with final_rc=1. V71+parent-signature host qualification, preclean logits, and native semantic checks passed, but two of four real P4 output-token arrays diverged from the same candidate's serial reference and the original auxiliary-prefix checks failed. A separate V71 GDN X4 operator probe passed on GPU0 with actual X4/deferred-gather dispatch, but it does not resolve the full-trunk mismatch. No performance win or quality certification is claimed.

- Profile: hierarchical EP6, one non-unified bank, four 16K streams, D3 MTP.
- Shape: 24 fixed-prefix input rows and 16 candidate target-verify rows.
- Gates retained: native semantic check, serial/P4 F32 preclean comparisons, full generated-token checks, diagnostic/clean context families, and both original 32-token auxiliary prefix checks.
- Mapping: all four P4 flows are required once in bank 0; the result indexer rejects duplicate or missing flow IDs.
- V30b runtime closure: V71, build ID `20261016T040204.430535Z`, with `GGML_META_FRAGMENT_PARENT_SIG=1`.
- V30b host runner SHA-256: `9f114e7101e909a3d30d1bb7d7c0823160b25e55520294b9dd30b51e3449b24c`.
- Host-only verification: AC922 `run-cpu-tests-v30.sh` and `compile-runtime-v30.sh`; V71 build report and `ldd` closure checks passed.
- V69 first attempt loaded the model once, then stopped before quality scenarios. The corrected V30b one-shot later loaded the model once; its full result is recorded below.

## V30 first attempt and V30b correction

- First attempt: `root-p4-v30-onebank4d3-hierarchical-ep6-shortctx16k-20261016t035857z`; full weights loaded once, then the fixture stopped at `target verify exceeded the 8-row runner budget`. No quality or capacity conclusion was produced. All six GPUs were released; the attempt log and manifest remain under `results/`.
- Correction is fixture-only: the runner uses `v30-verify-row-budget.h`, whose budget is `lanes * (draft_depth + 1)`. The runner calls the same helper covered by the CPU tests. The test accepts 16/rejects17 for four-lane D3 and accepts 8/rejects9 for the prior two-lane D3 shape.
- AC922 CPU policy/plan tests and V71 host link/closure passed after the correction. V30b also requires the V71 parent-signature path with `GGML_META_FRAGMENT_PARENT_SIG=1`. V30b runner SHA-256: `9f114e7101e909a3d30d1bb7d7c0823160b25e55520294b9dd30b51e3449b24c`.
- The source audit found no other hard-coded eight-row verify, allocation, or stride assumption. The remaining `8` literal limits warmup frames (1–8), not tensor rows.
- V29 prerequisites: preclean and semantic gates passed; generated arrays match V27 (42 diagnostic, 4 clean). Its two original auxiliary-prefix failures remain failed and V30 keeps both checks.


## V30b one-shot result

- Run: `root-p4-v30b-onebank4d3-hierarchical-ep6-shortctx16k-20261016t041327z`; V71 build ID `20261016T040204.430535Z`, `GGML_META_FRAGMENT_PARENT_SIG=1`, one model load. `run_rc=0`, `final_rc=1`.
- Preclean gate: PASS, 40/40 serial/P4 full-logit pairs exact, 42/42 same-flow before/after-MTP pairs exact, 42 finite logit rows. Native semantic gate: PASS, 32 cases across 12 scenarios. Shape observed: one bank, four 16K streams, D3, prefix 24 rows, verify capacity/observed 16 rows; all four requests produced 256 tokens.
- Own-candidate serial/P4 generation comparison failed for `real8k-0` at token index 2 and `real8k-1` at index 13; `real8k-2/3` matched. The corrected V27 comparator maps only the V27/V30 scenario prefixes and the intentional `2x2`/`1x4` P4 label change; it found diagnostic 40/42 and clean 2/4 exact arrays, with no missing or unexpected flows. The original raw-label comparator report remains preserved.
- Both original auxiliary prefix checks failed (`single8k-pp-probe-prefix`, `pp-warm-prefix`). Full V30 quality certification therefore failed. V30b token/text samples for own serial, diagnostic/clean P4, and V27 are in [v30b-real8k-token-review.json](results/root-p4-v30b-onebank4d3-hierarchical-ep6-shortctx16k-20261016t041327z/v30b-real8k-token-review.json).
- Clean P4 measured 879 shared tokens over 17.414982s (**50.474 tok/s**) and 32,768 aggregate prefill tokens over 45.339863s (**722.719 tok/s**). Six direct `LLAMA_STEP_TIMING avg_us.compute` windows had median 96.339ms, p90 1185.422ms, and mean 268.194ms; the windows have mixed batch/token shapes. The analysis artifact is [v30b-run-analysis.json](results/root-p4-v30b-onebank4d3-hierarchical-ep6-shortctx16k-20261016t041327z/v30b-run-analysis.json). These metrics are retained for review only; no gain is certified because exact token gates failed.
- V72 build interval: 04:17:56Z–04:19:20.368194Z. The clean shared generation interval was 04:23:37.711908Z–04:23:55.126890Z, after the V72 build froze. V72 overlapped earlier diagnostic work, not the clean P4 interval.
- Safety: no supervisor guard fired; final status showed no model or CUDA compute-app PIDs and all six GPUs returned to idle. Maximum observed GPU/BMC temperature remained below the 58C ceiling.
- Latest CPU regression log: `results/cpu-tests-v30-20261016T042748Z-140719.log`.
- Isolated GDN probe: V71 GPU0 `volta_gdn_x4` selected 5/5 and `gdn_state_gather` selected 5/5; attention, snapshot/cache/canary, and 16 rollback-slot checks were bitwise exact. The run log is [gdn-x4-v71-gpu0-20261016T045900Z.log](gdn-x4-probe/results/gdn-x4-v71-gpu0-20261016T045900Z.log), SHA-256 `4bfcaae541986ab7c1eba836190f54ace698526f0e25e2ac7ae71e848b8bead1`. GPU0 returned to 2 MiB used at 39C with no compute process. This is operator-level evidence only; the full-trunk token mismatch remains unresolved.
