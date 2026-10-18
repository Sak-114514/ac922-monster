# Target verify per-active-lane padding

Status: V33 private runner host-prepared; no model/GPU run.

- Baseline: V32 two banks × two lanes, D3, V77 build `20261016T051430.291943Z`; its semantic and preclean gates passed, while the two original auxiliary-prefix gates remained failed. V29/V31 runner sources were not changed.
- Candidate: private V32 runner copy with `ROOT_TARGET_VERIFY_PAD_PER_ACTIVE_LANE` and `ROOT_NSYS_PREFILL_PROFILE` set to exactly `0` by default; `run-target-verify-pad.sh` delegates every existing V32 gate unchanged. Opt-in target submissions emit `target-verify-pad-plan active=1` with active, real, pad, total, score, output, depth, and budget fields. The separate Nsight opt-in brackets only diagnostic `single8k-pp-warm-32` and leaves clean unprofiled.
- CPU planner: PASS locally and on AC922. Local log `results/cpu-planner-20261016T054455Z.log`, SHA-256 `b64f8cf37a9b38779b3303e8dbb6c0fb2ce69b41acc4aca5c3ce3996f572d5fa`; AC922 log `results/cpu-planner-ac922-20261016T054535Z.log` has the same SHA.
- Runner: V77 C++ syntax-only and host-link PASS. Executable `build-runtime-target-verify-pad-nsys-20261016T055956Z/dual-context-target-verify-pad`, SHA-256 `7dcd91ec6a7e80b309cbe2b2de7aa6b2965bb71315487bca8b91b49528cd0599`; source SHA-256 `81abc0b71e946b0b1b7102e97377d31296884809e06c538269eeba8014844e32`. It has not been run.
- Supervisor handoff: `v33-supervisor-contract.json` binds the V77 report/runtime, exact environment profile, candidate runner and hashes, row-padding marker, optional Nsight marker/window artifact, CLI, and CPU/V32 capacity evidence. Relay dispatch is not part of this fixture and remains separately gated.
- The server's existing `LLAMA_SPEC_PAD_MULTI` is not equivalent: its path appends rows to the last sequence in the total batch. The direct V32 fixture runner bypasses `server-context.cpp`, and the new per-active-lane planner is private to this fixture.
- Remaining proof: separately authorized V77 model run with the existing V32 quality gates and pad marker validation; if Nsight is enabled, verify its artifact and trace capture. This host-only preparation has not used GPU, model loading, or NVCC.
