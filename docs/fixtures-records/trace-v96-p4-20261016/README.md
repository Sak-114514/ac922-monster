# V96/V42 dual-range P4 trace preparation

Status: V42 full-model proof is complete and verified from its outcome plus hashed artifacts. The V43 source is still host-preparation work: no model or Nsight capture has been launched from this fixture.

## Trace design

The V43 trace-only source is a private copy of the immutable V41 runner. It adds one opt-in `V43_TRACE_PREFILL_RANGE=1` marker around the existing clean-family `clean-single8k-pp-probe-32` persistent-owner prefill call, after all diagnostic and preclean F32 gates. The existing `scenario.profile=true` marker starts the second range after P4 prefill and serial warmup and stops after the four-lane P4 owner loop. Nsight `--capture-range-end=repeat:2` collects those two `cudaProfilerStart/Stop` windows in one process. Nsight Systems 2023.4.4 help confirms `repeat[:N]` and that later ranges are ignored after the limit; the wrapper discovers one combined report or all matching per-range `.nsys-rep` outputs and records their exact paths. A host-only `/bin/sh` probe confirmed this Nsight build injects only `libToolsInjectionProxy64.so`; V43 permits that exact path while continuing to reject other `LD_PRELOAD` values and `AC922_SYNC_AUDIT`.

All 17 original diagnostic scenarios run first, including the 12 native semantic cases and F32 material needed by the existing preclean gate. Only after that gate succeeds does the clean family run the original uncached one-lane 8,192-token prompt for 32 generated tokens, followed by a four-lane P4 scenario using the original four real 8K prompts with generation capped at 18 tokens per lane. The prepared clean plan retains the original full-output gate, so the short trace scenario is never accepted as a clean-quality pass. The trace run is diagnostic only and does not replace or requalify the required V42 full-model run.

The prefill capture brackets the complete `prefill_banks_on_persistent_owners` call in the clean context. Full-logit scans and F32 file output are disabled there. The runtime's `target_prefill_us` field and actual Nsight kernel intervals must still be reported separately; do not call capture wall time a clean throughput result. Node-level graph tracing may add overhead.

The trace launch sets the existing, default-off `GGML_TURBOPREFILL_TRACE=1` diagnostic flag to collect the backend's actual split map. This is trace metadata only and does not change model math; it is recorded as an explicit V43 environment delta.

## Provenance and gates

The current V96 build report is `fixtures/dual-context-v42-memory-20261016/V96-build.json` (`20261016T125920.508355Z`, SHA-256 `c17cbcfc2b940bc982fff61d1a3606c380cfdd5c90c2c4de33b9da82e3b554f8`). The V42 profile SHA-256 is `ab510ee320c9a85cbed674970b6185dda9561848a1dce98b785dd5737f44e0a1`; its V96 selected-runtime `ldd` closure is pinned in `provenance/ldd-v41-runner-under-v96.txt` (SHA-256 `3fb10e6d080053b9eccc4143aac323e90bf573927fdd4c0346985c32bb74dde8`). The local V96/V42 micro gate is now PASS. Do not use the V41 quality outcome or a `V41_FULL_MODEL_QUALITY_PASS` environment token.

The actual V42 outcome is `fixtures/dual-context-v42-memory-20261016/results/root-p4-v41-chunked-ordered-flatmeta6-ep6-d3-16k-20261016t132440202842z/v42-outcome.json` (SHA-256 `85235952a9390ec436ecf2ac5b20aaad97ac5d8b2ca7edb49b6c99d313ad5e1c`). Its C++ runner returned 0 after one model load; native semantics 32/32, F32 40/40 plus 42/42, same-run S1/P4, and all-six activation passed. The outer supervisor exited 1 after a missing V41-local comparison helper, but the file-backed outcome records the C++ return code 0 and required gates. The canonical V40/V37 offline comparison is retained as informational: 32/47 exact, 15 differences. It does not block performance under the user's override. The V27 legacy delta remains separate. Run `verify_v42_trace_prerequisites.py` against that outcome, its hashed supervisor manifest, and its canonical offline comparator before any trace launch; the verifier writes a file-backed proof and the wrapper validates its hash with the selected V42/V96 files. There is no manual quality-pass environment token.

`run-v96-p4-trace-supervisor.py` defaults to a local host-only summary. Its `--launch` path requires the actual V42 outcome, matching unit manifest, canonical comparator output, root's separate trace authorization, and a V43 runner build record. It invokes the V42/V40 safety supervisor in memory, verifies V42 sources without refreshing them, and stages files only in this new V96 trace fixture.

The V42 supervisor creates the fresh unit output directory and writes `supervisor-manifest.json` before starting the V43 launcher. The launcher accepts exactly that one file after checking unit/output identity, the frozen V96 runtime and V43 runner hashes, and every trace-owned source hash in the manifest. It refuses non-directories, missing manifests, extra files, and old output directories. A CPU-only `--preflight-output-root-only` mode exercises this exact guard without invoking Nsight or the runner.

The original host-linked V43 runner is kept intact under `build-v43/` as evidence of the initial guard failure. The trace-only rebuild that allows the exact pinned Nsight injection lives under `build-v43-nsys/`.

The V42 full-model quality run is complete; the trace remains unlaunched. A separate root authorization and fresh hardware supervisor gate are required before any trace run. Do not retry automatically or claim a speed result from the trace.

## Existing prefill switch

An older `target-verify-pad` runner has a `ROOT_NSYS_PREFILL_PROFILE=1` code path for one warm 8K prefill. The V41 runner reused by V42 hard-requires `ROOT_NSYS_PREFILL_PROFILE=0`; this fixture leaves it at zero and uses the V43-private opt-in only in its trace source copy. No V41/V42 source or launcher is modified here.
