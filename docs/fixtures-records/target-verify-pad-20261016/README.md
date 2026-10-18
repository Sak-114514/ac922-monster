# Private V32 target verify padding candidate

This fixture is a host-prepared V33 candidate copied from the V32 two-bank,
two-lane, D3 runner. It adds an opt-in
`ROOT_TARGET_VERIFY_PAD_PER_ACTIVE_LANE=1` path only to the private runner copy.
The default is off. No core or server source was changed; the immutable V32
launcher still applies the existing V77, semantic, preclean, generated-token,
and GPU-window gates.

The baseline is V32 run
`root-p4-v32-parent-walkfast-shortctx16k-hierarchical-ep6-d3-20261016t052412z`
with V77 build `20261016T051430.291943Z`. Its semantic and preclean gates
passed (32 cases; 40/40 serial↔P4 logits and 42/42 before/after-MTP pairs),
but its strict overall gate retained the two existing auxiliary-prefix
failures. The baseline result is in
`fixtures/dual-context-v32-parent-walkfast-2bank2-20261016/STATUS.md`.

## Source feasibility

The V32 runtime log has three unambiguous target verifies with two active
lanes and seven real rows (`lanes=2 rows=7 max_draft_rows=3 active_chain=2
fallback=0 terminal_no_draft=0`, frames 152, 90, and 70). Both lanes were
drafting, but one lane supplied one fewer than three drafts. The log does not
record per-lane draft lengths, so the identity of the short lane is unknown.
The copied runner source at `dual-context-target-verify-pad.cpp:585-723`
confirms the row layout: each active lane records its seed row in `l.rows`,
adds only real draft rows to `l.rows`, and marks each target row for output.
The optional branch preserves those scoring row IDs and appends pad rows after
each lane's real rows; `collect()` still scores only `l.rows` at `:849-884`.
The target context retains the existing `n_outputs_max = n_parallel *
(draft_depth+1)` capacity at `:355-364`, which is eight for this two-lane D3 run.
With the opt-in set, each target submission also logs a
`target-verify-pad-plan active=1` marker with active/real/pad/total/output rows,
D3 depth, and row budget. A two-active-lane D3 submission must show eight
total output rows; the marker is absent when the switch is off.

The separate `ROOT_NSYS_PREFILL_PROFILE=1` diagnostic opt-in starts/stops the
CUDA Profiler API only around `prefill_banks_on_persistent_owners()` for the
existing `single8k-pp-warm-32` scenario. The runner checks that this scenario
is a warm-graph probe with an uncached prompt and prompt caching disabled. The
range is diagnostic-family only; the clean family logs that it stayed
unprofiled. It writes
`copy1/diagnostic/nsys-prefill-window.json` with monotonic and realtime start
and end brackets plus profiler-call brackets, matching the request-window
fields consumed by `fixtures/prefill-trace-20261015/analyze_prefill_trace.py`.
The V32 wrapper continues to force CUPTI off. For a capture-enabled run, the
external Nsight Systems command must use a CUDA Profiler API capture range;
the environment switch alone does not start Nsight Systems.

The V32 exact environment already contains
`LLAMA_SPEC_PAD_MULTI=20` / `LLAMA_SPEC_PAD_MULTI_AUTO=1`. That is a server
feature: `server_lane_pad_target()` caps the total row target at
`active_lanes * (depth+1)`, then `server-context.cpp` appends repeated rows to
the final sequence. V32's private fixture calls `llama_decode()` directly
from `run_target_frame`, so the server code does not run there; moreover, a
total of eight rows does not guarantee four rows for each active lane. The
new fixture switch is therefore different and scoped to the private runner.

## Row, score, hidden-state, and rollback invariants

- D3 has a maximum of four target rows per active lane: the current seed plus
  up to three real draft tokens. When the opt-in is enabled, missing tail rows
  repeat that lane's last real token at consecutive future positions.
- Every real and pad row requests logits and target next-token embeddings.
  Only seed/draft row IDs are stored in `lane.rows`; greedy scoring never sees
  a pad row. `common_speculative_mtp_make_layout` still groups rows in encounter
  order by sequence, and `common_speculative_accept` selects hidden row
  `min(accepted, n_rows - 1)`. Since accepted drafts cannot exceed the real
  proposal prefix, the selected hidden row remains real. The pending hidden
  state that was initially copied from the last (possibly padded) row is
  replaced by that accepted real row before the next draft step. See
  `common/speculative.cpp:1786-1798`, `:1987-1999`, and `:2722-2740`.
- For a four-row lane frame, acceptance `A=0..3` retains the seed plus `A`
  real drafts and trims from position `seed_pos + 1 + A`. The tail distance is
  `3-A`, selecting the matching recurrent SSM snapshot slot `3-A`. With V32's
  `n_rs_seq=4`, every such partial rollback fits. The existing single
  `llama_memory_seq_rm` after `common_speculative_accept` removes rejected
  drafts and pads together; no early pad cleanup is added.
- Qwen convolution history and GDN SSM state already snapshot
  `n_rs_seq + 1` rollback positions. Padding stays at the end of each active
  lane, so the existing position trim and rollback snapshot semantics apply.
  The recurrent memory trim/index contract is at
  `src/llama-memory-recurrent.cpp:203-213` and `:1353-1369`; GDN snapshot
  scattering is at `src/models/delta-net-base.cpp:563-595`; Qwen conv-history
  tail snapshots are at `src/models/qwen4exp.cpp:3335-3356`.
- Completed/EOS lanes are omitted exactly as before. Lane compaction remains
  outside this candidate because changing the active lane count changes the
  sequence layout.

The server pad implementation's cleanup ordering is documented in
`server-context.cpp:4541-4550` and `:5193-5222`. The private runner preserves
its own `collect()` order: MTP process, score real rows, accept, extend prompt
with real tokens, then trim target and draft memory once.

## CPU planner tests

Run:

```sh
./run-cpu-tests.sh
```

The CPU suite checks the default-off path, the observed two-lane row mapping
with draft lengths 2 and 3 (seven real rows), output requests on all eight rows, no padding for an inactive
lane, strict real-row scoring, fake-output exclusion, all D3 acceptance lengths
and their RS/KV trim positions, shorter draft lengths 0–2, and row-budget/D4
rejection. These tests validate the planning contract only; they do not execute
the model or llama recurrent-memory implementation.

The retained CPU output is
`results/cpu-planner-20261016T054455Z.log` (SHA256
`b64f8cf37a9b38779b3303e8dbb6c0fb2ce69b41acc4aca5c3ce3996f572d5fa`). The
same suite also passed under AC922 host `g++` with the same test output SHA;
that log is `results/cpu-planner-ac922-20261016T054535Z.log`.

## Host link and gated run path

`./build-target-verify-pad-runner.sh` host-links the private runner against the
frozen V77 runtime. The link passed with runtime build ID
`20261016T051430.291943Z`; candidate source SHA256 is
`81abc0b71e946b0b1b7102e97377d31296884809e06c538269eeba8014844e32`. The
pad-marker and optional Nsight-range runner is
`build-runtime-target-verify-pad-nsys-20261016T055956Z/dual-context-target-verify-pad`
with SHA256 `7dcd91ec6a7e80b309cbe2b2de7aa6b2965bb71315487bca8b91b49528cd0599`.
Remote `g++ -fsyntax-only`, V77 host-link, and dynamic-library closure checks
passed. The build does not rebuild the engine, invoke NVCC, load a model, or
use a GPU. The complete frozen supervisor contract is
`v33-supervisor-contract.json`.
`run-target-verify-pad.sh` delegates to the unchanged V32 quality
launcher. Its private environment switch is default-off:

```sh
ROOT_TARGET_VERIFY_PAD_PER_ACTIVE_LANE=0 ./run-target-verify-pad.sh RUNNER MODEL_GGUF NEW_OUTPUT_DIR
ROOT_TARGET_VERIFY_PAD_PER_ACTIVE_LANE=1 ./run-target-verify-pad.sh RUNNER MODEL_GGUF NEW_OUTPUT_DIR
```

For a separately authorized diagnostic Nsight run, add
`ROOT_NSYS_PREFILL_PROFILE=1` and launch the wrapper under Nsight Systems with
`--capture-range=cudaProfilerApi --capture-range-end=stop`. The request-window
JSON is emitted from captured host-clock brackets; the `clean` family never
calls the profiler API. Neither opt-in has been exercised on GPU. No model or
GPU run has been performed for this candidate.
