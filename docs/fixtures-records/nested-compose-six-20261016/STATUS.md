# V72 host-only preparation

Date: 2026-10-16 UTC

- The V71-qualified nested parent-manifest and PLE fixtures are copied into
  this isolated directory; no engine source is changed.
- The parent fixture retains its V71 exact/canary, A/B/A, A4 hot, same-UID
  changed-buffer, new-UID cold-drain, cache, and PLE owner-reset checks.
- It adds a post-invalidation stable A warmup followed by 16 same-UID/same-input
  graph calls. `GGML_META_NESTED_COMPOSE_TRACE=1` is fixture-only. The C++ gate
  requires at least 12 actual `phase=replay` log increments during that marked
  stable window, two AR6 nodes per physical rank, exact/canary output for all
  16 calls, and zero ordinary child AllReduce dispatch during the replay loop.
- Root's V72 source documents log fields: `meta-nested-compose ...
  phase=record|replay ... replays=N`, final summary `replays=N recordings=N
  declines=N`, and `CUDA compose ready ranks=6 children_per_rank=X
  ar6_per_rank=Y`. The fixture parser distinguishes `phase=record` from actual
  `phase=replay` and requires `ar6_per_rank=2` for this two-boundary graph.
- V73 host links composition, PLE, and pure-F32 segmented-AR control binaries;
  `ldd` resolves all four GGML libraries to `runtime-perf-v73/lib`. Tag:
  `v73-nested-compose-host3`.
- `bash -n` passes for all guarded shell runners. Both new C++ fixture sources
  passed `c++ -fsyntax-only` against the V71 source headers; the full host link
  against V73 then passed.
- One root-authorized large control reused an EP Q8 fixture with 64 rows per
  expert and composition explicitly off. It initialized flat6 and nested
  P2P communicators, then aborted before AllReduce at `ggml-cuda/mmid.cu:155`:
  `GGML_ASSERT(nbytes_shared <= smpbo) failed`, with `mul_mat_id` route batch
  shape `[32,32,85]`. This is an expert-kernel fixture limit and says nothing
  about segmented P2P AR.
- A pure F32 dense/ADD partial-AR control uses the already-qualified nested
  graph shape, scaled to `NESTED_COMPOSE_N=1,048,832`. Its output is 4,195,328
  bytes (1,024 bytes above the 4 MiB segmented threshold); it contains no
  `MUL_MAT_ID` or expert routing. The V74 binary is host-linked to
  `runtime-perf-v74`; binary SHA-256 is
  `cba2a09d746b51495833bad029dd48b1b43419eb2e08bd9a8c5c084af5a18a15`.
- One root-authorized V74 run (`v74-segmented-prep-a1`) was bounded to 120 s.
  It timed out with status 124 at 120 s. The log stopped after the nested
  P2P6 communicator initialized: there was no `p2p allreduce launch`, output
  shape, or PASS record. During the run the process used about 1.98 GiB host
  RSS and one CPU thread; this points to fixture setup/oracle cost but does not
  establish exactly where it spent time. This is **not** an AR failure and
  provides no segmented-kernel result. Raw log:
  `results/v74-segmented-prep-a1-segmented-ar-20261016T043533Z.log`.
- The bounded runner now validates the pure-F32 exact/canary output and
  six-rank `kind=1` P2P dispatch, without EP-only `foreign_zero` assertions;
  its timeout is 120 s. Postflight after timeout: GPUs 0–3 at 0 MiB, GPUs 4–5
  at 4–5 MiB, 38–39 C, util0, no compute PIDs. Do not rerun this large-N
  geometry. A direct six-rank P2P callback microfixture is a possible next
  route, but has not been implemented or run.
- V73 host binary hashes: nested composition
  `e8b7b4cc1df16584b6d478fcc80d85a03fab19ff3e8eab797f71ff85ee2d30cc`, PLE
  `987ca6bfbc8a0c54b9b7b9ad44cdd057a87638a97f451bd20b79e84a0ee999b1`, and
  F32 segmented control
  `f03b40dfa7558edee86a06fbbda0433ae101e61fa81e6cd626f0a7e88a6698f5`.
- After the EP attempt all six GPUs were free: 2–6 MiB/card, 39–41 C, util0,
  no compute PIDs. At that point no composition or PLE GPU probe had been run
  from this directory. Do not rerun a GPU case without a new root grant.

## V75 composition qualification attempt

- V75 build closure ID from `notes/V75-build.json`:
  `20261016T043346.193648Z`, with no blocked compile actions. The fixture host
  build tag is `v75-compose-root1`; the composition binary SHA-256 is
  `8d7912094f184426e180cbd4e225b14069c860a2a80e7fb164c1296fecb8f6a0`.
  `ldd` resolved `libggml-cuda`, `libggml-base`, `libggml`, and `libggml-cpu`
  to `runtime-perf-v75/lib`.
- One root-authorized composition attempt started at
  `2026-10-16T04:40:17Z` and exited in about 5 seconds with fixture status 1.
  The initial A-warmup, same-UID B-input, and A-restore outputs were exact and
  canary-clean. The runtime reported `CUDA compose ready ranks=6
  children_per_rank=3 ar6_per_rank=2 expected_ar6=2`, then an actual
  `meta-nested-compose ... selected=1 phase=replay ... replays=1` event.
- The fixture stopped at the preserved A4 assertion:
  `META_FRAGMENT_CACHE A4 did not submit exactly two async outer P2P6
  reductions`. The 16-repeat composed-replay window and subsequent invalidation
  checks were not reached. Because composition was selected on this call, the
  ordinary async-dispatch counter may not represent the equivalent two AR6
  nodes; that is a candidate interpretation, not a verified fixture fix. No
  assertions were relaxed and no retry was made.
- PLE was not run because the required nested-composition PASS gate failed.
  Postflight: GPU memory 1/1/1/1/5/5 MiB, 39–40 C, util0, no compute process.
  Raw log:
  `results/v75-compose-root1-nested-compose-20261016T044016Z.log`.

## V75 corrected composition and PLE qualification

- The first V75 attempt exposed a stale counter: when the composed graph replay
  already supplied both AR6 boundaries, the A4 gate still demanded two
  ordinary async-dispatch log events. The fixture now checks
  `async_AR_delta + 2 * selected_compose_replay_delta == 2`. A first cold drain
  is permitted only with exactly one `selected=1 phase=record` and a recording
  count delta of one; otherwise the A4 cold-drain delta must be zero. If A4
  consumes a compose replay, the fixture still requires `ar6_per_rank=2`.
- All later boundary-count gates also count two represented AR6 boundaries per
  selected composed replay. The 16-repeat window now changes input contents
  across frames 5, 6, and 7 while retaining the same parent UID and bindings;
  every output remains checked against the CPU oracle and canaries. The window
  requires at least 12 actual selected `phase=replay` logs and reported replay
  increments, zero ordinary AR/async dispatch, and zero cold drains. The
  composition-disabled pure-F32 branch is unchanged.
- V75 build closure ID remains `20261016T043346.193648Z`; the corrected host
  build tag is `v75-compose-root2`. The composition binary SHA-256 is
  `e394a9159f5f1c18f639163e9e39cb36875dfa77e9b908b7beeb458f97c8fa38` and
  `ldd` resolves all four GGML libraries to `runtime-perf-v75/lib`.
- One root-authorized composition run passed. A4 reported
  `async_outer_ar_delta=0`, `compose_recordings_delta=1`,
  `compose_record_phase_delta=1`, `compose_replay_delta=1`,
  `ar6_per_rank=2`, `ar_equivalent=2`, and `cold_drain_delta=1`. The 16-frame
  window reported 16 actual phase replays and 16 reported replays; all 16
  outputs were exact/canary-clean across frames 5/6/7, with ordinary AR and
  async AR deltas 0 and cold-drain delta 0. The runner also passed the
  parent-UID cold/capture ordering, same-UID new-binding invalidation, and
  parent-manifest forced-rebuild gates (`forced_rebuilds=3`). Raw log:
  `results/v75-compose-root2-nested-compose-20261016T044442Z.log`.
- Since nested composition passed within the same 120-second grant, the V75
  PLE probe then ran and passed: `owner_A_B_A=1`, fresh scheduler UID after
  reset, parent manifest selected 4 times across 8 calls with one forced
  rebuild, and `full_model_loaded=0`. Raw log:
  `results/v75-compose-root2-ple-parent-manifest-20261016T044447Z.log`.
- Postflight after both probes: GPU memory 1/1/1/1/5/5 MiB, 38–39 C, util0,
  no compute PIDs. No engine source was changed by this fixture task.
