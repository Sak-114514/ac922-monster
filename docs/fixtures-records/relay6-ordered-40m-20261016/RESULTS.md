# V94 ordered Relay6 40 MiB fixture results

## Host preparation

- Frozen V94 runtime: `runtime-perf-v94`, build
  `20261016T121300.582979Z`; report has no blockers and completed successfully.
- V94 `allreduce-p2p.cu` SHA-256 remains
  `b0790248c39350d862e09d0841575a8c7bad1e3c5de19fe50e55c8c9ad311b19`.
- Fixture-only V94 host build passed with report build ID
  `20261016T121300.582979Z`; executable SHA-256
  `7276f68fc909afa673d35ff508d5ca92e16470084920f70a9a217eb9b5c0b847`.
  `build-v94-host1.txt` records the fixture source, included AllReduce source,
  binary, four V94 GGML DSOs, and resolved dynamic closure. CUDA DSO SHA-256:
  `703d0e3f8689e9b90e8b44d9b54e6d1398b8bd34b691082f2f4b7dd2a7824dd2`;
  build-record SHA-256 `864bc15aab9b5e2968ba605632b08683b21e2b033c22c8f6501a056acfa11c8e`.
- The fixture uses one communicator and exactly 40 MiB, with a 1 GiB
  per-GPU added-memory cap. It checks rank-order output, A/B/A, all-negative-
  zero signs, changed inputs, canaries, exact selected kind, and rank markers.
- The explicitly authorized seven-case run completed after the V94 cold-F32
  window: ordered kind 4 on both graph entry paths for zero ranks -1/0/3,
  plus one 40 MiB old Relay6 kind 3 dispatch control for zero rank 3. All
  seven cases selected the expected kind on all six ranks and passed the
  rank-order FP32 oracle, A/B/A, changed-input, negative-zero sign, and
  canary gates. The validator re-ran locally on all seven preserved logs.
- Maximum measured one-communicator after-graph added memory was 644 MiB per
  GPU, below the 1 GiB fixture cap. Same-window 40 MiB dispatch max GPU event
  timing for zero rank 3 was 4.296 ms for ordered kind 4 and 7.187 ms for
  Relay6 kind 3. The graph-append ordered arm measured 4.820 ms. These are
  operator-fixture timings only, not model throughput or quality evidence.
- Run artifacts:
  `results/v94-ordered-40m-v94-40m-host1-20261016T122500Z/`.
  `summary.json` SHA-256:
  `6059e7aa06d49c338d92a06495d135f39ada4ecad182639e2a3cb62c4b2eaf83`;
  run log SHA-256:
  `ae9833873ee976ba4ee2c7aed2878f6d1bef9e212a30d9625d318d7f8d12fc18`.
- Postflight: GPUs 0–5 were idle at 38–39 C, 0% utilization, and
  2/2/3/3/7/6 MiB used. The runner enforced its 512 MiB free-memory floor,
  58 C temperature ceiling, 15-second case timeouts, and 60-second total cap.
- No old fixture logs were copied into this directory. No engine source,
  model, fan, BMC, or kernel state was changed.
- The runner's unauthenticated preflight returned 77 before any GPU query or
  launch. The successful run required the explicit root authorization.
