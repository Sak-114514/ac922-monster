# V49 status

- Stage: host-side candidate preparation
- Runtime: V103 `20261016T162913.468680Z`, pinned by report SHA in `v49-exact-env.json`
- Stream environment candidate: `CUDA_DEVICE_MAX_CONNECTIONS=32`; SM80+ copy-connection setting and unverified launch-queue setting are intentionally absent
- Topology: ordinary 3+3 EP3; two banks × two lanes; no hierarchy or nested Meta
- Workload: four-flow semantic canary, then four distinct 8K inputs with 512 outputs each
- Geometry: two 32,768-cell banks; four 16K sequences; 65,536 aggregate target KV cells
- Target/draft UBatch: 1024 / 512
- MTP: D3, fixed prefix 6 rows per lane, verify budget 8 rows per bank
- Hardware policy: 300 W GPU caps, 2.6 GHz CPU max, 512 MiB free floor per GPU, <58 C, 3000 RPM target and >=2650 RPM tach; preserve installed fan-monitor inactive/masked policy and active watchdog
- Model or GPU actions: none
- Build/link: not run
- Host checks: plan validator PASS; semantic checker self-test PASS; shell syntax PASS; C++ `-fsyntax-only` PASS using the local V100 public-header snapshot plus an alias for one newer per-sequence carry API
- Full-model canary/performance: not run
- Root GPU-window authorization: pending

Port-map caveat: V32 overlaps two full bank owner loops, but does not expose a stage-range API for deterministic A(ctx1) during B(ctx0). V49 records grouped concurrent decode evidence only; it cannot claim single-graph A/B stage pipelining.
