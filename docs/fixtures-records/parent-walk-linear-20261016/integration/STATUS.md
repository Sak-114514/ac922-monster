# V77 parent-walk integration status

Date: 2026-10-16 UTC

- This leaf directory is owned by the parent-walk integration fixture. The
  sibling generic CPU/ASan/UBSan tests and all V75 outputs are untouched.
- Copied the V75 nested-composition fixture with its A4 arithmetic, exact and
  canary, 16 changed-input stable replay, UID/buffer invalidation, and PLE
  owner-reset gates. Added fixture-only capture of selected
  `meta-parent-walk` records during the stable 16-call window and a gate that
  both `original_hits` and `prepared_hits` grow by at least 12.
- The copied shell runners enable `GGML_META_PARENT_WALK_FAST=1` and request
  `GGML_META_PARENT_WALK_TRACE=1`; the bounded wrapper requires an explicit
  root authorization and caps nested plus optional PLE execution at 60 s.
- V77 build report observed at
  `/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V77-build.json`:
  build ID `20261016T051430.291943Z`, `blocked=[]`, two compile results. Runtime
  manifest exists. Host-link against this exact runtime
  succeeded with `-Wall -Wextra -Werror`; both binaries resolve all four GGML
  libraries to `runtime-perf-v77/lib`. Main fixture binary SHA-256:
  `55250ac1e04487fb6bfb3ce088b8856bd3891198c78707460287a4fee81fc0be`;
  PLE fixture binary SHA-256:
  `1ecb3c3fd5aec96711ff4a69db7447057c99b98f994b6b35fca81b706c2d929a`.
  Host build tag/log: `v77-parent-walk-host1` /
  `build-v77-parent-walk-host1.txt`.
- V77 source adds fixture-scoped `GGML_META_PARENT_WALK_TRACE=1`, which logs
  every prepared hit; default logging remains throttled. The fixture callback
  now captures selected parent-walk counters during the 16-call window and
  requires at least 12 log events plus growth of 12 in both counters. The
  first V77 host build completed; no GPU process had been started before the
  authorized integration run below.

## V77 nested parent-walk integration result

- One root-authorized V77 nested+PLE GPU run passed within the 60-second
  bounded wrapper. Nested composition log:
  `results/v77-parent-walk-host1-nested-compose-20261016T051923Z.log`; PLE log:
  `results/v77-parent-walk-host1-ple-parent-manifest-20261016T051928Z.log`.
- The nested stable window reported 16 selected parent-walk records, with
  `original_hits` 5→21 and `prepared_hits` 5→21 (delta 16 each), across exact
  and canary-clean frames 5/6/7. Composition replay delta was 16; ordinary AR
  and async AR deltas were zero; cold-drain delta was zero. The A4 accounting
  proof was async AR delta 0 + one composed replay representing two AR6
  boundaries, with one first-record cold drain accounted.
- The inherited gates passed: A/B/A outputs, new-UID cold-drain before
  invalidation, same-UID new-buffer invalidation, and parent-manifest forced
  rebuilds=3. PLE nested owner A/B/A and fresh UID after scheduler reset also
  passed; no full model was loaded. PLE reported parent-manifest selected=1,
  calls=8, and one forced rebuild.
- Postflight: GPU use 2/2/2/2/6/6 MiB, 38–39 C, util0, no compute PIDs. The
  fresh BMC watch was 14.7 s old, host/chassis Running/On, max 40 C,
  `masks_ok=true`; fan0/2/3 targets 3000 RPM, tach 2849/2860/2845 RPM, no
  alarms/faults. No model was loaded and no system/BMC changes were made.
