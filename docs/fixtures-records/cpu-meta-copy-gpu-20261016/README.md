# CPU→Meta async copy qualification fixtures

These fixtures are scoped to the CPU-to-Meta byte-copy/lifetime path. They do
not load a model, run GEMM/graph compute, change core files, use NVCC, or claim a
performance gain.

## Direct V57 upload-ring proof

`cpu-meta-copy-gpu-fixture.cpp` adapts the CPU-only ring candidate in
`fixtures/cpu-meta-copy-20261016/cpu_meta_upload_ring.hpp` to real V57 Meta3
APIs. Two independent Meta3 owners share the helper's 64 MiB process budget.
Each owner copies five 32 KiB host vectors through two 64 KiB pinned slots.
This fills slot 0, flips to slot 1, and forces conservative synchronization of
the oldest recorded event before slot reuse. Source vectors are mutated and
destroyed after the ring call. Readback checks all five destinations exactly;
B is shut down with pending events and A runs again after B teardown. The
event-query callback always returns busy, because V57 has no generic query API.

The direct ring GPU run tag was `20261015T234028Z-73479`. The fixture binary
printed `PASS REAL_META3_CPU_COPY_RING` and all 15 destination checks were
exact with intact canaries. The shell wrapper initially exited 1 because its
summary assertion expected five writes for A's second phase, although A's
owner counter is cumulative and correctly reported ten. The original exit-1
summary remains preserved. A CPU-only validator was corrected to check A as
5 → 10 writes, B as 5, and rechecked the existing run log without GPU access:
`VALIDATION_PASS`.

- Run log SHA-256:
  `f1913f2325aa42343c2fe836cc6aa421ec026363cd0b206deabb7f92418cef58`
- Corrected validation log:
  `results/20261015T234028Z-73479/corrected-validation.log`
- Corrected per-phase checker:
  `validate-existing-log.sh`

The reusable `run-once.sh` counter check now reflects cumulative per-owner
counts. It was not rerun after this script-only correction.

## V58 actual CPU→Meta adapter dispatch

`cpu-meta-copy-adapter-fixture.cpp` exercises the root-integrated V58 adapter
through the public `ggml_backend_tensor_copy_async(cpu, meta, src, dst)` API.
It opts in before Meta backend creation with
`GGML_META_CPU_ASYNC_COPY=1`. Each temporary source tensor is a contiguous CPU
host COMPUTE allocation. The source values are mutated, then the CPU tensor
buffer/context are freed immediately after `ggml_backend_tensor_copy_async()`
returns. Destination tensors use the Meta COMPUTE buffer; the fixture reads
each result back from all three Meta member tensors after backend
synchronization.

The run covers two independent Meta contexts in A/B/A order and frees the
shared CPU backend before Meta-context teardown. A separate CPU WEIGHTS source
is passed through the API to verify the `weights_source` decline and legacy
synchronous fallback.

V58 was frozen as build `20261015T234005.671446Z`. The adapter fixture linked
against `runtime-perf-v58`; binary SHA-256:
`256e0530ace8946a689f39a29d419361b046480db8b3c49fe1c028d7ba48c381`.

The corrected root-authorized run passed under tag
`adapter-20261015T234541Z-74037`:

- Two Meta contexts logged `requested=1 enabled=1` and the actual CPU→Meta
  async active marker.
- A summary: 6 copies, 196,608 bytes, 1 slot wait.
- B summary: 3 copies, 98,304 bytes, 1 slot wait.
- The WEIGHTS fallback logged `reason=weights_source`.
- All 30 all-rank destination readbacks matched exactly with canaries intact.
- The source CPU backend was freed before Meta-context teardown.
- GPU3–5 returned idle at 16,125 MiB free each, 38/38/39°C.

The first adapter launch stopped before any copy at the fixture's CPU
descriptor allocation because its source context was not `no_alloc`. That
failure is retained in
`results/adapter-20261015T234413Z-73793/run.log`; the fixture now creates
descriptor-only CPU contexts before `ggml_backend_alloc_ctx_tensors()`.

Corrected adapter artifacts:

- Run log SHA-256:
  `2d8561428118ec5853b5de7163b96913ea174d15acd6ea1f1f3681ac112eb6c7`
- Summary SHA-256:
  `eac21fcfb6d648e687e7c656362a40d6febf2865e3971b42ac518e44104760f2`
- V58 host-build log:
  `results/host-build-adapter-v58-fixed-20261015T234456Z-73952/host-build.log`

This proves the V58 adapter selection and source-lifetime/readback path for the
tested CPU→Meta copies. It is not a model result or a performance measurement.

## V59 1 MiB query and policy-limit probe

`cpu-meta-copy-adapter-v59-fixture.cpp` exercises the frozen V59 adapter through
`ggml_backend_tensor_copy_async(cpu, meta, src, dst)`. It submits five 1 MiB
copies in each A/B/A phase, mutates and frees every temporary CPU source after
the call, checks all three Meta members with exact values and canaries, then
submits a 2 MiB copy with `GGML_META_CPU_ASYNC_COPY_MAX_BYTES=1048576`. The
2 MiB copy must log `reason=copy_policy_limit` and take the existing synchronous
fallback. The shared CPU backend is freed before either Meta context is
destroyed.

`meta-event-query-trace.cpp` is a fixture-only `LD_PRELOAD` observer for the
exported Meta event creation, record, and query API. It confirms actual V59
query return values without changing engine source. The fixture binary and
observer were host-linked to the frozen `runtime-perf-v59` manifest
`83761eec8803f6147057ea78f07a0086af79938b1cb9ee5bce83b076a438f041`:

- Adapter binary SHA-256: `1d24a1d65ebc398e3ccc44c82fbc07d25a5f01ae7ec545bc38e0e15f6925f3f8`
- Query observer SHA-256: `4989d40c1de5bf4979585325180248700e44cb26db8b8771a3fbdec560437b48`
- Host-link log: `results/host-build-v59-20261016T001632Z-78279/host-build.log`

The one authorized GPU3–5 run, tag `20261016T001641Z-78322`, passed the
1 MiB async path, 2 MiB policy fallback, A/B/A source-lifetime checks, all 48
exact all-rank readbacks, and canaries. The trace recorded 15 Meta event
records and 13 real Meta event queries, all returning ready (`status=1`). The
copies completed quickly enough that the ring created one event slot per Meta
owner; **this run does not qualify two simultaneous 1 MiB slots**. No synthetic
stream delay was used. The two-slot case remains unproven by this run, while
the actual query/reuse path and completed-event handling were observed.

- Run log SHA-256: `1e2e0047dc3ee7e1b1cbdf6b230d47f8d70101956fb2e58ee1794135068550d3`
- Summary and route validation: `results/adapter-v59-20261016T001641Z-78322/summary.log`
- CPU-only log validator: `validate-adapter-v59-log.py`
- Final GPU guard: GPU3–5 idle, 16,123–16,125 MiB free, 37–38°C, no compute
  process listed.

The V59 probe is a byte-copy fixture only. It loaded no model and makes no
performance claim.

## Reproduction commands

V57 direct ring helper (root GPU window required):

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
CPU_META_COPY_GPU_AUTH=ROOT_CPU_META_COPY \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v57 \
  ./fixtures/cpu-meta-copy-gpu-20261016/run-once.sh
```

V58 actual adapter (root GPU window required):

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
CPU_META_COPY_ADAPTER_GPU_AUTH=ROOT_CPU_META_COPY_ADAPTER \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v58 \
  ./fixtures/cpu-meta-copy-gpu-20261016/run-adapter-once.sh
```

V59 actual adapter (root GPU window required):

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
CPU_META_COPY_V59_GPU_AUTH=ROOT_CPU_META_COPY_V59 \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v59 \
  ./fixtures/cpu-meta-copy-gpu-20261016/run-adapter-v59-once.sh
```
