# V35 preparation status

Status: V80 host-only re-link and one authorized full-model run are in progress.
The inherited V30b four-lane parity and auxiliary-prefix failures remain
unresolved and are still mandatory failures in the V35 quality gates.

## Frozen candidate contract

- Runtime: V80 whole dynamic-library closure, build ID
  `20261016T071002.948627Z`; report SHA-256
  `a942e863575383bf05c9ee7c8e1cab6a75741e5a8b52fe0f57ad74a1f2af19b1`.
- Geometry: one bank, four lanes, D3, non-unified KV, 16K cells per sequence,
  65,536 aggregate cells, target ubatch 1,024, MTP ubatch 512.
- Features: standard indexer plus standard sparse QSA and standard head-loop,
  narrow GET_ROWS, fused T16, D3 four-lane target pad (16-row budget), V77
  compose/parent-walk/parent-cache, 10 MiB bounded CPU-copy cap with the
  existing 64 MiB pinned budget.
- Instrumentation: sparse Meta/step timing; no Nsight, CUPTI, or preload shim.
- Quality: all existing native, F32, generated-token, auxiliary-prefix,
  clean-family, and V27 token-array gates remain active. The clean family also
  measures one uncached 8K request with 32 output tokens after the original P4
  generation. It has no F32 dump or GPU spans and is exact-compared against the
  same-run diagnostic single-lane probe.

## Local kernel evidence

- Standard-QSA head-loop: eight GPU helper shapes passed bitwise score and
  top-k-ID checks.
- Narrow GET_ROWS: seven GPU helper shapes passed output/canary checks, with
  a selected production-shaped width-1 dispatch marker.
- EP fused T12/T16: seven quant formats at 12 and 16 tokens passed bitwise
  parity (14/14), zero maximum error, and foreign-route-zero checks.

These bounded operator fixtures do not certify full-model quality, activation,
or performance. The V35 runtime comparator requires actual selected markers
for four-stream QSA head-loop, eligible narrow GET_ROWS, and fused T16.

## Known limits

- Prior V30b failed two same-candidate serial/P4 token comparisons and both
  auxiliary-prefix checks. V35 must pass those original gates; passing the
  native semantic subset alone is not certification.
- One-bank capacity and graph allocation are not proven by the static ledger.
  Keep the existing 512 MiB/card free-memory floor, 58°C limit, and BMC fan
  watch active during the authorized model run.
- The 16K per-request capacity is a bounded short-context test profile, not a
  production context claim or a throughput target result.

## Verification status

The final host CPU tests, V80 link, live run unit, logs, runtime measurements,
and all quality-gate outcomes will be appended here. A full-model run is only
considered successful if the original exact and semantic gates pass; known
failures remain visible if they recur.
