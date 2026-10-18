# Results: Meta graph merge descriptor lifetime

Date: 2026-10-16 UTC. This test used only a small F32 Meta3 graph on GPUs 0–2;
no inference model was loaded.

## V64 legacy result

- The warmup graph ran unmerged and produced exact output with canary intact.
- The next same-UID call merged a graph with 3 subgraphs / 2 partial boundaries.
- Output remained exact for this call, but the immutable physical descriptor
  snapshot recorded **6 mutations**: `mergeA.add1.src[0]` and
  `mergeA.add2.src[0]` changed on each of the 3 CUDA ranks, from the original
  partial tensor to temporary AllReduce tensors owned by `ctx_merged`.
- The runner accepted this only as the expected V64 regression. It stopped
  before attempting a later UID rebuild, avoiding a stale-pointer use after the
  merge context reset.

Log: `results/v64-final-20261016T022245Z.log`.

## V65 result

- The same graph merged with 3 subgraphs / 2 partial boundaries.
- The snapshot contained 18 physical descriptors across three ranks; comparing
  `src[]`, `view_src`, and `view_offs` after merge found **0 mutations**.
- A warmup, same-UID A merge, distinct-UID B rebuild, and A return all produced
  bit-exact outputs (`max_abs=0`) with intact canaries under changed inputs.
- The final run exited 0. GPU usage afterward was 4 MiB per assigned card at
  39–40 C; the compute-process list was empty.

Log: `results/v65-final-20261016T022248Z.log`.

## Provenance

- Final fixture source SHA-256: `1ae760675431c5ea3f14181d301fa46bbfa4d88cd53b62139e85a1f19b4a6e85`
- V64 final fixture binary: `125378e0ce14d8b8d8fc58a3597beb0dd34375817b56cacb5a07b34ab687be77`
- V65 final fixture binary: `a3782ee4e2e822eb6faba975fc0b2938415bc6d3d55b6bcb43a414903c310ea8`
- Runtime binaries were the frozen `runtime-perf-v64` and `runtime-perf-v65` outputs.

Earlier failed logs are retained. The V64 abort in `v64-20261016T021823Z.log`
was from the fixture writing a split input through the generic Meta allocator;
the fixture was changed to the dedicated Meta tensor allocator. The V65 B
mismatch in `v65-20261016T022039Z.log` was a fixture argument-order error in
its B input setter; the corrected `v65-final` run passes. Neither failure
required or caused a core-source change.
