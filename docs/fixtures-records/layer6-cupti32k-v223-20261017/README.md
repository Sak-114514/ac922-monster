# V223: one 32K target CUPTI window on the V219 UB256 candidate

This is a prepared, source-only diagnostic fixture. It reuses V219's V218-bound
runner, exact six-GPU layer split, binding-key graph route, QSA F32 indexer and
virtual TopK settings. It keeps only the UB256 family and the already-prepared
scenario order: native `Paris` canary, then one uncached 32K logical prompt.

The one CUPTI window begins immediately before the long scenario's target
`llama_decode()` and stops after the runner's existing natural
`llama_synchronize(target)`. The subsequent MTP prefill and generated-token
processing are outside that marker window. No extra CUDA synchronization was
added. Long-output quality and performance are not scored; the native Paris
canary remains the correctness gate. The CUPTI run is diagnostic and must not
be compared directly with the unprofiled V219 timing.

`collector.cpp` is the V214 R4 direct CUPTI Activity collector. Its fixed record
capacity is raised to 4,194,304 records. R4's documented 304-byte event makes
the maximum record table about 1.19 GiB of BSS; pages are backed as records are
written. This is an explicit bound, not an unbounded trace buffer. The previous
4K capture produced 509K records, so this cap gives roughly eight times that
count for the single 32K target window. The actual run remains invalid if CUPTI
reports overflow, drops, starvation, parse errors, bad buffers, or invalid
device/context metadata. The collector writes JSONL directly; do not import the
trace into Nsight.

## Artifacts

- `dual-context-v223-cupti32k.cpp`: V219-derived runner with one UB256 family,
  exactly two scenario guards, and the target-only CUPTI marker.
- `scenarios.json`: the native Paris canary followed by the single 32K request.
- `cupti-window-policy-v223.h`, `collector.cpp`, `buffer_drain.h`: exact marker
  policy and bounded direct-CUPTI collector sources.
- `trace_summary.py`: CPU-only JSONL summarizer from the qualified R4 collector.
- `v223-root-profile.json`: V218 runtime, inherited V219 environment, workload,
  trace path, memory budget, and required launch gates.
- `manifest.json`: SHA-256 identities of the prepared sources and inputs.
- `verify-v223-static.py` and `static-contract.json`: source/profile/scenario
  contract checks. They do not compile or access the server.

## Root-owned hostlink inputs

Use V219's host compiler include/library list with this fixture as `F` and
`dual-context-v223-cupti32k.cpp` as the source. Link against the frozen
`runtime-perf-v218/lib` set named in the profile. Build the collector with the
CUDA 12.4/CUPTI include and library paths documented by
`fixtures/cupti-window-20261016/r4/README.md`, outputting
`libac922-cupti-window.so` beside the source. Verify runner `ldd` resolves all
llama/ggml DSOs from V218 and record the newly built runner/collector hashes
before the root guard launches anything.

The profile intentionally remains `launch_eligible: false` until the root-owned
hostlink, live library-identity check, scenario transfer, and in-band safety
guard are ready. No build, model load, GPU request, or service launch was done
while preparing this fixture.
