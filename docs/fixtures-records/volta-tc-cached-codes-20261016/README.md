# Cached exact-code Volta TC fixture

This fixture measures a distinct schedule: it decodes the existing quantized
weight codes once into exact FP16 integer coefficients, retains each original
FP32 weight scale, and reuses that cache across the input token tiles. It keeps
Q8 activation bytes and their FP32 `d4` scales unchanged. `cached-codes.cuh` is
owned by the parent task; this directory supplies only the fixture harness and
its host scripts.

The source corpus is the existing 21-row-sample manifest at
`fixtures/volta-tc-real/manifest.json` (7 formats × global experts 0, 17, and
511). The copied `manifest.json` is the manifest only; raw payloads are not
duplicated. On AC922, the default sample directory is
`$ROOT/fixtures/volta-tc-real`. The optional `--samples-dir` lets a runner use
the same source bytes from another mounted location.

## Gates

`build-v96.sh` requires `CACHED_TC_BUILD_AUTHORIZED=1`. It links only against
the frozen `runtime-perf-v96` shared-library closure and records fixture-source,
compiler-dependency, manifest, sample, binary, and DSO hashes. `run-v96.sh` requires
`CACHED_TC_GPU_AUTHORIZED=1` and exactly one visible GPU in
`CUDA_VISIBLE_DEVICES`; the binary selects visible device 0.

The harness has 21 correctness runs: every format × the three output shapes
`(N, cols) = (16, 7), (113, 16), (640, 80)`. Each format run holds the prepared
cache while feeding three Q8 inputs in A/B/A order, compares FP32 outputs bit
for bit against the canonical grouped-MMQ reference, and checks weight/cache/
output guards. It records bit differences and maximum absolute and relative
errors. A finite rounding difference is labeled `NUMERIC_SHIFT_PENDING_F64`
and does not skip bounded timing. Nonfinite outputs and state/canary violations
stop the run. It uses positive finite FP32 `d4` scales with full 23-bit
mantissas and signed Q8 bytes spanning `[-128, 127]`. IDs reverse the valid
local columns in each tile; padded IDs are `-1`.

The two formats whose source K is 640 (`IQ4_NL` and `Q2_0`) are zero-padded to
K=768 because this fixture kernel currently requires K divisible by 256. The
appended quantized blocks are all zero, so the source K=640 dot is preserved.
The harness reports both source K and executed K. This is a fixture padding
step, not a production shape change.

Three bounded performance cases use `IQ3_XXS`, `IQ2_XS`, and `IQ4_NL` with
cols=80 and E=16. The first two use source K=2560,N=640. `IQ4_NL` uses its
actual source K=640,N=2560 and is padded to K=768. The repeated source rows
occupy distinct expert addresses; the cached FP16 coefficients alone exceed
V100 L2. Five CUDA-event-timed graph replays report reference-only,
cached-only, and prepare-plus-cached latency. The reference and cached timing
both include K padding. Only the combined prep-plus-consumer number can
support a speedup claim. Each case checks its estimated additional device
allocation is below 1 GiB. These are operator fixture measurements and do not
establish a model-level P4 result.

The missing-cache path is tested as a local fixture dispatch that falls back
to the canonical reference. No engine integration or production fallback is
claimed.

## Commands (after explicit parent authorization)

```sh
CACHED_TC_BUILD_AUTHORIZED=1 ./build-v96.sh
CUDA_VISIBLE_DEVICES=GPU-... CACHED_TC_GPU_AUTHORIZED=1 ./run-v96.sh
```

The harness is not built or launched as part of host-only preparation. Running
the scripts without their authorization variables exits before build or GPU
work.
