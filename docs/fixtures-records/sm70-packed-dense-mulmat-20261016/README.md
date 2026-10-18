# SM70 packed dense MULMAT shape fixture

This fixture checks the V89 opt-in `GGML_SM70_DECODE_LINEAR_PACKED` branch on
ordinary 2D quantized `ggml_mul_mat` weights. It is intended to isolate a
possible batch-shape effect reported at T32. It does not assign expert IDs,
construct an EP tensor, load a GGUF, prune a model tensor, or claim that a raw
expert sample retains its original model semantics after being treated as a
standalone matrix.

## Inputs and shapes

The Q5_K arm reuses the saved first 2,048 rows of `output.weight`, with
`K=2560`, `N=2048`, and SHA-256
`37f8db597eb0a511b2a968bf7fa6da30cc2c2c6b6a31a3305d4658770645ebc1`. It
tests T32, T24, T48, T64, then retains the V87 lower-shape cases T4, T6, T8,
T12, and T16. T32 is first so the CUDA path logger's small detailed-record
limit can show the target shape.

The seven original expert-170 raw samples are reused byte-for-byte as
standalone 2D weight matrices: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL,
and Q2_0. All seven have source samples and each is checked at T32, T24, T48,
and T64. Their K/N dimensions match the original tensor metadata, including
the reversed dimensions for the two down-projection samples. The existing
sample manifest verifier checks hashes without scanning the GGUF. These are
original quantized bytes, not the model's complete dense weights.

Each graph uses deterministic F32 input A, a changed F32 input B, then restored
A. It verifies input readback, finite outputs, canaries, A/B/A exact replay,
and unchanged weight bytes. OFF and ON are separate processes because the
runtime caches the environment choice. A CPU comparator requires every raw F32
output and both input arrays to match byte-for-byte across arms. The parser
also records whether the packed T32 dispatcher marker appeared per dtype; a
parity result with no selection marker is reported as a tested fallback, not
as packed-path qualification. Q5_K records the same 40-point FP64 comparison
used by the V87 Q5 fixture for each A/B/A frame.

No performance gate is included. A fixture-only result cannot identify the
cause of the V38 model failure or qualify a model run.

## Host link

V89 build: `20261016T101245.741848Z`; runtime:
`/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v89`.
The host build pins the frozen V89 shared-library and GGML API-header hashes.
It compiles only this fixture and does not rebuild engine code. It does not use
live CUDA source hashes as link prerequisites.

```sh
./fixtures/sm70-packed-dense-mulmat-20261016/build-host.sh host2
```

Run the complete host entry check once after linking. It exercises T32 for all
eight formats on the V89 CPU backend, verifies A/B/A, output canaries, sample
dimensions, and unchanged weights, and does not initialize CUDA:

```sh
BASE=/home/debian/ac922-monster/experiments/standard-opt-20261014
FIX="$BASE/fixtures/sm70-packed-dense-mulmat-20261016"
OUT="$FIX/results/host-entry-v89"
mkdir -p "$OUT"
LD_LIBRARY_PATH="$BASE/runtime-perf-v89/lib" \
  "$FIX/sm70-packed-dense-mulmat-v89-host2" --host-check \
  "$BASE/runtime-perf-v89" \
  "$BASE/fixtures/q5k-v85-chunk4-hc-20261016/output-q5k-first2048.raw" \
  "$BASE/fixtures/monpack-ep-readiness-20261016/operator-fixture/samples" "$OUT"
```

## Bounded GPU run

GPU execution requires an explicit root grant and one assigned V100 UUID. The
runner preserves a 512 MiB free-memory floor, a 512 MiB added-used cap, and a
58 C ceiling. It writes unique tagged logs and separates stdout from CUDA
stderr before parsing, so backend diagnostics cannot split fixture records.
It does not change fan or BMC state.

```sh
ROOT_SM70_PACKED_DENSE_GPU_WINDOW_AUTHORIZED=1 \
GPU_UUID=GPU_ID CUDA_VISIBLE_DEVICES=GPU_ID \
./fixtures/sm70-packed-dense-mulmat-20261016/run-bounded.sh "$(date -u +%Y%m%dT%H%M%SZ)"
```

No GPU run has been authorized by this fixture's preparation record.
