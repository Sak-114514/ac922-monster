# Monpack grouped-TC operator probe

This is an isolated copy of the operator fixture for the V49 grouped-TC path.
It reads two complete original expert slices (global E=170 and E=171), keeps
full K/N, routes local IDs 0/1 with `n_used=2`, and covers T=16/32/128 for all
seven packed formats. The two experts together contain 8,268,800 raw bytes and
the same number of packed bytes. No expert is pruned or requantized.

## Prepare original samples and CPU reference

Run on AC922 from this directory:

```sh
./prepare-samples.sh
./run-cpu-reference.sh
```

`prepare-samples.sh` invokes the copied source extractor once per expert. Each
run reads only seven full-K/N expert slices from the source GGUF and matching
sidecar, reverses the sidecar byte permutation, and requires byte equality with
the raw GGUF slice. The combined manifest records the global IDs and fixture
local IDs. It does not load a model or initialize CUDA.

The host harness compares three results:

- CPU `MUL_MAT_ID` baseline using V49's existing CPU backend.
- A high-precision reference: call GGML's public per-row `to_float` dequantizer
  on the original raw expert bytes, then dot the F32 activation with those
  dequantized F32 weights using FP64 accumulation.
- On the authorized GPU run, stock CUDA and Monpack CUDA results against that
  same reference and against each other.

For each type/T it records finite status, max-absolute error, max-relative
error, RMSE, reference max-absolute value, and reference L2 norm. CPU baseline
error is reported separately; the CPU backend's activation quantization is not
assumed to match CUDA.

## Expected TC selection

With `n_used=2`, the grouped row counts are `m_total=2*T` = 32, 64, and 256.
These cases exceed `MMVQ_MAX_BATCH_SIZE=8` and the runner pins
`GGML_CUDA_MONPACK_TC_MIN_M=16`, so V49's source dispatch selects
`mp_run_tc_moe()` when its per-device TC buffers were reserved. Registration
logs and the expected branch are recorded, but V49 has no success-return or
kernel-launch marker for every TC call. If its TC buffers are null,
`mp_run_tc_moe()` returns false; that return can fall through to the stock
kernel, which would interpret packed bytes as ordinary GGML bytes. Treat the
per-format numeric reference comparison as the gate and preserve any failure.

This is a deliberately compact two-expert case, not a production route-density
model. The production D3 verify path has two banks of eight tokens, top-10 of
512 experts, and EP3; each rank's active expert commonly receives about one
row. Here every token routes both slots over only two local experts, so
`m_total` is 32/64/256 and each used expert is much denser. V49 selects TC by
total `m_total>=16`, not by per-expert occupancy. This probe can check TC
numerics and low/medium token-batch behavior; its timing cannot be extrapolated
to sparse production decode. A future E>=32 schedule-only probe could repeat a
raw expert byte-identically across local E slots to study sparse work-list
overhead, but that would be synthetic routing and must not be presented as
model quality or production throughput evidence.

The V49 work-list scratch `g_tc[device]` is a process-global array indexed by
device, not context-local storage. The mutex protects reservation, not per-op
use. Safe reuse therefore depends on MetaFIFO serializing work on the same
device; this one-context, synchronize-after-each-graph fixture does not prove
multi-context concurrency safety.

## Authorized GPU3 command

Wait for the root-assigned GPU3 window, then run:

```sh
MONPACK_TC_GPU_WINDOW=3 CUDA_VISIBLE_DEVICES=3 ./run-cuda-tc-gpu3.sh
```

The runner refuses any other window tag or visible device, checks that physical
GPU3 has no compute process, uses the existing V49 shared libraries with a
host `g++` build (no NVCC), and caps the kernel workload at 50 seconds plus a
3-second kill grace, leaving time for preflight and setup inside the 60-second
grant. `MONPACK_TC_TIMEOUT_SEC` is limited to 1..50 seconds. It explicitly
selects stock defaults (`GGML_IK_FUSED_MOE=0`, `GGML_CUDA_IQ2S_VOLTA=1`) and
records a unique log in `results/`.

The one-output run and each of 100 repeated runs by default are timed with
`steady_clock` around `ggml_backend_graph_compute()` plus
`ggml_backend_synchronize()`. `MONPACK_TC_TIMING_REPEATS` can reduce the repeat
count to 1..100 if the authorized window needs it, and the chosen count is
logged. These are host completion-wall measurements, not CUDA event timings.
They include launch and synchronization overhead; do not present them as
direct kernel duration or a performance claim.

The sample uses local expert IDs 0/1 to select the E170/E171 fixture slices.
It does not prove production global-ID/owner-offset mapping, foreign-zero
behavior, EP history/rollback, fused gate/up, or concurrent backend-context
safety. No result may be called production EP support.

## Recorded GPU3 run

- UTC start: `2026-10-15T21:19:28Z`; physical GPU3 was a Tesla V100-SXM2-16GB
  at 38 C, 0 MiB used, with no compute process. After completion it was 40 C,
  0 MiB used, with no compute process.
- V49 `libggml-cuda.so.0` SHA-256:
  `614cc8685ba9d8c2f5a998a34af330609d7a372d7e764d07052c4ef3d808e7d1`.
  The host harness compiled with `g++`; no NVCC or model load.
- All 21 type/T outputs were finite. The runner's exit status 0 means metrics
  completed and values were finite; it does not certify a numerical tolerance.
  Raw log: `results/tc-gpu3-20261015T211928Z.log`, SHA-256
  `eca5cfb2448a47a23a5dc92149b3ce0b1babbd8abe69b341c4359ae6998dee40`.
- The raw stdout/stderr stream contains interleaved CUDA-graph warmup messages.
  The derived JSON and CSV remove only those exact diagnostic lines; the raw
  log remains unchanged. JSON: `results/tc-gpu3-20261015T211928Z-metrics.json`.
  JSON SHA-256 `c939d197399b6cab4a201f5725fd2af52d5799cc95357f84093cbbe589008b7f`.
  CSV: `results/tc-gpu3-20261015T211928Z-summary.csv`, SHA-256
  `13fe8a134f40ce7449925092796cc8121af98744f0a43183de9e7b27852df953`.
- CPU reference log: `results/tc-cpu-reference-20261015T211813Z.log`, SHA-256
  `29a8969cf5a861d963c207205fefbbf906140c3bd5358c41c6fb2ef13e56410c`.

Each range in the table is across T=16/32/128. `TC max_abs` and `TC RMSE` are
against the per-case GGML-dequantized FP64-dot reference. Wall ratio is the
Monpack/stock mean over 100 host-completion-wall repeats per case.

| Format | CPU baseline max abs | Stock CUDA max abs | Monpack TC max abs | Monpack TC RMSE | Repeat wall ratio |
|---|---:|---:|---:|---:|---:|
| IQ2_XXS | 0.00298–0.00357 | 0.00320–0.00340 | 1.25–1.31 | 0.241–0.242 | 8.91–10.14× |
| IQ2_XS | 0.00364–0.00420 | 0.00339–0.00403 | 0.000281–0.000316 | 6.27e-5–6.28e-5 | 5.72–6.42× |
| IQ2_S | 0.00312–0.00312 | 0.00284–0.00328 | 0.000213–0.000235 | 5.13e-5 | 6.51–7.80× |
| IQ3_XXS | 0.00296–0.00324 | 0.00289–0.00360 | 0.000214–0.000256 | 5.19e-5–5.22e-5 | 7.21–8.10× |
| IQ3_S | 0.00282–0.00304 | 0.00248–0.00288 | 0.46369 | 0.1046–0.1047 | 8.80–9.00× |
| IQ4_NL | 0.00227–0.00260 | 0.00212–0.00272 | 0.000148–0.000167 | 2.56e-5–2.58e-5 | 2.31–6.93× |
| Q2_0 | 0.00139–0.00169 | 0.00139–0.00169 | 8.34e-5 | 1.67e-5–1.68e-5 | 2.04–5.98× |

Per-case finite/max-rel/RMSE/reference max-abs/L2 values remain in the raw log
and derived JSON. Max-relative error can become large when a reference element
is close to zero; use max-abs and RMSE alongside it. The Monpack TC errors for
IQ2_XXS and IQ3_S are grossly above stock and the high-precision reference.
Keep those two formats blocked pending a source fix and a fresh isolated rerun.
IQ2_XS, IQ2_S, IQ3_XXS, IQ4_NL, and Q2_0 are close to the FP64 reference in
this fixture, but their host-completion-wall means are slower than stock in
these measurements. These results are not CUDA-event kernel times or production
throughput evidence.

The source-level branch is expected to select `mp_run_tc_moe()` for all cases:
T exceeds 8, `m_total=2*T` is 32/64/256, the runner pins `tc_min=16`, and the
register log reports all seven `[K,N,2]` tensors without a TC-buffer reservation
failure message. V49 has no per-call TC success marker; the raw log preserves
the capped hook records, not an explicit kernel-return trace.

The root source review identified candidate arithmetic/indexing defects for
later core repair: IQ2_XXS reads a 32-bit auxiliary word from `code + 4*ib32`
although the byte-pointer stride should be `8*ib32`; IQ3_S indexes `qh` with
`64 + (ib32 & 1)` where the per-block layout requires `64 + ib32`. This
follow-up did not change either kernel. Root also identified a signed-char
promotion risk in IQ4_NL's high-nibble decode; this V100 run is close to the
reference for IQ4_NL and does not show that issue triggering here.

## V52 regression lane

V49 artifacts above are immutable. V52 uses the same samples/harness but a
separate runtime, script, and `results/v52/` directory. The build evidence is
`notes/V52-build.json`: build ID `20261015T212314.156656Z`, no blocked items,
only `ggml/src/ggml-cuda/monpack/monpack-kernels.cuh` changed since the prior
runtime, and one incremental `monpack-kernels.cu.o` compile. Its CUDA compile
plan uses NVCC `-O3 -DNDEBUG -std=c++17`, `compute_70/sm_70`,
`-DMONPACK_AVAILABLE=1`, and `-use_fast_math`; the compile result return code
is 0.

V52 library hashes are: `libggml-cuda.so.0`
`0b88451d7342dfe4f5f3d1dcbe5896fa0ba358558545251b34a0bc1df0c2f290`,
`libggml-cpu.so.0`
`c1fd8d19e54df03e97f1c71c8c95565626ec73f44f7d4f3e334bfe3e74f464fc`,
`libggml.so.0` `73ddff0764ae9545cc0cc561baffee5909c820e13e07060cc1bc3ed7314814e2`,
and `libggml-base.so.0`
`e6578aca8f533f14feb05dfc18588330b1dfc8a38dec2003899b5e6592f16c17`.
The changed `monpack-kernels.cuh` SHA-256 is
`e05e0eff2b6c04dd3cefb532b989978d9f2cf8a3f3f2f9733054c1267bd09c57`.

V52 host link completed with `LLAMA_RUNTIME_LIB` and `LD_LIBRARY_PATH` both
pointing at `runtime-perf-v52/lib`. `ldd` resolves the GGML CUDA/CPU/base
libraries from V52 and the executable RUNPATH is V52. The persistent host-link
log is `results/v52/hostlink-20261015T212915Z.log`; the probe was linked with
ordinary `g++` and was not run against CUDA during host-link verification.

The V52 GPU3 window completed at `2026-10-15T21:30:12Z`, using ten timing
repeats per stock/Monpack case. Preflight was 39 C / 0 MiB / no GPU3 process;
the post-run check was 39 C / 1 MiB / no compute process. `runner_exit_status=0`
means the 21 cases produced finite metrics; it is not a quality-threshold pass.

The raw V52 log is `results/v52/tc-gpu3-v52-20261015T213012Z.log`, SHA-256
`33574d3d7590e91f804c111a265837ab5e917a110f3bc1ecb99762484a6d8a80`; stderr
and dispatch diagnostics are separate at
`results/v52/tc-gpu3-v52-20261015T213012Z.stderr.log`, SHA-256
`2a8c24062a938cbf483d5a055200079773874a40a53d75d5e12002330cf6e218`.
The link log is `results/v52/hostlink-20261015T212915Z.log`, SHA-256
`fef2c3757153b7aec331a5cdada94ca71a45d677e09382de86d6a1ea34465160`.

Per-case max-relative error, RMSE, reference max norm, and timings are retained
in the V52 log and the derived metrics files. Each range below spans T=16/32/128.

| Format | CPU baseline max abs | Stock CUDA max abs | V52 Monpack TC max abs | V52 TC RMSE | Repeat wall ratio |
|---|---:|---:|---:|---:|---:|
| IQ2_XXS | 0.00298–0.00357 | 0.00320–0.00340 | 0.000238–0.000268 | 5.19e-5–5.22e-5 | 7.60–8.98× |
| IQ2_XS | 0.00364–0.00420 | 0.00339–0.00403 | 0.000281–0.000316 | 6.27e-5–6.28e-5 | 5.14–5.66× |
| IQ2_S | 0.00312–0.00312 | 0.00284–0.00328 | 0.000213–0.000235 | 5.13e-5 | 3.94–6.91× |
| IQ3_XXS | 0.00296–0.00324 | 0.00289–0.00360 | 0.000214–0.000256 | 5.19e-5–5.22e-5 | 6.54–7.38× |
| IQ3_S | 0.00282–0.00304 | 0.00248–0.00288 | 0.000199–0.000221 | 4.95e-5 | 5.39–7.84× |
| IQ4_NL | 0.00227–0.00260 | 0.00212–0.00272 | 0.000148–0.000167 | 2.56e-5–2.58e-5 | 1.98–5.91× |
| Q2_0 | 0.00139–0.00169 | 0.00139–0.00169 | 8.34e-5 | 1.67e-5–1.68e-5 | 1.81–5.47× |

The V52 changes restore the IQ2_XXS and IQ3_S operator outputs to the same
small-error range as the other formats in this fixture. For all 21 format/T
cases, Monpack TC max-absolute error and RMSE are below the corresponding stock
CUDA error against the same high-precision reference. Max-relative errors can
be inflated by near-zero reference elements, so retain the raw per-case
max-abs, RMSE, and reference norms when reviewing; no new tolerance was added.
TC completion-wall repeats remain slower than stock in this two-expert shape.
This confirms a narrow numerical repair, not a performance win or production
decode benefit. The older strict GEMV bitwise gate and its V49 result are
unchanged by this TC test.

The V52 runner is ready for a future authorized GPU3 window at:

```sh
MONPACK_TC_V52_GPU_WINDOW=3 \
CUDA_VISIBLE_DEVICES=3 \
LLAMA_RUNTIME_LIB=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v52/lib \
LD_LIBRARY_PATH=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v52/lib:/usr/local/cuda-12.4/targets/ppc64le-linux/lib:/usr/local/cuda/lib \
./run-cuda-tc-gpu3-v52.sh
```

The V52 runner defaults to 10 repeats, pins the same TC threshold and stock
IQ2_S settings, validates the V52 dynamic-library resolution, and writes only
under `results/v52/`. It retains the 50-second workload cap and distinct V52
log tag so it cannot overwrite or replace the V49 result.
