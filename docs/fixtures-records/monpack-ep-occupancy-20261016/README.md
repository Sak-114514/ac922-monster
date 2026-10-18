# Monpack E85 occupancy screen

This is a bounded, one-layer operator screen for the V75 Monpack grouped-TC
path. It uses the complete, original GGUF expert spans E[85:170] for only
`IQ3_XXS` and `IQ3_S`; every expert retains full K=2560 and N=640. No model is
loaded, no expert is pruned or requantized, and the fixture never changes the
source GGUF or sidecar.

## Why this shape

The earlier V52 TC work already repaired both formats numerically in the E=2
probe, but its T=16..128 wall measurements were slower than stock. This fixture
does not repeat that experiment. At T=1024 and top-2, it creates 2048 expert
rows across E=85 local IDs: 8 experts get 25 rows and 77 get 24. That workload
is close to the ~20-row/expert density expected around UB1024 in an EP6 layer,
while retaining complete K/N expert matrices.

The route IDs are fixture-local 0..84. They do not test production global-ID
translation, EP owner routing, foreign-zero behavior, Meta scheduling, or the
core's currently blocked Monpack+EP dispatch. Therefore a positive result only
keeps this operator shape eligible for a separately authorized integration
test; it does not qualify production EP or establish a speedup.

## Bounded source sampling

On AC922, run `./extract-ep85-samples.py --out ./samples`. It reads only the
GGUF and archived Monpack sidecar byte ranges for two tensors:

| Type | Tensor | Full shape `[K,N,E]` | Bytes per expert | Bytes for 85 experts |
|---|---|---:|---:|---:|
| IQ3_XXS | `blk.35.ffn_gate_exps.weight` | `[2560,640,512]` | 627,200 | 53,312,000 |
| IQ3_S | `blk.8.ffn_gate_exps.weight` | `[2560,640,512]` | 704,000 | 59,840,000 |

The script performs `pread` for each raw expert and corresponding packed expert,
reverses the Monpack `(N tile, K tile, d/code planes)` layout, and requires
byte-for-byte equality for every E before writing the samples. Payload reads
are 226,304,000 bytes; raw plus packed output files are another 226,304,000
bytes. The combined I/O/write cap is 450 MiB. GGUF metadata parsing does not
scan or decode the model payload.

The source-sample read plus sample storage stayed within that cap. During this
preparation I also ran the host hash validator five times while iterating on
the runner, causing 1,131,520,000 additional logical bytes of reads from the
small fixture files (not from GGUF or sidecar). If the cap is interpreted as
all reads including repeated integrity checks, cumulative logical I/O was
1,584,128,000 bytes; no further hash pass is planned.

Then run `./run-host.sh`. It builds and links the isolated C++ runner with the
existing `runtime-perf-v75` shared libraries using host `g++`; it invokes no
NVCC. Its `--host-plan` branch returns before `ggml_backend_cuda_init()` and
does not run a graph. The host check also verifies sample hashes/byte counts,
the 24/25 row occupancy schedule, and the 32-token selection for the FP64
accumulation oracle.

Before the authorized GPU attempt, a host-only `--host-plan` gate constructed
the exact 3D `[K,N,E]`, F32 activation, I32 top-2, `MUL_MAT_ID`, and graph
metadata for both types with `no_alloc=true`. The 1 MiB context arena passed
without initializing CUDA or allocating weight data.

## Prepared CUDA comparison

`monpack-ep-occupancy.cpp --cuda samples/` compares the full stock-CUDA and
registered Monpack outputs for all 1024 tokens and both route slots. It also
checks all outputs for finiteness and reports full-output stock-vs-Monpack
max-absolute error/RMSE. Separately, it compares selected tokens 0..30 plus 41
(32 tokens total) × 16 output rows to an FP64-accumulation reference using
GGML's public row dequantizer on the original GGUF-format bytes. This keeps CPU
oracle work bounded; a full CPU FP64 reference over all routed rows would
require billions of multiply accumulates. It also records three repeated
host-wall measurements around graph compute plus synchronization for stock and
Monpack. Those times are screening data, not CUDA-event kernel time or a speed
claim. No numeric tolerance is silently added.

The GPU script refuses to execute unless the root explicitly sets
`MONPACK_EP_OCCUPANCY_GPU_GRANT=root-authorized` and selects exactly one
assigned device in `CUDA_VISIBLE_DEVICES`. It pins the TC threshold to 16,
checks that the selected device has no compute process, and caps a run at 50
seconds. Even after a GPU result, the local-ID limitation above remains.

## Bounded GPU3 screen

The first authorized attempt (`results/cuda-screen-20261016T050145Z.log`) failed
before tensor dispatch: the GGML context arena was too small for graph metadata
(84,080 bytes requested, 23,552 available), followed by a fixture segfault.
The fixture was corrected to use a 1 MiB no-allocation metadata arena and gained
the host graph-setup gate above. No numeric assertions or tolerances changed.

One corrected GPU3 run completed at `2026-10-16T05:03:27Z` and is retained in
`results/cuda-screen-20261016T050327Z.log`. All full outputs were finite; no
quality tolerance was applied. The full-output column compares stock CUDA with
Monpack. The selected-oracle columns compare 32 tokens × 16 rows against GGML
row dequantization plus FP64 accumulation. Timing is three repeated host-wall
measurements of graph compute plus synchronization, not CUDA event time.

| Format | Full-output stock/Monpack max abs / RMSE | Selected stock-oracle max abs / RMSE | Selected Monpack-oracle max abs / RMSE | Mean stock / Monpack ms | Monpack/stock |
|---|---:|---:|---:|---:|---:|
| IQ3_XXS | 0.00396243 / 0.00065938 | 0.00214730 / 0.00064431 | 0.00020584 / 0.00005159 | 1.0358 / 8.9120 | 8.60× slower |
| IQ3_S | 0.00324860 / 0.00064053 | 0.00191917 / 0.00062827 | 0.00017972 / 0.00005015 | 1.0097 / 9.8665 | 9.77× slower |

The selected oracle slice is numerically closer for Monpack in both formats,
but that limited result is not a full-output quality qualification. The 85-local
expert workload is substantially slower than stock in this screen, so this
Monpack TC path has no performance case at this shape. No production EP6 route
was exercised.

Dispatch evidence is bounded: the logs show registered weight pointers and
`mul_mat_id` hook calls with `reg=1`; the test geometry satisfies the source's
TC thresholds (`T=1024`, `M_total=2048`, local E=85), and no TC work-list
allocation failure was logged. The V75 implementation has no per-call
`mp_run_tc_moe` success marker, so the actual TC branch completion was not
directly instrumented; the oracle/output behavior is consistent with the
Monpack path but does not replace such a marker.

## Current status

- Extraction and exact byte proof: PASS for 170 raw/packed expert pairs.
- Host-only compile/link and occupancy checks: PASS with V75, using `g++` and no NVCC.
- Host graph-setup gate: PASS for both formats with a 1 MiB metadata arena.
- CUDA operator screen: one corrected GPU3 run completed; numerical and timing
  metrics are recorded above. A first fixture-only allocation failure is also
  retained; no further GPU run is authorized by this record.
- Production EP integration and performance: not tested.

The final host graph-setup log is
`results/host-graph-setup-20261016T050411Z.log` on AC922. Its executable SHA-256
was `077b7200f360c21ad445875ff6c6f7a16abf0c381863f09915175e4c8e79be08`.
The V75 CUDA library hash was
`7d4cdf24f0c8463d7063452df25f78e214aaa0c2b054af3d3af63cc5d2d2d5c5`; the
initial host sample-validation log is `results/host-20261016T045547Z.log`.
The corrected CUDA run left GPU3 at 3 MiB used, 0% utilization, and 38–39°C on
the immediate fresh check; the BMC watcher remained fresh with all three main
fan targets at 3000 RPM, no alarm/fault, `masks_ok=true`, and maximum observed
temperature 40.125°C. No BMC/host thermal settings were changed.
