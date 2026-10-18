# Meta3 P2P64 versus NCCL F32 route fixture

This fixture is a route-selectable copy of `hardware-ar-grid-20261016`'s
deterministic Meta3 graph: Q8_0 `MUL_MAT_ID` partials, AllReduce, then RMSNorm.
It keeps the five distinct target sizes (200 KiB, 400 KiB, 4 MiB, 16 MiB,
32 MiB), a repeated 200 KiB shape, and the `{1,1,0}` empty-rank check. Both
route arms use the same inputs, CPU oracle, rank checks, canaries, warmups, timed
round order, and full `graph_compute + synchronize` wall timing. Each output
also gets an FNV-1a hash so the runner can require bitwise equality across arms.

## Route selection from existing controls

Source inspected on AC922 under:

`/home/debian/ac922-monster/experiments/standard-opt-20261014/src`

For a flat Meta3 with simple CUDA members, the meta backend initializes the
ordinary communication context and P2P AllReduce independently. Its fallback
tries the P2P callback first and then `comm_allreduce`. The existing controls
are sufficient to compare the two boundary routes without changing core code:

- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_backend_cuda_comm_init()` selects the
  communicator from `GGML_CUDA_ALLREDUCE`; `ggml_backend_cuda_comm_allreduce_nccl()`
  checks `GGML_CUDA_NCCL_F32` before it chooses the wire dtype.
- `ggml/src/ggml-cuda/allreduce-p2p.cu::ggml_cuda_p2p_ar_init()` returns null
  when `GGML_CUDA_P2P_AR_DISABLE` is present.
- `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_context` creates the comm
  context and P2P owner separately. With `GGML_META_P2P_COLD=1`, its compute
  path tries the cold P2P callback first, then the existing comm fallback.

- `p2p64`: P2P AR enabled, `GGML_CUDA_P2P_AR_BLOCKS=64`, and
  `GGML_META_P2P_COLD=1`.
- `nccl-f32`: `GGML_CUDA_P2P_AR_DISABLE=1`,
  `GGML_CUDA_ALLREDUCE=nccl`, and `GGML_CUDA_NCCL_F32=1`. Disabling the P2P
  AR owner leaves the Meta3 communicator fallback active. `nccl_wire_f32` must
  report `explicit_f32_no_bf16_scratch`, and the run rejects any P2P AR launch.

Each arm runs in its own process, so route and F32 settings are fixed for the
communicator's lifetime. The runner compares output hashes after both processes
finish. The two runs use separate log files under one unique tag.

## Safe threshold design for a future default-off core change

The existing global P2P-disable control is enough for this fixture, but it
cannot keep decode-sized tensors on P2P while routing only large prefill
reductions through NCCL. If paired measurements justify that split, the narrow
process-static control is `GGML_META_PREFILL_NCCL_MIN_BYTES`, default `0` (off):

1. In `ggml-backend-meta.cpp`, evaluate the threshold from the F32 contiguous
   partial bytes while deciding whether `try_merge` can add P2P
   `GGML_OP_ALLREDUCE` nodes (the `max_bytes`/`p2p_ar_reserve` block around lines
   4560-4621). For an admitted large group, decline the P2P merge before any
   P2P node is written into or captured with the child graphs.
2. At the later boundary dispatch (around lines 4959-4982), decline the cold
   `p2p_ar_allreduce` callback for the same large group and let the existing
   `comm_allreduce` callback handle it. Keep the small group on the existing
   P2P64 path. Require flat Meta3, three CUDA children, contiguous F32 tensors,
   `GGML_CUDA_ALLREDUCE=nccl`, and `GGML_CUDA_NCCL_F32=1` for the large route.
3. Leave `ar->blocks` and the P2P epoch state fixed for that communicator. The
   threshold is parsed once, never toggled during replay, and every rank must
   take the same payload-based branch. NCCL stays a host-side group call from
   the Meta boundary, outside per-rank workers and outside captured CUDA graphs.

The existing `ggml_backend_cuda_comm_context` already owns the NCCL communicator;
the separate P2P context owns its epochs. The default-off threshold would not
switch a communicator's route at runtime. This is a design note only; no core
source was changed or patched in this fixture task.

Both arms set `GGML_CUDA_DISABLE_GRAPHS=1`. They exercise the same GGML Meta3
operator graph and collective boundary, but they do not capture NCCL in a CUDA
Graph. This fixture therefore compares the P2P and NCCL host-boundary routes;
it does not establish NCCL CUDA-Graph capture safety or full-model performance.

## Host build and CPU contract

On AC922, build/link against the frozen runtime (default V56) and run the
CPU-only geometry/oracle contract:

```bash
cd /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nccl-prefill-route-20261016
RUNTIME_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v56 ./build-host.sh
```

This does not initialize a GPU or load a model.

## GPU comparison only after root authorization

The runner requires a root token, verifies that GPU0–2 UUIDs match and are
unused, checks memory/temperature, and limits each route process to 15 seconds.
After an exclusive GPU0–2 window is granted:

```bash
cd /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nccl-prefill-route-20261016
NCCL_ROUTE_GPU_AUTH=ROOT_NCCL_ROUTE RUNTIME_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v56 ./run-route-comparison.sh
```

The script keeps complete per-route logs, checks the P2P64 dispatch counters and
the NCCL F32 marker, compares all final-output hashes, and refuses to overwrite
an existing run directory. The paired GPU result from the authorized run is
documented in
`results/20261015T224016Z-67151/ANALYSIS.md`; it is a synthetic Meta3 graph
comparison, not a full-model result.

## Paired run and threshold reading

The authorized V56 GPU0–2 run passed CPU-oracle checks for all 21 outputs per
route and all three ranks' consistency checks. The two route arms produced
identical final-output hashes. The route evidence shows 114 P2P launches at
grid 64 for P2P64; the NCCL arm reported the `nccl_wire_f32` path with reason
`explicit_f32_no_bf16_scratch`, zero P2P launches, and `nccl_f32_required=1`.
Full per-payload measurements, retained log hashes, and the interleaved-line
note are in `results/20261015T224016Z-67151/ANALYSIS.md`.

The complete graph-compute plus Meta synchronization median was slower for
NCCL at 213,044 B and 409,700 B, and faster at 4,211,716 B, 16,797,700 B, and
33,579,012 B. A 4 MiB minimum-byte threshold is a conservative initial
candidate for a future default-off source change; the crossover itself was
not measured between 409,700 B and 4,211,716 B. The measurements have graph
capture disabled and establish neither capture safety nor full-model benefit.

If the threshold is implemented, it must be a process-static decision made
before the P2P merge/capture path and honored again at the later Meta boundary:
large eligible F32 collectives decline P2P and use the existing grouped NCCL
communicator, while smaller collectives keep the fixed P2P64 path. Do not
change the P2P grid/epoch or switch routes during communicator replay. The
fixture has not changed core source.

## Source provenance

At preparation time, current remote SHA-256 values were:

- `ggml/src/ggml-cuda/ggml-cuda.cu`: `0a591c48a351e86ab3b1a6e2a42e1add4e0e48fe260d6f1ee61de6004abb5883`
- `ggml/src/ggml-cuda/allreduce-p2p.cu`: `25d3619a3fbd518c06949698e2f886cd3066ea9690d39cd2dc54e907d5ddf0af`
- `ggml/src/ggml-backend-meta.cpp`: `cea773c4632c54bfc00cee63a1fe02cdc46d2fabc7d5437189489bed185a7131`

Existing V53 P2P64 whole-graph medians are retained in
`fixtures/hardware-ar-grid-20261016/results/v53/RESULTS.md`; they are historical
reference only. The paired V56 route run uses one identical runtime for both
arms; its results remain specific to this synthetic graph and GPU window.
