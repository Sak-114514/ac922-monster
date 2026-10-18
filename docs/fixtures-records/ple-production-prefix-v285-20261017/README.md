# V285 production-shaped PLE prefix fixture

This fixture isolates the PLE prefix placement boundary reported in the
production model without loading model weights or running inference. It uses a
12-row synthetic IQ4_NL table with the V122 width-160 row layout, split across
six Meta members. The input reproduces 16 heads × 249 tokens (3,984 I32 row
IDs, repeated within 0..11), `GET_ROWS`, cast/scale-to-zero, reshape/repeat,
mirrored add, `[2560,249]` flatten, and one ordinary CUDA0 consumer.

Two modes are built into the same binary:

- `production`: only repeat-zeros and mirrored join are explicitly pinned to
  Meta6, as in `qwen4exp.cpp`; the row IDs, GET_ROWS, cast/scale/reshape, and
  final PLE reshape use scheduler placement. In the saved V280 run the row IDs
  landed on CPU but all intermediate operations landed on Meta6; the graph
  passed, so this tiny scheduler fixture did not reproduce the reported model
  assertion.
- `pinned-all`: explicitly pins row IDs and each PLE prefix intermediate to
  Meta6, while keeping the post-reshape consumer on CUDA0. This is the
  comparator for whether complete graph placement avoids the failure.

The fixture checks CPU/GPU output equality and input/output canaries if graph
execution completes. It proves only scheduler placement behavior for this tiny
graph, not full-model correctness or performance.

## Static check and host link

```sh
./check-v285-contract.py
./build-v285-host.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v280 v285-r2
```

The host-link script verifies the frozen V280 report identity and compiles the
fixture only; it does not initialize CUDA.

## Guarded GPU runs

Run only with the parent-granted bounded GPU window and six physical UUIDs in
index/PCI order. Each mode has a unique log, 60-second timeout, 58 C ceiling,
512 MiB minimum free-memory threshold, 2 GiB incremental memory cap per GPU,
and preflight check for other compute PIDs.

```sh
AC922_PLE_PRODUCTION_PREFIX_GPU_WINDOW_AUTHORIZED=1 \
CUDA_VISIBLE_DEVICES=<six validated physical V100 UUIDs> \
  ./run-v285-guarded.sh ./ple-production-prefix-v285-v285-r2 ./backend-v285-r2 \
  v285-r2-production-repro production assert

AC922_PLE_PRODUCTION_PREFIX_GPU_WINDOW_AUTHORIZED=1 \
CUDA_VISIBLE_DEVICES=<same six validated physical V100 UUIDs> \
  ./run-v285-guarded.sh ./ple-production-prefix-v285-v285-r2 ./backend-v285-r2 \
  v285-r2-pinned pinned-all pass
```

## Execution record

- Host link: frozen V280, build id `20261017T134510.041882Z`; report SHA is
  pinned by `build-v285-host.sh`. The successful binary/source receipt is
  `build-v285-r2.txt`.
- `v285-r2-production-repro.log`: production-only explicit pins did **not**
  reproduce the reported assertion. The scheduler placed the unpinned row ID
  tensor on CPU, while GET_ROWS, cast, scale, the row reshape, repeat-zero,
  join, and PLE reshape all landed on Meta6. The CUDA0 consumer completed and
  matched the CPU oracle exactly; input/output canaries held.
- `v285-r2-pinned.log`: pinning all prefix intermediates to Meta6 also passed
  exact CPU/GPU comparison and canaries. Observed placement matched the
  production-only case, so this tiny graph does not isolate a placement change.
- Both successful runs stayed at 38–40 C. Peak sampled incremental allocation
  was 442 MiB/GPU in production mode and 443 MiB/GPU in pinned-all mode; GPU
  usage returned to 4–27 MiB after each process exited.
- `v285-r2-production.log` is retained as the first guard-stop attempt: an
  initially aggregate 2 GiB cap stopped during CUDA context initialization at
  2,069 MiB total across six GPUs (about 413–418 MiB/GPU), before graph setup.
  The authorized cap was clarified as 2 GiB per GPU before the two completed
  runs above.

These results rule out reproducing the failure with this small scheduler graph
and V280; they do not rule out full-model graph placement, context input
allocation, or a different runtime interleaving as the source of the observed
production assertion.

## Direct foreign-CUDA VIEW regression

`meta-foreign-view-v285.cpp` adds an alloc-only CUDA0 `GGML_OP_VIEW` to the
actual Meta6 PLE graph's node list without making it a numerical consumer. Its
source is an op-NONE CUDA0 leaf with the same CUDA buffer. The PLE GET_ROWS,
zero/repeat/mirrored-join and final reshape/scale still execute and are checked
against an exact CPU oracle. This directly tests the Meta graph-node
materialization boundary implicated by the production backtrace.

- Frozen V280 host link: `build-foreign-v285-v280-r2.txt`.
- Old-runtime regression: `results/v285-v280-r2-foreign-assert.log`.
  It reproduced `GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer))` at
  `ggml-backend-meta.cpp:1172` after the log confirmed `op=VIEW`, source
  `op=NONE`, CUDA0 buffer, and no numeric consumer. The process exited 134;
  the guarded runner classified it as the expected reproduction.
- GPU temperature was 37–40 C; final memory returned to 3–22 MiB per card.

The first V288 attempt selected the new foreign-view path but then hit a second
trailing-subgraph assertion at `ggml-backend-meta.cpp:5942`; it is retained in
`results/v285-v288-foreign-pass.log`. The V290 source adds trailing no-op suffix
flush/accounting. The first V290 run then mismatched the CPU oracle because this
fixture allocated the Meta graph buffer before `ggml_build_forward_expand`
finalized node compute flags. The corrected fixture now builds all graph roots
first, then allocates the Meta tensors, matching the scheduler lifecycle.

The corrected V290 direct regression passed exact CPU/Meta parity. V290 build
id is `20261017T141349.669384Z`, with report SHA
`92de7803ff9a9f9889ee85c7756a25f480c76ec92628bb9e2cb8e327d8225d05`. Its
host-link receipt is `build-foreign-v285-v290-foreign-r2.txt`, and its result
is `results/v285-v290-foreign-r2-pass.log`. Six Meta allocator logs confirm
the CUDA-backed `VIEW` was preserved as a non-compute alias; the independent
IQ4_NL PLE graph still matched the CPU oracle exactly.

## Four-copy PLE plus CPU-token-embedding pipeline probe

`ple-production-prefix-v288-pipeline.cpp` extends the real `[2560,249]` PLE
prefix with a tiny CPU-resident F32 token-embedding table. CPU `GET_ROWS`
produces one `[2560,249]` tensor which joins the reshaped Meta6 PLE result at
the CUDA0 consumer. Twenty fresh graphs cycle four scheduler copy slots. Each
frame checks exact CPU/GPU output parity, repeated row IDs, the PLE reshape,
graph UID uniqueness, and host input/output canaries. It is a bounded routing
and lifetime probe, not a throughput test.

The pipeline runner enables `GGML_META_FIFO_PIPELINE`, both Meta event-wait
flags, PLE-Meta pinned input staging with a 256 MiB cap, Meta context views and
replay guard, `GGML_META_P2P_COLD`, and CUDA P2P; it explicitly unsets
`GGML_CUDA_P2P_AR_DISABLE`. Its pass gate requires the real cold-P2P selection,
Meta-to-CUDA event bridge selection with `host_wait=0`, four-copy pinned-input

The first V290 invocation used unsupported `GGML_SCHED_LAYER_PINNED_CAP_MIB=64`,
so pinned staging was declined; that log is retained at
`results/v286-v290-pipeline.log`. The corrected run uses the supported 256 MiB
cap and passes all four selected-path gates in
`results/v286-v290-pipeline-cap256.log`: cold P2P AllReduce, Meta-to-CUDA0 event
wait with `host_wait=0`, four-copy pinned-input staging, and CPU F32 token
embedding H2D staging (`2549760` bytes). The 20 fresh graphs cycled all four
copy slots, had unique UIDs, matched the CPU oracle exactly, and preserved all
canaries. Pinned memory high-water was `10199040` bytes with zero declines.
Temperatures stayed at 38–40 C; final GPU memory was 2–15 MiB per card with
zero utilization. This establishes selected paths and correctness only, not a
throughput result.

Host-link receipt for the pipeline is `build-v286-v290.txt`.

## Computed CPU GET_ROWS reshape view regression

`meta-computed-host-view-v291.cpp` covers the actual stricter case from the
V289 native canary: a CPU F32 table runs `GET_ROWS`, then a 4-D `RESHAPE` view
aliases that computed CPU result. The CPU producer graph is executed first.
For the Meta consumer graph, only the already-computed reshape view is appended
to `graph->nodes`; its `GET_ROWS` producer is explicitly absent. The fixture
updates the Meta graph's visited hash/use count directly, rather than expanding
the view recursively and accidentally including its CPU producer in the Meta
graph. A separate Meta6 IQ4_NL PLE graph supplies the exact CPU-output oracle.

Host-link once against frozen V290 and expect the old `get_simple` assertion;
then link against frozen V292 and require exact CPU/Meta parity:

```sh
AC922_PLE_PRODUCTION_PREFIX_GPU_WINDOW_AUTHORIZED=1 \
CUDA_VISIBLE_DEVICES=<six validated physical V100 UUIDs> \
./build-computed-host-view.sh /path/to/runtime-perf-v290 /path/to/V290-build.json V290_SHA V290_ID v291-v290
./run-v285-guarded.sh ./meta-computed-host-view-v291-v291-v290 ./backend-computed-view-v291-v290 \
  v291-v290-computed-host-assert computed-host-view assert

./build-computed-host-view.sh /path/to/runtime-perf-v292 /path/to/V292-build.json V292_SHA V292_ID v291-v292
./run-v285-guarded.sh ./meta-computed-host-view-v291-v291-v292 ./backend-computed-view-v291-v292 \
  v291-v292-computed-host-pass computed-host-view pass
```

```sh
./build-v286-pipeline-host.sh RUNTIME REPORT REPORT_SHA256 BUILD_ID TAG
AC922_PLE_PRODUCTION_PREFIX_GPU_WINDOW_AUTHORIZED=1 \
CUDA_VISIBLE_DEVICES=<six validated physical V100 UUIDs> \
  ./run-v285-guarded.sh ./ple-production-prefix-v286-pipeline-TAG ./backend-v286-TAG \
  v286-TAG-pipeline pipeline pass
```
