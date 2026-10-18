# Shared Meta3 scheduler-owner and TP AllReduce fixture

This real GGML/CUDA harness targets frozen runtime-perf-v49. It creates one
immutable F32 WEIGHTS buffer and two independent Meta3+CPU scheduler/backend
pairs over the same three CUDA members.

The MIRRORED phase uses a dynamic strided view of one shared root followed by
CONT -> MUL_MAT -> SQR -> SUM. It runs A/B/A sequentially and rebuilds B's graph
after scheduler reset to demand same-address view reuse with a new owner epoch.

The TP-AR phase is separate. W0 and x0 split the matmul K dimension over the
three Meta members. MUL_MAT is PARTIAL, then ADD(mirrored bias, partial)
requires the full result and exercises Meta's inserted P2P AllReduce. Two
independent schedulers submit this graph concurrently and compare exact
integer-valued F32 results.

A bounded V49 lockstep regression forces a split after the TP AllReduce with GGML_META_CHUNK_NODES=1, then serially warms A's new TP shape once, then races the second (merge/cold-chunk) P2P-AllReduce graph_compute against cold B contiguous-view alloc/query. It records the host API interval overlap and requires it to be positive. The run wrapper must impose a 30-second timeout.

The executable also checks B scheduler/backend destruction while A and the
single shared static weight buffer remain alive, followed by another exact A
dispatch. Console identity lines print the shared root/buffer and distinct
scheduler/backend/view pointers; V48 Meta INFO trace provides opaque owner IDs.

Build only with ./build.sh. The build compiles and links and does not execute
the fixture. Do not run until a GPU window is assigned. For a run, expose only
the verified three-GPU NVLink island, set GGML_META_CONTEXT_VIEWS=1 and
GGML_META_CONTEXT_VIEWS_TRACE=1, keep P2P AllReduce enabled, and do not enable
Meta replay-guard tracing. This is not a performance benchmark.

