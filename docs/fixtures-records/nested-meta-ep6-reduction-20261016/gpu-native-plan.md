# Nested Meta2x3 GPU-native reduction bridge

Status: source-backed proposal plus CPU contract check only. No core source, engine build, or GPU was touched for this artifact.

## Decision

Use one flattened six-rank CUDA collective at the outer Meta2 boundary. Prefer the existing CUDA P2P-AR callback with ranks ordered `[islandA0,islandA1,islandA2,islandB0,islandB1,islandB2]`: this tree already has a six-rank relay and its P2P reducers add FP32 values in ascending rank order. If six-way peer access cannot initialize, use the existing six-rank NCCL communicator with `GGML_CUDA_NCCL_F32=1`, and accept it only against the existing strict numeric tolerance; NCCL does not promise the same left-to-right addition order. Do not sum two island totals and then add them if rank-order equivalence is required.

The second option is a two-stage inner-P2P3 plus outer two-island reduction. It is GPU-native, but its FP32 order is `((r0+r1)+r2)+((r3+r4)+r5)`, which is not rank-order `((((r0+r1)+r2)+r3)+r4)+r5`. `contract_cpu.py` contains a cancellation witness where those results differ by 3.0. It is a valid later candidate only if the established tolerance explicitly admits its error.

## Source evidence

All line numbers below refer to the frozen V54 source tree at `/home/debian/ac922-monster/experiments/standard-opt-20261014/src` and should be rechecked before integration.

- `ggml/src/ggml-backend-meta.cpp:3439-3458` creates an outer context's immediate children; `3485-3525` explicitly disables outer collectives for nested members and initializes NCCL/P2P only for flat children.
- The normal boundary dispatch is `ggml/src/ggml-backend-meta.cpp:4950-4988`: child subgraphs are submitted, then a collective is attempted, then `allreduce_fallback(i)` runs on decline.
- The generic fallback at `4834-4947` copies output tensors between immediate children. On nested Meta outputs carrying `local_partial_axis=12`, this enters `ggml_backend_meta_buffer_get_tensor` (`2939-3036`), whose default deliberately aborts for axis 12. Returning child 0 as mirrored would multiply or discard partials; that is not a fix.
- Recursion is available through the exported `ggml_backend_meta_n_backends` / `ggml_backend_meta_simple_backend` APIs (`ggml/src/ggml-backend-impl.h:102`) and `ggml_backend_meta_buffer_get_n_simple` / `ggml_backend_meta_tensor_get_simple` (`ggml/include/ggml-backend.h:419-422`). Repeated calls expose the six physical CUDA backends and six physical output tensors without `tensor_get` or D2H.
- CUDA exports `ggml_backend_comm_init/free/allreduce_tensor` and `ggml_backend_p2p_ar_init/free/reserve/allreduce` from `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_backend_get_proc_address`, around 8904-8931). `ggml_backend_cuda_comm_init` validates direct CUDA children and records their order (around 1955-1994). NCCL uses F32 only when `GGML_CUDA_NCCL_F32=1` (around 1591-1676); the flag is captured on first call, so set it before the first reduction.
- `ggml/src/ggml-cuda/allreduce-p2p.cu:449-509` admits up to 8 devices and requires pairwise peer access. Its six-rank `relay6` reducer (`150-211`) sums ranks 0..5 left-to-right and copies the completed result through the 3+3 relay. The ordinary one-shot and segmented reducers also sum ranks in ascending order (comments around 98-120 and 221-245). The cold CUDA wrapper (`ggml-cuda.cu:1843-1887`) accepts a flat list of direct CUDA backend/tensor pointers, enqueues all rank kernels, then synchronizes all six streams.
- `ggml_meta_submit_order` (`ggml/src/ggml-meta-submit-order-private.h:14-40`) can serialize shared physical-device submission cohorts. The outer graph currently locks its immediate Meta children at `ggml-backend-meta.cpp:4102-4125`; a flattened lock must be acquired only around the six-rank collective, after child Meta graph submission has returned. Holding the physical locks while launching child Meta graphs can deadlock their worker submission lanes.

## Minimal MetaCPP-only bridge shape

Keep this separate from `backend_ctx->p2p_ar` and `backend_ctx->comm_ctx`. The current merged-graph path assumes a flat `n_backends`-sized communicator and calls `p2p_ar_set_params(ar, j, ...)`; assigning a six-rank handle to the two-child outer context would index the wrong ranks. Add nested-only owned state, e.g. `nested_leaf_backends[6]`, `nested_p2p_ar`, its callbacks, and optional `nested_comm_ctx` / callback.

1. At nested outer-context initialization, recursively walk each immediate child backend depth-first. Accept exactly two Meta children, each with exactly three leaves; each leaf must advertise the CUDA collective procedures through its backend registry. Preserve depth-first order (island A ranks then island B ranks), reject duplicate device handles, and require exactly six leaves. The P2P init procedure itself rejects non-CUDA leaves and unsupported peer topology.
2. At a boundary whose outer `get_split_state(node, assume_sync=false)` is `PARTIAL`, fetch each immediate island output `node_j`, then descend its Meta buffer with `ggml_backend_meta_tensor_get_simple(node_j, k)` for `k=0..2`. This yields the six physical tensors in the same order as the backend vector. Require all six to be contiguous F32, same shape/bytes, non-null data, and a valid allocated device buffer. Never call `ggml_backend_tensor_get`, `ggml_backend_tensor_copy_async`, or CPU staging on this path.
3. Reserve P2P staging for the largest F32 partial in the current graph shape before dispatch (or grow only when a larger shape is first seen). Try the exported six-rank P2P callback first. If it was unavailable at init, or declines a shape, try six-rank NCCL with forced F32. Log selected route and rank list. If both decline, return `GGML_STATUS_FAILED` with a loud diagnostic for a `PARTIAL` boundary; do not enter the unsupported generic host-copy fallback.
4. Before calling the collective, acquire a `ggml_meta_submit_order<ggml_backend_dev_t>` over the six leaf devices in its normal sorted order. Hold it through the six rank enqueues. P2P's cold wrapper drains all six streams before return; NCCL is asynchronous but consumers must be enqueued later on the same physical streams. Release only after the collective has been enqueued. This serializes communicator epochs across competing Meta contexts while allowing disjoint island work to remain concurrent.
5. During outer-context destruction, drain every physical leaf backend, then free the nested P2P/NCCL handles, then free immediate Meta children. For nested partial boundaries, the successful collective must leave the global sum replicated to all six physical outputs before the next graph segment consumes it.

This is a small Meta CPP bridge, not a DPA implementation. It does not implement token routing, attention/KV ownership, or change any weight placement.

## Bounded GPU fixture after authorization

Use the already existing nested/flat Q8 EP toy payload, not the full model. For each run, initialize the same six F32 local partial vectors on the six direct CUDA leaves, with one disabled-compute contributor to verify the collective's existing zero-rank behavior. Run flat Meta6 P2P and nested Meta2x3 bridge for 3 repeated epochs and at least two payload sizes. Include the cancellation witness and ordinary bounded random values. At final validation only, read the six outputs back and check:

- nested bridge vs flat Meta6 P2P: bitwise identical outputs for P2P6;
- every physical leaf has the same allreduce result;
- both match the CPU sequential-FP32 rank-0-through-rank-5 oracle;
- the disabled-compute rank contributes exact zero;
- consumer after the boundary (use a nonlinearity such as RMS/SQR, not `scale(1)`) sees the reduced result;
- repeated epochs have no stale flag/epoch reuse and two concurrent context submissions serialize;
- nested path logs no fallback, no `tensor_get`, no D2H, and no CPU partial gather.

For NCCL6 F32, compare with the existing strict tolerance and report max abs/rel error and rank-order oracle difference; do not require bitwise parity with P2P unless experimentally shown.
