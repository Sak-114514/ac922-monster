# Direct NCCL AllReduce microfixture

This fixture isolates the installed NCCL 2.27.7 path from GGML/llama.cpp and
compares ordinary grouped all-reduce calls with CUDA-graph captured and replayed
calls. It is prepared for host linking only; running it consumes GPUs 0, 1, and
2 and requires an explicit run window from the root task owner.

## Fixed scope

- Ranks map to AC922 CUDA devices 0, 1, and 2. The runner checks the expected
  PCI bus IDs and UUIDs before executing.
- Payloads are exactly 200 KiB, 400 KiB, 4 MiB, 16 MiB, and 32 MiB, using F32.
- Inputs are small exactly representable integers. Every iteration restores the
  send and receive allocations, verifies the exact all-reduce sum, and checks
  leading/trailing canaries.
- Each size has five warmup calls and twenty measured calls in each mode.
- CUDA events measure the stream interval around the collective. Separate host
  completion timing includes NCCL enqueue/group calls and stream synchronization;
  input/output reset and correctness copies are outside the timed interval.
- The graph mode uses one process per GPU, with every rank capturing and replaying
  the same collective in the same order. This follows NCCL's graph-capture guidance
  because multi-GPU-per-process capture/launch can deadlock. Group boundaries wrap
  each collective consistently on all ranks; all stream synchronization occurs
  after `ncclGroupEnd`.

The fixture reports timing and arithmetic correctness only. It does not prove a
GGML integration, a production route, or an end-to-end speedup.

## Host build on AC922

Copy this fixture directory to the corresponding AC922 fixture path, then run:

```bash
cd /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nccl-allreduce-readiness-20261016
./build-host.sh
```

This uses g++ and the installed NCCL/CUDA runtime headers and libraries. It does
not invoke NVCC, initialize a CUDA device, or launch a test.

## Authorized GPU run only

After root confirms the GPU window and independently checks target GPU processes,
run the following on AC922:

```bash
cd /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nccl-allreduce-readiness-20261016
./run-nccl-microtest.sh
```

Each run writes to a new UTC-tagged directory under `results/`; existing logs are
not truncated. The script has a 50-second process timeout and returns nonzero on
any failed communicator setup, graph operation, canary, or exact-sum check.

## Read-only GGML integration seam (current remote V53 source)

Source of truth inspected on AC922:

`/home/debian/ac922-monster/experiments/standard-opt-20261014/src`

- `ggml/src/ggml-backend-meta.cpp:3450-3485` initializes both the ordinary
  communication callback and the in-graph P2P AllReduce callback for a multi-CUDA
  meta backend.
- `ggml/src/ggml-backend-meta.cpp:4915-4940` invokes cold P2P first, then the
  existing NCCL/internal communication callback, then the generic butterfly
  fallback.
- `ggml/src/ggml-cuda/ggml-cuda.cu:1594-1670` implements the existing NCCL
  reduction and calls `ncclGroupStart`, one `ncclAllReduce` per device, then
  `ncclGroupEnd`. Its current large-tensor path converts F32 to BF16 and back.
- `ggml/src/ggml-cuda/ggml-cuda.cu:1844-1890` validates and enqueues cold P2P
  all-reduce nodes for all ranks before synchronizing streams.
- `ggml/src/ggml-cuda/allreduce-p2p.cu:410-448` is the captured `GGML_OP_ALLREDUCE`
  kernel dispatcher. It retains the one-shot P2P path below
  `GGML_CUDA_P2P_AR_RING_MIN_BYTES` and uses segmented P2P above that threshold.

If a future production candidate is justified by this microfixture, the narrowest
dispatch seam is the cold meta-backend P2P callback before it creates `GGML_OP_ALLREDUCE`
nodes: add an opt-in large-payload NCCL branch only for a validated three-device
group, retain the P2P callback for decode/small payloads, and leave `GGML_OP_ALLREDUCE`
dispatch and its replay-safe P2P state intact. NCCL graph use must be collective and
uniform across all ranks, with rank-order-identical launches. The existing generic
NCCL callback is a second seam, but it currently receives grouped partial tensors
and converts large F32 reductions through BF16; changing that behavior would affect
more than the P2P cold path.

Remote source SHA-256 at preparation time:

- `ggml/src/ggml-cuda/ggml-cuda.cu`: `b57dcd93590e69c63ae725ca2357659fa472573122d011447bd97a86a4aaf65a`
- `ggml/src/ggml-cuda/allreduce-p2p.cu`: `25d3619a3fbd518c06949698e2f886cd3066ea9690d39cd2dc54e907d5ddf0af`

NCCL graph support/rank-uniformity for the installed 2.27 line, group semantics,
and the multi-GPU-per-process capture caveat in current NCCL guidance:

- [NCCL 2.27 Using NCCL with CUDA Graphs](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2277/user-guide/docs/usage/cudagraph.html)
- [NCCL 2.30 Using NCCL with CUDA Graphs](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/usage/cudagraph.html)
- [NCCL 2.27 Group Calls](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2277/user-guide/docs/usage/groups.html)
