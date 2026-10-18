# Six-rank P2P relay AllReduce fixture

This fixture directly includes the private `ggml-cuda/allreduce-p2p.cu` translation unit so its static one-shot, segmented, and relay6 kernels and the matching `p2p_ar_args` layout are tested together. It calls the peer allocator APIs from that same source and links only the frozen V17d CUDA backend libraries for shared helpers such as `ggml_cuda_set_device`.

The test resets identical rank inputs for each run and executes one-shot -> relay6 -> one-shot -> segmented -> relay6 -> segmented. Every output is compared bit-for-bit with a host FP32 rank-ordered reference and the first one-shot result. Inputs include large cancellation values and a rotating zero-slice rank. Sixteen 32-bit canary words surround the tensor on every rank.

Graph modes capture one independent kernel graph per rank and algorithm without launching or synchronizing during capture. They instantiate all rank graphs before any replay, then queue all six rank graphs in one order before synchronizing. The fixture logs graph build time plus cold and hot replay event/wall timings; graph replay shares the same device epoch and flags across mixed algorithm calls. `--graph-window` covers 20,480, 81,920, 204,800, 1M, and 15M floats (80 KiB, 320 KiB, 800 KiB, 4 MiB, and 60 MiB).

The requested full sizes are 0, 1, 127, 2560, 20480, 81920, 204800, 1000000, and 15000000 float values. `--quick` limits the run to the first five. The full allocation estimate is 2 x 6 x 15M x 4 bytes of stage per GPU plus 15M x 4 bytes of data, below 1 GiB per device. The fixture rejects allocations above that bound.

Build and run are separate. Do not run while the model or another CUDA process is active. `run.sh` pins the known six UUID order, checks for idle GPUs, refuses start at 58 C, and polls temperature while the test runs. The test queues all six rank kernels before synchronizing any rank; that order is required for peer progress. Timing reports per-rank CUDA event durations, the maximum rank duration, and total host wall time. It does not sum device times.

```bash
./fixtures/p2p-relay6-20261015/build.sh
./fixtures/p2p-relay6-20261015/run.sh --quick
./fixtures/p2p-relay6-20261015/run.sh --full
./fixtures/p2p-relay6-20261015/run.sh --graph-window
```
