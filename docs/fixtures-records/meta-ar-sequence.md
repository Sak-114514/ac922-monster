# Three-rank repeated Meta AllReduce fixture

Status: source updated and compiled successfully; it has not been run. Do not launch it while the root task owns the GPUs. It loads the prepared runtime directly; do not use the old CUDA/NVML wrapper.

The fixture creates an actual 3-member CUDA Meta backend and repeats eight graphs, alternating 16 KiB and 64 KiB tensors. Each graph contains four dependent partial-reduction boundaries (add → square → add → square), and queues different per-rank stream delays before dispatch. Every watched intermediate is marked GGML_OUTPUT before scheduler allocation so post-sync reads cannot observe allocator-reused storage. It checks each intermediate consumer output plus the final local partial on every rank.

Three process modes isolate backend selection without changing drivers or libraries:

- fallback3: P2P AllReduce object disabled; GGML_CUDA_ALLREDUCE=none. The communicator callback returns false and Meta runs its generic copy/add butterfly.
- NCCL reference: GGML_CUDA_ALLREDUCE=nccl with P2P AllReduce disabled.
- graph-P2P: GGML_CUDA_ALLREDUCE=none with P2P enabled. First occurrences of new shapes may use the cold Meta fallback; repeated shapes should enter the merged in-graph P2P path. The quick shape mix can set the P2P segmented threshold to 64 KiB.

Build on the AC922 against the already prepared runtime:

    export SRC=/home/debian/ac922-monster/experiments/standard-opt-20261014/src
    export RT=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-standard-combined/lib
    /usr/local/cuda-12.4/bin/nvcc -std=c++17 -O2 --generate-code=arch=compute_70,code=sm_70 -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -DGGML_USE_NCCL -DMONPACK_AVAILABLE=1 -I"$SRC/ggml/include" -I"$SRC/ggml/src" -I"$SRC/ggml/src/ggml-cuda" /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/meta-ar-sequence.cu -L"$RT" -Xlinker -rpath -Xlinker "$RT" -lggml-cuda -lggml-base -lggml -o /tmp/meta-ar-sequence

When the root task releases the selected three GPUs, export CUDA_VISIBLE_DEVICES with the intended three-card island order and LD_LIBRARY_PATH="$RT:/usr/local/cuda-12.4/targets/ppc64le-linux/lib:/usr/local/cuda/lib". Use one fresh process per mode and pin each run to CPU 0 with taskset -c 0.

fallback3:

    taskset -c 0 env GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P_AR_DISABLE=1 /tmp/meta-ar-sequence

NCCL reference:

    taskset -c 0 env GGML_CUDA_ALLREDUCE=nccl GGML_CUDA_P2P_AR_DISABLE=1 /tmp/meta-ar-sequence

Captured graph P2P:

    taskset -c 0 env GGML_CUDA_ALLREDUCE=none GGML_CUDA_P2P_AR_RING_MIN_BYTES=65536 /tmp/meta-ar-sequence

For graph-P2P mode, leave GGML_CUDA_P2P_AR_DISABLE unset. Inspect startup and Meta logs to confirm P2P initialization and merged-graph dispatch; correct output alone does not prove graph capture/replay was used.

This is a correctness fixture, not a throughput or full-model test. It specifically stresses repeated fallback scratch use, dependent reduction boundaries, rank-arrival skew, shape changes, and the NCCL reference path.
