# Two-stream buffer-clear ordering fixture

**Status:** source only; not compiled or run. Do not run on the AC922 until the root task owner explicitly schedules it. It allocates one tiny device word and uses one cudaStreamNonBlocking compute stream plus cudaStreamPerThread, so the run is brief, but it still occupies a GPU.

This isolates the ordering pattern present in the current CUDA buffer clear:

1. enqueue compute on a nonblocking backend-style stream;
2. clear and synchronize only cudaStreamPerThread;
3. allow the already-enqueued compute to write after the clear;
4. compare with draining compute before clearing.

Build command on the AC922 CUDA 12.4 toolchain:

    /usr/local/cuda-12.4/bin/nvcc -std=c++17 -O2 \
      /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/clear-ordering-two-stream.cu \
      -o /tmp/clear-ordering-two-stream

Run only when GPU ownership is explicitly granted:

    /tmp/clear-ordering-two-stream

Expected result:

    clear-before-drain: after clear, before release = 0x00000000
    clear-before-drain: after compute drains = 0xa5a55a5a
    drain-before-clear: final value = 0x00000000
    ordering fixture: PASS

This demonstrates CUDA stream semantics, not that a full llama.cpp model has this race or that it alone explains malformed output. The next validating step, if the fixture passes, is source-level synchronization-before-clear in the warmup and sequence-removal capability probe, then a controlled request correctness check on the existing server runtime. Do not infer full-model correctness from this fixture.
