# V210 scheduler-backed Q5_K output-head fixture

This fixture checks the real `ggml_backend_sched` route for a four-way row-sharded M=1 output head. It reuses V203's metadata reader and exact-Q5_K `output.weight` payload helper by including `vocab-row-shard-fixture.cpp`; that original fixture is not modified. No GGUF model is loaded. The reader validates the real Q5_K tensor shape and reads only its 437,043,200-byte payload.

The scheduler backend order is GPU5, GPU0, GPU1, GPU3, CPU, with `parallel=false` and one scheduler copy. The F32 M=1 input is allocated on GPU5. Four contiguous original Q5_K row shards are allocated on GPUs 0, 1, 3, and 5 and marked `WEIGHTS`; each `MUL_MAT` node is pinned to its shard owner. The graph expands all four matmul producers before it constructs the GPU5 concat chain and argmax. The fixture explicitly allocates the scheduler graph, uses scheduler async-compute plus synchronize for warmup and timed iterations, reports the actual split/copy counts and assigned owners, and compares every full F32 logit and argmax with the V203 full-GPU5 reference.

The M=1 dispatch profile sets `GGML_SM70_DECODE_LINEAR_CHUNKS=4` and `GGML_SM70_DECODE_LINEAR_PACKED=1`. For Q5_K with one input column, the current Volta gates reject the multi-column packed chunk route, leaving the ordinary single-column MMVQ kernel. The V203 [source dispatch audit](../mtp-vocab-row-shard-v203-20261017/results/v203-dispatch-audit.md) documents those source gates; this run records its own V196 runtime path and closure.

This is an operator/scheduler integration proof using synthetic hidden input. It does not prove model-level MTP semantics, sampling behavior, graph replay, or full-model speed. The completed run and artifacts are in [`results/v210-summary.md`](results/v210-summary.md). Build and GPU launch scripts remain separately gated for any future rerun.

```sh
MTP_VOCAB_SCHEDULER_BUILD_AUTHORIZED=1 \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-vXXX \
  ./build-v210-host.sh
```

```sh
MTP_VOCAB_SCHEDULER_GPU_AUTHORIZED=1 \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-vXXX \
CUDA_VISIBLE_DEVICES=<six-fixed-UUIDs> \
  ./run-v210-once.sh
```
