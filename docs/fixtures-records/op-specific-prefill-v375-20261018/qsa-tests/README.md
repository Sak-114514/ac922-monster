# V375 QSA prefill chunk fixture

Actual GGML graph fixture for the existing
`llama-qsa-prefill-chunks-private.h` helper. It tests the production QSA shape:
24 query heads, 2 KV heads (GQA 12), 4:1 cell compression, 8192 cells, TopK
width 2051, and 1024/2048 queries split into 512-query chunks.

It compares whole-sequence and chunked F32 indexer scores, whole-batch host
block bias against per-chunk native device bias, stable selected IDs against a
CPU oracle that reproduces radix threshold membership and the CUDA stable
gather's two output groups (`>threshold` columns, then the selected threshold
ties), and Q8_0 sparse attention outputs. Key and
query positions are nonzero absolute positions; empty cells are represented as
`+inf`, including the cache tail. The fixture checks that all input tensors,
Q8 cache bytes, and canaries remain unchanged. It loads no model.

## Host link

The completed V368r2 host-link command was:

```bash
ssh ac922 'cd /home/debian/ac922-monster/experiments/standard-opt-20261014 && bash fixtures/op-specific-prefill-v375-20261018/qsa-tests/hostlink-v375.sh r1'
```

The binary, SHA-256, and `ldd -r` closure are recorded in
`results/hostlink-v375-r1.txt`. Host linking does not execute the binary or touch
the GPU.

## Parent-controlled GPU run

Only the root agent should run this one-shot command after granting the GPU
window and choosing an idle GPU index and its UUID:

```bash
QSA_PREFILL_V375_GPU_WINDOW_GRANTED=1 \
QSA_PREFILL_V375_GPU_INDEX=<physical-index> \
QSA_PREFILL_V375_GPU_UUID=<GPU-UUID> \
QSA_PREFILL_V375_RUNTIME=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v368r2/lib \
QSA_PREFILL_V375_BIN=/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/op-specific-prefill-v375-20261018/qsa-tests/bin/qsa-prefill-chunks-v375-r6 \
python3 /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/op-specific-prefill-v375-20261018/qsa-tests/run-v375-once.py
```

The runner requires the selected GPU to have no compute owner, be below 58 C,
and have more than 4096 MiB free. It times out after 180 seconds, runs both
query shapes once, verifies postflight ownership, and saves the logs under
`results/`. The GPU test has not been run by this fixture-preparation task.
