# Nested PLE weight-view fixture

This fixture targets the V61 owner-lifetime failure around
blk.1.ple_conv1d.weight. The GGUF C API inspection found:

    type=f16 ne=[4,10240,1,1] bytes=81920

Qwen4Exp's production path in src/models/qwen4exp.cpp:3495-3502 creates four
views from one static weight tensor, makes each view contiguous, reshapes it
to 1D, then casts it to F32 before multiplying by an F32 activation. The fixture
reuses the real packed F16 tensor bytes and this exact graph shape.

The CPU probe attempts standalone Meta3 and nested Meta2{Meta3A,Meta3B} with
mirrored static weights and F32 compute inputs. Its graph checks every static
leaf byte-for-byte against the GGUF payload, retains all four view->cont source
links, and compares A/B/A output on standalone3 and both nested islands. The
GPU mode uses the first three V100s for standalone3, then the full 3+3 islands
for nested2x3. Both modes require GGML_META_CONTEXT_VIEWS=1,
GGML_META_REPLAY_GUARD=1, GGML_META_FIFO_PIPELINE=1, and
GGML_META_NESTED_EXPERIMENTAL=1.

Build the CPU-only binary on AC922:

    ./build-cpu-fixture.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v61
    ./run-cpu-fixture.sh

The CPU binary links only libggml-cpu and libggml-base. Its nested probe stops
before graph execution because the CPU backend does not provide the submit-tail
event required by the requested Meta settings; the exact result is in
results/cpu-meta-event-unsupported.txt. This does not validate or refute the
GPU owner/view issue.

Build the GPU-capable binary only after the root freezes V63, then run the
bounded GPU probe only after a separately assigned window:

    ./build-gpu-fixture-v63.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v63
    CUDA_VISIBLE_DEVICES=... PLE_WEIGHT_VIEW_GPU_WINDOW_AUTHORIZED=1 ./run-gpu-bounded-v63.sh

## Qualification status

The metadata-only inspector confirms F16, shape [4,10240,1,1], and 81,920
packed bytes at file offset 2,479,353,664. The GPU-capable V63 fixture
host-linked successfully; its direct dependencies are only libggml-cpu and
libggml-base, with CUDA loaded dynamically only for --gpu-probe. Binary SHA:
5284391e8c276f51f7e7836f55596a4463fd9d130ec1d6fdea8b1dd317eca2f0.

The CPU-only nested Meta run stopped before the PLE view/cont graph because
the CPU backend returned no submit-tail event (ggml-backend-meta.cpp:3328,
tail != nullptr). Its exact result is preserved in
results/cpu-meta-event-unsupported.txt. The CPU backend cannot exercise the
requested Meta event path.

The first V63 GPU attempt segfaulted before CUDA initialization because the
fixture eagerly evaluated dlerror() on a successful dlopen. That attempt is
preserved at results/run-v63-20261016T013705Z.log. After making the error read
null-safe, the bounded 3+3 V63 GPU run passed:

- The GGUF F16 weight was 4x10240 (81,920 bytes); all standalone3 and nested
  physical copies matched the original packed bytes.
- The four tap views remained bound to the static root
  blk.1.ple_conv1d.weight; logs show owner binds for each tap in both nested
  islands with views/replay guard/FIFO enabled.
- Standalone Meta3 and both nested Meta3 islands produced exact A/B/A output;
  the two island outputs and standalone/nested outputs matched exactly.
- CUDA graph reuse and the final PLE_WEIGHT_VIEW_GPU_FIXTURE_PASS marker were
  observed. No full model was loaded.

The successful run is results/run-v63-20261016T014000Z.log. On exit there
were no compute PIDs, temperatures were 37–38 C, and memory use was 0–15 MiB
per card.
