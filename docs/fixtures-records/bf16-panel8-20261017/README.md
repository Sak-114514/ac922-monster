# BF16 panel8 verify fixture

This small fixture checks the root-owned `bf16-panel-private.cuh` 8-column
dispatch against the same existing BF16 GEMV kernel called twice at four
columns. The reference is two `<4,4,256,false>` launches; the candidate calls
the actual `ggml_sm70_bf16_panel::dispatch` for `<4,8,256,false>` once. No new
kernel or engine path is added here.

The cases are K=2560,N=8192; K=10240,N=320 (the TP3 narrow-output shape); and
the non-four-aligned tail K=2560,N=8195. Each shape checks three A/B/A F32
activation panels over all output bytes, finite values, input immutability, and
device canaries. Weights are deterministic, finite BF16 patterns; activations
stay F32 with full mantissas. Five event repetitions time the complete
reference launch group and the one-dispatch candidate on the first two shapes.

The host build checks hashes for both the original `kernel_sm70_bf16.cuh` and
the current `bf16-panel-private.cuh`, then compiles only this fixture and CUDA
Runtime. It does not build the engine or load a model. This fixture is prepared
only; no compile or GPU run has been performed. Build later with:

```sh
./build-host.sh /home/debian/ac922-monster/experiments/standard-opt-20261014 panel8-host1
```

The baseline loop uses two launches (`8/4`), and the candidate dispatch uses
one. The file is an operator check for two-bank × two-lane verify8 only; it
does not establish a model-level quality or throughput result.
