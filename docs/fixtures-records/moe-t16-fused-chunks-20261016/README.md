# Fused EP 12/16-row chunk qualification fixture

This fixture qualifies the private Volta path that executes an exact 12- or
16-row fused expert gate/up operation as the existing 8+4 or 8+8 fused
operation on the same stream. It uses the seven original full-K expert-170 raw
samples under
`fixtures/monpack-ep-readiness-20261016/operator-fixture/samples/`; the
sample checker verifies their recorded SHA-256 values and reads no GGUF.

The graph fixture creates full-K, single-local-expert `MUL_MAT_ID` gate and up
weights, F32 activation rows, and global IDs `[170, 169, 341, 511]` with EP
metadata `(offset=170, global_experts=512)`. Each token has one owned and three
foreign routes. It compares the full GLU output from candidate T12/T16 graphs
with explicit existing fused T8+T4/T8+T8 graph executions. Acceptance is
bitwise equality for all output elements, finite results, zero foreign-route
contributions, unchanged input/ID/weight readbacks, and actual dispatch
selection for all 14 candidate cases.

This remains an operator fixture. The single expert sample and fixture EP
metadata do not establish production EP loader/scheduler integration, a full
model, token history, or performance. No model is loaded and the fixture does
not recompile CUDA code.

## Host-only gates

The descriptor test covers padded strides, view offsets, route IDs, EP
coordinates, original descriptors and backing bytes. T12 must split as 8+4;
T16 must split as 8+8. T1/T8/T9/T10/T11/T13/T14/T15/T32, wrong op/types,
pointer mismatches, invalid strides and null data must decline.

```sh
./fixtures/moe-t16-fused-chunks-20261016/run-cpu-tests.sh
c++ -std=c++17 -Wall -Wextra -Werror -pedantic -fsyntax-only \
  -I src/ggml/include -I src/ggml/src \
  fixtures/moe-t16-fused-chunks-20261016/moe-t16-fused-chunks.cpp
./fixtures/moe-t16-fused-chunks-20261016/verify-original-samples.py \
  fixtures/monpack-ep-readiness-20261016/operator-fixture/samples
```

The host build script requires explicit `LLAMA_SRC_ROOT` and `LLAMA_RUNTIME_LIB`
so a stale runtime cannot be selected accidentally. Use it only after the
parent freezes the CUDA runtime/build; it invokes ordinary `g++`, not NVCC.

## Bounded CUDA command (requires an explicit root grant)

On AC922, from the matching source checkout, first set the exact frozen V79
source/runtime paths and the single physical GPU explicitly assigned by root.
Then run:

```sh
ROOT_MOE_T16_GPU_WINDOW=1 \
LLAMA_SRC_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/src \
LLAMA_RUNTIME_LIB=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v79/lib \
LLAMA_BUILD_ID=20261016T070400.271089Z \
CUDA_VISIBLE_DEVICES=GPU_ID \
./fixtures/moe-t16-fused-chunks-20261016/run-cuda-parity.sh
```

Replace `GPU_ID` with the root-assigned single physical device. The runner
requires the exact frozen V79 build ID, library SHA, dispatch source SHA, and
chunk-helper SHA, plus an idle selected GPU with no compute
PID, zero utilization, memory below 512 MiB, and temperature below 58 C. It
creates a new UTC-tagged log and refuses to overwrite an existing one. It sets
`GGML_CUDA_MOE_FUSED_T16=1` and
`GGML_OPTIMIZATION_PATH_LOG=1`, then requires all 14 cases and a selected
`ep_fused_mmvq_t16` dispatch summary with `requested=14 selected=14`.
Do not treat a missing marker or an ordinary fallback result as a pass.

The V79 host binary linked successfully without NVCC; its CUDA runtime library
SHA-256 was `2336febdc57e436a21aae38f105d9747428380f6a23f49b86394ae5d0b673f89`.
The descriptor and source fixtures passed local and AC922 host-only checks.
No GPU execution or performance measurement has been performed from this
fixture directory yet.
