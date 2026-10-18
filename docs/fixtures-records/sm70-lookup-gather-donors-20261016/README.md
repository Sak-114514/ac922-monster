# SM70 GET_ROWS donor oracle (host-only)

This fixture checks the address mapping proposed by P100 patch 37 without CUDA, a model, or engine changes.
It is not a GPU benchmark and does not exercise Q8_0 dequantization.

## CUDA narrow GET_ROWS fixture (host-prepared)

`narrow_getrows_cuda_fixture.cpp` is the V78 operator-level follow-up for the narrow batched gather. It builds public `ggml_get_rows` graphs and compares their F32 outputs bitwise with a CPU cast/copy oracle; it does not call a private CUDA kernel. The default `--cpu` mode exercises all shapes without a GPU. The later `--cuda 3` mode is present but must only be run after an explicit root GPU-window grant.

The seven cases include the production QSA mask path, F16→F32 with `ne00=1`, `ne10=2051`, `ne11=1024`, `ne12=1`. Source rows are F16 and the public GET_ROWS output is F32, matching `ggml_get_rows()`'s current API contract. Additional F32→F32 and F16→F32 cases cover row widths 1, 2, and 4 across batched indices and a third stream axis, with deliberately padded source, ID, and batch/stream strides. A width-4 F32 control is large enough for the existing vector kernel; a second small width-4 F32 shape stays below its block threshold to exercise the narrow-vs-generic region without taking work from the vector path. Source/ID padding uses safe sentinel values and is read back after execution.

Build on the target against the exact runtime under test (no engine rebuild/NVCC is needed for this host fixture), for example:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
g++ -std=c++17 -O2 -Wall -Wextra -Werror -DGETROWS_ENABLE_CUDA \
  -Isrc/ggml/include -Isrc/ggml/src \
  fixtures/sm70-lookup-gather-donors-20261016/narrow_getrows_cuda_fixture.cpp \
  runtime-perf-v80/lib/libggml-cuda.so.0 \
  runtime-perf-v80/lib/libggml-cpu.so.0 \
  runtime-perf-v80/lib/libggml-base.so.0 \
  -Wl,-rpath,/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v80/lib \
  -o fixtures/sm70-lookup-gather-donors-20261016/narrow_getrows_cuda_fixture
```

Before any future GPU run, run `narrow_getrows_cuda_fixture --cpu` and preserve the exact runtime/source/binary hashes. `run-narrow-getrows-gpu-once.sh` then runs two separate V80 processes on GPU3: default generic mode with `GGML_CUDA_GET_ROWS_NARROW` unset, then opt-in mode with it set to `1`. Each mode first executes the CPU source oracle; each CUDA case then gets one exactness warmup and three timed graph/synchronize repetitions. Separate processes are required because the narrow-path environment is cached at first dispatch in each template instantiation. `GGML_OPTIMIZATION_PATH_LOG=1` enables the CUDA dispatch marker; the runner requires seven cases, seven warmups, 21 timing samples per arm, a `get_rows_narrow selected=1` marker in the opt-in arm, and no narrow marker in default-off. The runner also refuses unless `GETROWS_GPU_WINDOW_GRANTED=1`, checks GPU3 temperature/free memory/compute owners, and stores the combined log under `results/`.

Acceptance gates for each CUDA arm: all seven cases complete, exact F16→F32/F32→F32 output equality, all output finite, and source/ID padding canaries unchanged. Only the production F16 `1×2051×1024` shape and width-2/4 scalar cases are expected to use the opt-in path. The same-type width-4 vector-control case must remain on the existing vector route; V80 moved the narrow gate after the vector early return. Confirm the frozen-source dispatch order and per-path marker in the log. Wall time is a bounded operator sample, not a prefill performance claim.

## Host-prep result

V78, V79, and V80 CPU backends passed all seven graphs with exact oracle equality and intact input canaries; the evidence is under `results/`. The bounded V80 GPU3 comparison is complete: both arms passed all seven shapes, and the opt-in arm confirmed the dispatch marker. See `results/narrow-getrows-v80-gpu3-20261016.md` and its raw log. The production synthetic mask shape measured 3.577 ms generic vs 0.201 ms opt-in mean across three repetitions; this operator fixture is not a model prefill benchmark.

Run:

```sh
python3 -m unittest -v fixtures/sm70-lookup-gather-donors-20261016/test_narrow_batched_getrows_oracle.py
```

## Patch 37 port proposal (not applied)

The donor flattens `(row, element)` into one scalar thread index in `get_rows_cuda_float`. A narrow opt-in port would:

1. Keep the existing vectorized same-type path first for rows that qualify.
2. Add an early narrow path for `ne00 <= 2` across `ne11`/`ne12`, plus an explicit bounded opt-in max width if a later single-stream measurement warrants it.
3. Compute index offsets as `i10*s10 + i11*s11 + i12*s12`, source address as `id*nb01 + i11*nb02 + i12*nb03`, and output address as `i10*s1 + i11*s2 + i12*s3 + i00`.
4. Reject/skip zero-element launch shapes before creating a zero-sized CUDA grid; preserve existing behavior for unsupported dtypes and non-contiguous dimension-0 layouts.
5. Keep the fallback generic kernel unchanged.

Do not widen the default threshold to 128 based on the synthetic row-width test. That case only proves the indexing formula. It does not prove a performance gain or that a 128-wide row is a profitable flattening shape on SM70.

