# QSA long-batch gate for V110

This bounded fixture checks the old adapter rejection, the V110 8192-query adapter limit, and actual SM70 long-query operator execution. It loads no model and changes no engine source.

## Intended gates

- Build `qsa_standard_longbatch_graph.cpp` once with `QSA_LONG_BATCH_USE_OLD_HEADER`: 4096 queries must be rejected by the prior helper (`invalid_shape`).
- Build it against the current private header: 4096 and 8192 queries must construct one explicit-cell `FLASH_ATTN_EXT` node; 8193 must be rejected as `query_count_exceeds_8192`.
- The host graph gate runs the accepted/rejected shapes for both F16 and F32 input masks.
- The CUDA fixture compares the current helper on 4096 queries / width 513 / KV 4096 and 8192 queries / width 2051 / KV 8192 against concatenated 1024-query calls to the same helper and kernel.
- The CUDA cases use the native F16 attention-mask input and assert that the helper's gathered/cast output reaches the QSA op with its required F16 type.
- Each CUDA shape runs A/B/A through the same graph. A and A outputs must be byte-identical; B must change output; every full/chunk pair must be byte-identical. It also checks finiteness, full query/mask/ID immutability (including padded query rows), and unchanged Q8_0 KV tail canaries.

The kernel's grid z axis is the query count. The static source audit found no kernel-level 2048/4096 query gate; the prior 2048 gate was in the graph adapter. This fixture is the runtime check for the 4096/8192 launch path.

## Build commands (do not run until root schedules the fixture)

Run from the experiment root. Link to the exact frozen V110 libraries on AC922; do not use a library from another runtime version.

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -DQSA_LONG_BATCH_USE_OLD_HEADER \
  -Isrc/ggml/include -Isrc/ggml/src \
  fixtures/qsa-standard-longbatch-v110-20261017/qsa_standard_longbatch_graph.cpp \
  runtime-perf-v110/lib/libggml-base.so.0 \
  -Wl,-rpath,/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v110/lib \
  -o fixtures/qsa-standard-longbatch-v110-20261017/qsa_standard_longbatch_graph_old

g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -Isrc/ggml/include -Isrc/ggml/src \
  fixtures/qsa-standard-longbatch-v110-20261017/qsa_standard_longbatch_graph.cpp \
  runtime-perf-v110/lib/libggml-base.so.0 \
  -Wl,-rpath,/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v110/lib \
  -o fixtures/qsa-standard-longbatch-v110-20261017/qsa_standard_longbatch_graph_new

g++ -std=c++17 -O2 -Wall -Wextra -Werror -DQSA_ENABLE_CUDA \
  -Isrc/ggml/include -Isrc/ggml/src \
  fixtures/qsa-standard-longbatch-v110-20261017/qsa_standard_longbatch_cuda_fixture.cpp \
  runtime-perf-v110/lib/libggml-cuda.so.0 \
  runtime-perf-v110/lib/libggml-cpu.so.0 \
  runtime-perf-v110/lib/libggml-base.so.0 \
  -Wl,-rpath,/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v110/lib \
  -o fixtures/qsa-standard-longbatch-v110-20261017/qsa_standard_longbatch_cuda_fixture
```

## Run commands (root-controlled GPU window only)

```sh
fixtures/qsa-standard-longbatch-v110-20261017/qsa_standard_longbatch_graph_old
fixtures/qsa-standard-longbatch-v110-20261017/qsa_standard_longbatch_graph_new
QSA_LONG_BATCH_GPU_WINDOW_GRANTED=1 \
QSA_LONG_BATCH_BMC_HEALTH_CONFIRMED=1 \
QSA_LONG_BATCH_RUNTIME=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v110/lib \
python3 fixtures/qsa-standard-longbatch-v110-20261017/run-qsa-longbatch-once.py
```

No fixture build or GPU invocation has been performed while root's V109/V110 checks are in flight. A passing fixture is operator-level evidence, not a model-level semantic result or a prefill performance claim.
