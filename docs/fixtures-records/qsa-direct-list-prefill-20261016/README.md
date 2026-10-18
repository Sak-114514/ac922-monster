# QSA prefill direct-list fixture

This fixture checks the opt-in `GGML_CUDA_QSA_PREFILL_DIRECT_LIST` variant of the existing explicit-cell QSA CUDA kernel. It loads no model and retains the production Q8_0 K/V cache format. It is an operator-level correctness/timing check, not a model prefill benchmark.

## Matrix and gates

The fixture now covers GQA4 and GQA8 at 128 and 1024 query tokens, a GQA4/256-token model-shape case, and a 32-token fallback case. The first V81 matrix ran the five original cases; the V82 head-warps follow-up adds GQA4/256. Every case has an 8192-cell active Q8_0 view in a 16384-cell backing cache, 2051 selected cells padded to 2052, distinct seeded query/K/V/IDs/masks, and a Q8 cache-tail canary.

The generic/shared-list arm checks an independent FP64 oracle for eight evenly spaced tokens across every query head. Both arms then check finite full output, replay exactness, changed-input response, and intact cache-tail canaries. The runner compares the full F32 outputs from both seeds bitwise between arms. Exact full-output equality makes the direct arm an all-output equivalence check against the oracle-checked generic arm.

The V100 dispatch estimates use 80 SMs, `n_work=513`, and the production split formula:

| Query tokens | GQA | Expected splits | Rows/split | Shared-list dynamic smem | Direct-list dynamic smem |
|---:|---:|---:|---:|---:|---:|
| 32 | 4 | 10 | 52 | 1664 B | 1664 B (direct request deliberately falls back) |
| 128 | 4 or 8 | 3 | 171 | 5472 B | 0 B |
| 256 | 4 | 2 | 257 | 8224 B | 0 B |
| 1024 | 4 or 8 | 1 | 513 | 16416 B | 0 B |

The path logger's `auxiliary` field is checked against these byte counts. The runner launches one case per process so the path logger's first-dispatch record identifies each actual token shape. It checks the final per-process summary for two host dispatch/capture observations. Graph replays and changed-input replay do not redispatch the kernel, so they do not increase that counter. The 32-token case must report `decode_retains_shared_list_dynamic_smem_bytes_in_auxiliary` when the opt-in variable is set.

Each timing sample is host-observed `ggml_backend_graph_compute()` plus backend synchronization. Three samples are recorded per case/arm; the fixture does not claim a model-level speedup.

## Build and host checks on AC922

Build against the exact V81 runtime under test. No CUDA source rebuild is needed for this host executable:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
g++ -std=c++17 -O2 -Wall -Wextra -Werror -DGETROWS_ENABLE_CUDA \
  -Isrc/ggml/include -Isrc/ggml/src \
  fixtures/qsa-direct-list-prefill-20261016/qsa_direct_list_fixture.cpp \
  runtime-perf-v81/lib/libggml-cuda.so.0 \
  runtime-perf-v81/lib/libggml-cpu.so.0 \
  runtime-perf-v81/lib/libggml-base.so.0 \
  -Wl,-rpath,/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v81/lib \
  -o fixtures/qsa-direct-list-prefill-20261016/qsa_direct_list_fixture

numactl --membind=0,8 \
  fixtures/qsa-direct-list-prefill-20261016/qsa_direct_list_fixture --host-plan
numactl --membind=0,8 \
  fixtures/qsa-direct-list-prefill-20261016/qsa_direct_list_fixture --cpu-oracle
```

## Bounded GPU comparison

Only run after the root explicitly grants a GPU window and confirms the existing BMC telemetry is healthy. The runner requires `QSA_DIRECT_LIST_GPU_WINDOW_GRANTED=1`, selects physical GPU3 by UUID as the only visible device, refuses a busy/hot/low-memory GPU, runs host checks first, and preserves per-case logs/full outputs under `results/`.

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
QSA_DIRECT_LIST_GPU_WINDOW_GRANTED=1 \
QSA_DIRECT_LIST_RUNTIME=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v81/lib \
python3 fixtures/qsa-direct-list-prefill-20261016/run_qsa_direct_list_gpu_once.py
```

The opt-in is set only in the direct-list subprocess. The generic subprocess inherits no value for it. The runner checks actual `qsa_prefill_direct_list` path records field by field (including `requested`, `selected`, `tokens`, `width`, and `auxiliary`), checks path-summary totals, and rejects incomplete matrices or mismatched full-output files. No fixture command starts an inference server, loads model weights, or edits runtime/system configuration.

Run the dispatch-log parser tests without a GPU:

```sh
python3 -m unittest -v fixtures/qsa-direct-list-prefill-20261016/test_qsa_direct_list_runner.py
```

## V82 head-warps follow-up

The V82 follow-up keeps `GGML_CUDA_QSA_PREFILL_DIRECT_LIST=1` fixed and compares `GGML_CUDA_QSA_PREFILL_HPW=0,1,2` on GQA4/8 at 128 and 1024 tokens plus the production-shaped GQA4/256 case. It verifies per-case `qsa_prefill_head_warps` marker fields (`width=HPW`, `auxiliary=n_grp` warps), the direct-list marker, three graph-wall samples, full output finiteness/canaries, and bitwise seed0/seed1 equality across all HPW modes. HPW0 also passes the independent FP64 sample. A 32-token HPW0/direct-list-requested control must match the V81 full output exactly and retain the original grouping/list staging.

Build a separate binary against the frozen V82 whole runtime; preserve the V81 binary and V81 result tree:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -DGETROWS_ENABLE_CUDA \
  -Isrc/ggml/include -Isrc/ggml/src \
  fixtures/qsa-direct-list-prefill-20261016/qsa_direct_list_fixture.cpp \
  runtime-perf-v82/lib/libggml-cuda.so.0 \
  runtime-perf-v82/lib/libggml-cpu.so.0 \
  runtime-perf-v82/lib/libggml-base.so.0 \
  -Wl,-rpath,/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v82/lib \
  -o fixtures/qsa-direct-list-prefill-20261016/qsa_head_warps_fixture_v82
```

After a fresh explicit GPU/BMC gate, run the V82 comparison:

```sh
QSA_HEAD_WARPS_GPU_WINDOW_GRANTED=1 \
QSA_HEAD_WARPS_BMC_HEALTH_CONFIRMED=1 \
QSA_HEAD_WARPS_RUNTIME=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v82/lib \
QSA_V81_RESULTS_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/qsa-direct-list-prefill-20261016/results/qsa-direct-list-gpu3-v81-20261016T074932Z \
python3 fixtures/qsa-direct-list-prefill-20261016/run_qsa_head_warps_gpu_once.py
```

The host-only path-log parser tests are:

```sh
python3 -m unittest -v fixtures/qsa-direct-list-prefill-20261016/test_qsa_head_warps_runner.py
```
