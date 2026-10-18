# Standard QSA per-stream sparse consumer fixture

Bounded operator fixture for the default-off standard-indexer sparse-attention helper. It uses separate cache streams, local cell IDs, changed per-stream/per-token inputs, and a deliberately larger backing capacity than the active view to catch stream-stride mistakes.

Run the CPU oracle:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v fixtures/qsa-standard-streams-20261016/test_stream_cell_oracle.py
```

`test_stream_cell_oracle.py` is a fast F32 indexing/stride oracle. Its results are in `host-oracle-results.json`; it checks address formulas only and does not measure model or CUDA error.

`qsa_standard_sparse_graph.cpp` invokes the exact private helper and checks one tagged explicit-cell attention node per active stream, the backing-cache root and accumulated `view_offs`, full `nb3` stream strides, padded ID/mask shapes, and token concat order. `qsa_standard_sparse_cuda_fixture.cpp` contains the actual backend branch (built with `QSA_ENABLE_CUDA`, no NVCC): helper output, a fixture-only reconstruction of the prior explicit-cell per-stream consumer, an independent Q8_0→FP64 CPU oracle, and a dense masked `ggml_flash_attn_ext` check on the one V77 SM70 shape already proven to have device code (`streams=1, ntps=4, heads=4`). That dense check reads the original Q8_0 KV backing cache, without gathering selected cells into an F16 cache. Other shapes explicitly report the dense FA arm as unsupported/skipped while still requiring helper-vs-old bitwise equality, fixed-tolerance FP64 comparison, finite outputs, distinct stream/token inputs, and unmodified KV-tail canaries.

Fresh AC922 host-only evidence:

- Both fixture C++ files pass local `-Wall -Wextra -Werror -fsyntax-only` checks.
- The exact-helper graph shape matrix passed for stream counts 1/2, queries-per-stream 4/32, and GQA 4/8. With the dense arm included only for the supported shape, output shapes were `[1024,12]`, `[2048,8]`, `[1024,64]`, `[2048,64]` for one stream and `[1024,16]`, `[2048,16]`, `[1024,128]`, `[2048,128]` for two streams. Each graph had one explicit-cell node per stream.
- The Q8_0 CPU FP64 selected-order vs dense-masked reference had max absolute error `1.03e-14` and max relative error `5.02e-12` across the eight shapes. The tail canary spans 1,984 rows per stream; the oracle confirms that using `view_nkv*nb2` instead of the actual `nb3` yields different data.
- Full Q8_0 host results are in `host-q8-oracle-results.json`. This is a CPU mathematical dense masked reference, not GGML's actual `FLASH_ATTN_EXT` output.
- The GPU-enabled C++ fixture binary is host-linked against V77. The `--host-cpu-upload` mode allocates and uploads every matrix shape through the real CPU backend, with no graph compute; it exists to exercise the non-null-backend input-upload path before CUDA is considered. The guarded runner is `run-qsa-cuda-once.sh`; it requires `QSA_GPU_WINDOW_GRANTED=1`, checks GPU3 temperature `<58C`, free memory `>512MiB`, and no compute process on that GPU, then runs only the tiny fixture. It never launches a model or an all-reduce cohort.
- Every shape uses a fixed per-element helper-vs-FP64 gate (`atol=2e-6`, `rtol=2e-5`) and a selected-order-vs-dense CPU gate (`max_abs<=1e-12`). Dense FA is checked only on its proven-compatible shape. The preserved first attempt documents the V77 `flash_attn_ext_f16` device-code failure for another sm_70 shape: `results/gpu3-v77-attempt-20261016.md`.
- Fixture success is local operator evidence only; it does not replace model-level semantic/logit validation or a production benchmark.

GPU evidence for the helper is complete for this tiny fixture: all eight V77 shapes passed helper-vs-old explicit-cell bitwise comparison, the fixed FP64 tolerance, finite-value checks, distinct-input checks, and KV-tail canaries. The direct dense FA reference passed only its proven-compatible `1/4/4` shape; its other seven shape variants are explicitly skipped due missing V77 sm_70 F16 device code. See `results/gpu3-v77-final-pass-20261016.md` for exact binary/source/log hashes. The earlier unsupported-dense and empty-mask-upload failure logs remain preserved in `results/`.

## Four-stream production-shaped follow-up

`qsa_standard_sparse_4stream_cuda_fixture.cpp` is a separate V83 regression, leaving the earlier V77 binary and matrix unchanged. It tests exactly four Q8_0 streams, four queries per stream, 4/8 query heads, 2051 selected cells padded to 2052, an 8192-cell active view over a 16384-cell backing cache, non-tight stream strides, two changed input seeds, and per-stream tail canaries. Its graph compares the actual helper output bitwise against four independent original explicit-cell consumers and checks seed-0 outputs against the FP64 oracle. It does not instantiate the dense F16 FlashAttention path.

Build and run the host gates against V83:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
g++ -std=c++17 -O2 -Wall -Wextra -Werror -DQSA_ENABLE_CUDA \
  -Isrc/ggml/include -Isrc/ggml/src \
  fixtures/qsa-standard-streams-20261016/qsa_standard_sparse_4stream_cuda_fixture.cpp \
  runtime-perf-v83/lib/libggml-cuda.so.0 \
  runtime-perf-v83/lib/libggml-cpu.so.0 \
  runtime-perf-v83/lib/libggml-base.so.0 \
  -Wl,-rpath,/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v83/lib \
  -o fixtures/qsa-standard-streams-20261016/qsa_standard_sparse_4stream_cuda_fixture
numactl --membind=0,8 fixtures/qsa-standard-streams-20261016/qsa_standard_sparse_4stream_cuda_fixture --host-plan
numactl --membind=0,8 fixtures/qsa-standard-streams-20261016/qsa_standard_sparse_4stream_cuda_fixture --cpu-oracle
numactl --membind=0,8 fixtures/qsa-standard-streams-20261016/qsa_standard_sparse_4stream_cuda_fixture --host-cpu-upload
```

After a fresh root GPU/BMC gate, use the bounded GPU3 runner:

```sh
QSA4_GPU_WINDOW_GRANTED=1 QSA4_BMC_HEALTH_CONFIRMED=1 \
QSA4_RUNTIME=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v83/lib \
python3 fixtures/qsa-standard-streams-20261016/run-qsa-4stream-gpu-once.py
```

The completed V83 report is `results/gpu3-v83-4stream-20261016.md`; raw artifacts are under `results/qsa-standard-sparse-4stream-gpu3-v83-20261016T080502Z/` on AC922.

