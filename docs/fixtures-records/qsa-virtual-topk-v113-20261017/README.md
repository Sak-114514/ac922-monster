# V113 virtual QSA TopK prototype fixture

This directory isolates a prototype for removing three full F32 `[K,Q,S]` intermediates from standard QSA selection: expanded per-cell block scores, F16 mask converted to F32, and the masked score add. Root owns `qsa-virtual-topk.cuh` and `PLAN.md`; the CUDA reference harness below calls that API without editing engine dispatch.

## CPU oracle

Run the pure Python oracle locally:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 fixtures/qsa-virtual-topk-v113-20261017/qsa_virtual_topk_oracle.py
```

It rounds F32 and F16 values explicitly, applies one F32 add after F16 conversion, checks the prototype's ordered-uint32 transform (including `-0` versus `+0`), and compares virtual score loads against a materialized F32 row. The threshold selector emits all IDs above threshold in physical order, then the first physical IDs tied at the threshold. It deliberately does not sort all selected scores by value.

Cases cover ties, negative zero, `-inf`, masked and empty rows, two streams, foreign and padded cells, a shuffled `cell_block` map, and a partial final ratio-4 block. A deterministic 48-shape randomized small oracle checks score bits and selected IDs. This is CPU contract evidence only; it does not emulate the CUDA radix kernels or validate GGML runtime dispatch.

## CUDA harness draft

`qsa_virtual_topk_cuda_fixture.cu` builds the baseline as GGML's explicit `GET_ROWS(block_scores, cell_block)`, F16-to-F32 mask cast, F32 add, and stock `ggml_top_k`. It compares the root header against the exact stock CUDA TOP_K output under `GGML_CUDA_TOPK_STABLE=1`, including output order. The tiny case is `B=257,K=1025,Q=4,S=2,k=16`; an optional bounded large-K case is `B=16384,K=65536,Q=512,S=1,k=2051`. The large case avoids allocating the Q4096 3-GiB three-intermediate baseline during the first harness run.

Each case runs A/B/A through fixed-shape graphs. Two virtual contexts submit with separate streams and independently allocated state/histogram/counter buffers before output is read. The harness checks exact IDs, repeated-A equality, input immutability, and leading/trailing output canaries. An ID mismatch is reported but does not skip timing. It then records five host-inclusive `steady_clock` samples per arm: stock reference is `ggml_backend_graph_compute + ggml_backend_synchronize`; virtual is header launch (including counter zero) plus the arm stream synchronize. Each arm's A input is uploaded once before its timed loop, and no input transfer or output copy is timed. This is host wall time, not a GPU event measurement. CUDA graph capture is disabled at startup for both arms. The harness also reports actual GGML input/reference buffer sizes plus caller-owned virtual scratch and output bytes. It prints allocation sizes as raw evidence; it does not label them as measured peak-memory savings.

The stock CPU `GGML_OP_TOP_K` implementation uses `std::partial_sort` and does not guarantee threshold-tie physical order. Therefore it is not the exact-ID tie reference for the prototype. The harness reference is the frozen runtime's stock CUDA `ggml_top_k` path with the stable selector enabled.

## Build and later run

The harness has not been compiled or run on GPU. V54 owns the GPU window; do not run the build or CUDA fixture until root schedules a bounded window and gives the matching runtime. The build script refuses to compile unless `QSA_V113_BUILD_AUTHORIZED=1`, builds into a new timestamped tag directory, refuses an existing tag, and times out after 120 seconds.

Build later on AC922 using the exact frozen runtime:

```sh
QSA_V113_BUILD_AUTHORIZED=1 \
QSA_V113_BUILD_TAG=approved-unique-tag \
QSA_V113_RUNTIME=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v110/lib \
bash fixtures/qsa-virtual-topk-v113-20261017/build-v113-harness.sh
```

The build script uses `/usr/local/cuda-12.4/bin/nvcc`, `--cudart shared`, `-Xlinker -rpath` for the selected runtime and CUDA 12.4 target libraries, and links the exact runtime GGML DSOs. It stores the binary and log in `builds/<tag>/`; an existing tag is never overwritten.

After explicit root authorization, run the tagged binary path printed by the build script. The tiny case runs by default; `QSA_V113_INCLUDE_LARGE=1` adds the K65536/Q512/k2051 case:

```sh
GGML_CUDA_TOPK_STABLE=1 \
QSA_V113_INCLUDE_LARGE=1 \
fixtures/qsa-virtual-topk-v113-20261017/builds/approved-unique-tag/qsa_virtual_topk_cuda_fixture --cuda 3
```

No engine source is changed here. A passing fixture would be operator-level numerical evidence only, not a model-level semantic or performance result.
