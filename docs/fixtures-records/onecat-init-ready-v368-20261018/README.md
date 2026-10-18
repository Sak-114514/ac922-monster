# V368 Onecat initialization/readiness candidate

This is a source-review and CPU-model candidate only. It does not modify active
`src/`, change Onecat math, build GGML, or call CUDA.

## Live source basis

Read from the AC922 experiment tree at
`/home/debian/ac922-monster/experiments/standard-opt-20261014/src/ggml/src/ggml-cuda/`.

| File | SHA-256 |
|---|---|
| `fattn-volta-onecat.cu` | `5958b7bdaba4e23a072e022a471cc0d59327ef6c5624ca26515ea5e729873912` |
| `ggml-cuda.cu` | `cecb639ccd56be159fb9c2fdb6d61b85de080f5faf11ce1816e94fe300cc2e47` |
| `fattn.cu` | `8fa366def24db4b65bdc03a86156d2bee917a2200a4ec850233babf2ae3dccac` |
| `fattn.cuh` | `e3a56f9e753b3994a086f668903e9c960ade092e134df9ff55b351d84828d7b6` |
| `fattn-volta-onecat.cuh` | `f129d83e4cd34560636467bf9044430bcac340b9f4588c968ed943bb70b9f368` |
| `fattn-common.cuh` | `7f314e73434dc7ee1ee5de8b4981f6e13ac3ee3d565f95423866470992c97652` |
| `common.cuh` | `602774b94fa20f4defbb8486d8c42298f1c9213d4c37d0b57b0201d0ef7c6553` |

## Findings

- The fix is gated by both the existing `GGML_CUDA_FATTN_ONECAT=1` and a new
  `GGML_CUDA_ONECAT_INIT_READY=1`. Without the new flag, the existing Onecat
  stream-table and attribute-init path remains unchanged, even if the existing
  retained recipe already sets its Onecat flag to `1`.
- The exact Onecat admission point is `ggml_cuda_get_best_fattn_kernel()` in
  `fattn.cu`: SM70, `GGML_CUDA_FATTN_ONECAT=1`, and
  `ggml_onecat_v100::supported(dst)`. Sparse QSA is returned before this
  selection in `ggml_cuda_flash_attn_ext()`.
- The supported path casts `Q->ne[3]` to `int` for `n_strm`. The candidate adds
  an explicit positive/int-range guard to `supported()` so graph preflight can
  plan a representable table capacity and runtime dispatch cannot bypass it.
- `ggml_backend_cuda_graph_compute()` selects graph mode, may begin capture,
  then calls `ggml_cuda_graph_evaluate_and_capture()`. That is the preflight
  boundary for scanning the actual graph for nodes that the same dispatch
  function admits as Onecat.
- `stream_table()` currently has one global mutex. It allocates the pointer,
  enqueues the iota kernel on the current stream, sets `entries`, and returns
  without a cross-stream completion fence. Another stream can observe the
  pointer and launch Onecat before the iota has completed.
- `smem_attr_set[device]` is an unsynchronized static bool. Multiple streams
  can race through `cudaFuncSetAttribute` and publish the flag concurrently.
- The selector's Q8-head and Onecat log-once guards are unsynchronized
  `static bool`s. The candidate uses relaxed `std::atomic_flag` for these
  log-only guards; selection and kernel math are unchanged.
- Onecat's `get_extra()` begins with the same K/V F16 conversion tail helper
  (`ggml_cuda_flash_attn_ext_get_f16_extra_data`) used by stock attention and
  then appends Onecat-only Q/partial/stat buffers. Stock F16/MMA and tile paths
  request both K/V conversions; the vector path requests at most a subset.
  The candidate still checks the actual Onecat-reserved and stock-required
  allocation sizes before it switches a declined compute to stock.
- The proposed boundary is a private helper included only by
  `ggml-cuda.cu`, `fattn.cu`, and `fattn-volta-onecat.cu`; it does not alter
  kernel arithmetic or a high-fanout CUDA header.

## Proposed contract

1. The new off default skips the graph scan and preserves the existing Onecat
   behavior when the old Onecat flag alone is set; when both flags are set, the
   graph scan and ready-only path are enabled.
2. Before evaluation/capture, a graph scan calls the same Onecat admission
   predicate used by `ggml_cuda_get_best_fattn_kernel()`. The required table
   capacity is the maximum admitted `Q->ne[3]` in the graph.
3. `ensure_ready(device, capacity, stream)` uses a mutex per physical CUDA
   device. When initializing or growing, it performs CUDA allocation, the
   three shared-memory attribute calls, iota launch, a temporary event record,
   and event synchronization before publishing a new table pointer.
4. Old table pointers remain allocated after growth because captured graphs
   can retain them. Each device's initialization lock is independent.
5. In an active external stream capture, a sufficiently sized ready table is
   reusable without any CUDA initialization or event wait. A cold or undersized
   table declines Onecat; stock fallback is used only if its required tail
   allocation fits the already-reserved Onecat allocation. Otherwise the
   compute returns `GGML_STATUS_FAILED` with an explicit log.
6. During Onecat launch, the adapter only looks up a ready table and asserts
   its capacity. It performs no lazy allocation, iota launch, or function
   attribute setup.
7. Successful first initialization and capacity growth emit one explicit
   `Onecat init-ready prepared` record with device, requested entries, and the
   `initialized_ready` or `capacity_grown_ready` reason. Already-ready calls do
   not emit this record, so the message proves setup ran without per-compute
   log spam.

`source-pin/base/` contains the exact fetched remote versions of the three
modified `.cu` files; `source-pin/candidate/` contains the generated candidate.
`prepare_exact_patch.py` verifies their source pins and regenerates the unified
diff. The patch was applied to an isolated copy of the pinned files and compared
byte-for-byte with the candidate tree. It was not applied to the active `src/`
tree, compiled, or run on CUDA.

Expected incremental build closure after integration: recompile
`fattn-volta-onecat.cu`, `fattn.cu`, and `ggml-cuda.cu`, then relink only
`libggml-cuda.so.0`. The change does not touch `fattn.cuh`, `common.cuh`, public
ABI, other CUDA translation units, or kernel arithmetic.
`readiness-model.hpp` and `readiness-model-test.cpp` exercise the lifecycle and
concurrency contract on CPU only; they do not prove CUDA stream behavior.
