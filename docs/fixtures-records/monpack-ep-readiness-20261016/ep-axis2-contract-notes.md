# Monpack EP axis-2 integration contract

Status: contract-only CPU fixture and source review. This is not a production
patch. Shared runtime sources, model state, CUDA binaries, and device state were
not changed by this follow-up.

## IQ2_S parity and dispatch return

The retained V49 run is in
`operator-fixture/results/cuda-parity-20261015T210022Z-gpu0.log`.
It has two strict bitwise failures, IQ2_S T=4 and T=8, with max absolute
difference `1.1920929e-7`; T=1 and all other type/token pairs are exact.

The CPU comparison is much larger and uses a different activation quantizer.
`ggml-cpu.c` assigns IQ2_S `vec_dot=ggml_vec_dot_iq2_s_q8_K` and
`vec_dot_type=GGML_TYPE_Q8_K` (around lines 364-368). CUDA MMVQ and Monpack GEMV
quantize the F32 input to `Q8_1`. The logged IQ2_S CPU-vs-stock-CUDA max-abs is
`0.00336189568` at T=4/8, about 28,202 times the Monpack-vs-stock-CUDA
`1.1920929e-7`. The log does not retain output vectors for a direct CPU-vs-
Monpack calculation, but the triangle inequality bounds that max-abs between
`0.00336177647071` and `0.00336201488929`. The CPU result is not a bitwise CUDA
oracle; its discrepancy is a different scale from the Monpack/stock delta.

No source-level arithmetic-order change was found in the dedicated Volta
IQ2_S MoE dot. Stock `vec_dot_iq2_s_group_volta()` and Monpack
`mp_vec_dot_iq2_s_group_volta()` both use the same four-grid loop, DP4A
accumulators, integer scale-rounding expression, and final multiply order:
`float(d) * low_float(q8_1.ds) * sumi`. `mp_d_to_float()` is a
`__half2float()` conversion over `ggml_half`, matching stock's conversion of
`block_iq2_s::d`. Both kernels accumulate per-row group results and apply a
warp reduction. Stock stages Q8_1 input blocks in shared memory; Monpack reads
the same Q8_1 buffer directly. That changes memory staging, but there is no
obvious source-level FMA or reduction-order difference. Stock's
`GGML_IK_FUSED_MOE` gate defaults to 0, and the IQ2_S Volta kernel is enabled
unless `GGML_CUDA_IQ2S_VOLTA=0`; the fixture runner did not pin either setting
or save its process environment in the GPU log. A later SSH environment query
found both unset, but that is not an exact snapshot of the completed run.
`cuobjdump` and `nvdisasm` were unavailable on the host, so generated SASS was
not compared. The mismatch cause remains unresolved; retain the two IQ2_S cases
as strict bitwise failures with `max_abs≈1.19e-7` and do not attribute them to
full-model text corruption. For a future authorized repeat, pin
`GGML_IK_FUSED_MOE=0`, `GGML_CUDA_IQ2S_VOLTA=1`, and unset
`GGML_SM70_DECODE_LINEAR_CHUNKS` to keep both arms on their default ordinary
GEMV paths.

The V49 source does prove the successful `try_mul_mat_id()` return path for
the completed fixture invocation, subject to the completed CUDA calls not
raising an error:

1. The harness explicitly unsets `GGML_CUDA_MONPACK_DISABLE`, registers each
   packed tensor, and verifies `is_registered()` for all seven types.
2. The fixture creates contiguous `[K,N,1]` weights, F32 inputs, I32 IDs with
   owner-local ID 0, and T=1/4/8. EP metadata is absent, so the dispatch's
   `ep` predicate is false. Those inputs satisfy `try_mul_mat_id()`'s shape,
   type, stride, and K alignment checks (`monpack-kernels.cu:798-823`).
3. `MMVQ_MAX_BATCH_SIZE` is 8. For all three token counts,
   `try_mul_mat_id()` returns `mp_run_gemv()` directly. After the type switch
   launches the GEMV, `mp_run_gemv()` returns `true` (`monpack-kernels.cu:561-632`).
4. The `ggml_cuda_mul_mat_id()` callsite returns immediately when that result is
   true (`ggml-cuda.cu:2856-2857`). The run completed all 21 cases without a
   CUDA error. Thus it did not fall through to stock CUDA on the packed tensor.

There is still no per-call success log in V49: `[mp-dbg]` confirms hook entry
and registration state only, and its static counter records only the first
six calls. The control-flow proof above establishes the return from source and
the fixture's satisfied preconditions; the hook line alone does not.

## CPU EP contract test

Run from the parent fixture directory:

```sh
./run-ep-route-cpu.sh
```

The standalone test is `monpack-ep-route-contract.cpp`. It compiles with
`-Wall -Wextra -Werror -pedantic` and covers all seven packed formats:

- Full K and N stay intact on each axis-2 owner; expert-byte stride and
  contiguous ranges cover E=512 exactly for offsets/counts `[0,170)`,
  `[170,341)`, and `[341,512)`.
- Global route IDs map to local E only inside the owning range. Foreign IDs
  produce positive `+0.0f` gate, up, SwiGLU, and TC output values.
- The fused gate/up contract checks both expert lookups before SwiGLU.
- TC histogram and work-list indices use local E. Foreign IDs are filtered
  before indexing `cnt[local_e]`, preventing both in-range silent misroutes
  (for example global 169 on owner 170) and out-of-bounds indices (global 341
  on owner 170). Foreign output slots are initialized to zero before sparse
  work-list rows are consumed.

This is an intended-behavior contract model, not a call into the CUDA kernels.
It proves the ownership arithmetic and expected zero/fused semantics on CPU;
it does not enable EP or validate runtime integration.

Local verification output:
`operator-fixture/results/ep-route-contract-20261015T210631Z-local-cpu.log`
(SHA-256 `32f15c0ddfc2c973ce62cb6ba8a8aa366ce0c3f2edf84dd6bda0b99fe6f34698`).
AC922 verification output:
`operator-fixture/results/ep-route-contract-20261015T210637Z-ac922-cpu.log`
(same SHA-256). Both runs passed.

## Minimal production patch seams (not applied)

Use the existing expert-major sidecar bytes and existing consumers. No new
kernel or sidecar format is needed for the axis-2 storage slice:

1. **Loader — `src/llama-monpack-loader.cpp::tensor_meta_override()`**
   (`695-755`, then member validation/copy through roughly `957`). Add an
   axis-2 branch for one segment and one range. Validate the full index tensor
   against `[K,N,E=512]`; validate member shapes `[K,N,170/171]`, positive E
   shares, and exact prefix sum. The current predicate accepts only axis 0/1
   and requires a Meta member's E to equal global E. For axis 2, set
   `expert_stride = blob_len / global_E`, then copy each local tensor from
   `blob + blob_off + owner_e0 * expert_stride` for
   `local_E * expert_stride` bytes. Full K and N make this one contiguous
   byte copy per rank; do not gather/repack K or N tiles. Register both local
   `[K,N,local_E]` geometry and global `(owner_e0, global_E)` metadata.
2. **Route metadata and existing Monpack consumers —**
   `ggml/src/ggml-cuda/monpack/monpack.h`, `monpack-kernels.cu`, and
   `monpack-kernels.cuh`. Extend the registry with owner offset/global E.
   In the existing GEMV route path, check ownership before weight access,
   translate global ID to local E, and write positive zero for every foreign
   output row. Apply this to the generic MoE GEMV and the IQ2_S Volta special
   kernel. Do not pass a foreign global ID through as local E.
3. **Existing grouped-TC prelude —** `monpack-kernels.cuh::mp_tc_build_wl()`
   (`1318-1363`). Its current phase 1 indexes `cnt[ids[...]]` and phase 2
   indexes `cnt[e]` with the unmodified route ID, while `cnt` has only local
   `n_experts` entries. Filter foreign IDs before either access, map owned IDs
   to local E, count/scatter only those local IDs, and set every skipped
   `(token,slot)` output row to `+0.0f`. The grouped consumer then sees only
   local `mp_wl_entry.e` values.
   The scratch arrays are `static g_tc[device]`, global to the process per
   device rather than context-local. `g_tc_mutex` protects reservation, not
   operation use; safe reuse relies on MetaFIFO serializing TC work on the same
   device. Any future concurrent-context use must preserve that serialization
   or make the work-list scratch context/slot-local.
4. **Dispatch stays fail-closed —** `ggml-cuda.cu::ggml_cuda_mul_mat_id()`
   (`2852-2857`). Keep the registered+EP abort until the loader, GEMV, TC,
   foreign-zero writes, and fused route are all safe. If an EP-aware Monpack
   attempt declines a registered tensor, abort rather than falling through to
   a stock kernel that would read packed bytes as raw GGML layout.
5. **Fused gate/up —** `ggml-cuda.cu::ggml_cuda_mul_mat_vec_q_fused_route()`
   (`5917-5935`) rebuilds the routed output metadata and calls the stock
   `ggml_cuda_mul_mat_vec_q()` path; it does not call Monpack's MMID hook.
   First safe option: prevent fusion when either expert matrix is registered
   Monpack, and run two EP-aware Monpack MMIDs followed by the existing GLU
   op. If the graph cannot take that route, keep fusion fail-closed. Do not
   send packed bytes to the stock fused vector kernel.

The CPU contract test should remain alongside any future core patch. A
production enablement still needs GPU owner-range tests, a full output/history
oracle including rollback and slot reuse, fused and TC dispatch coverage, and
the existing EP abort replaced only after those consumers pass.
