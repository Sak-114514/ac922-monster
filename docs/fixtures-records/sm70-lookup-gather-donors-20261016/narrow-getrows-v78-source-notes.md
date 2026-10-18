# Narrow batched GET_ROWS: V78 source contract and port hazards

Read-only source review on 2026-10-16; no core files changed. V78 source reviewed on AC922 at `src/ggml/src/ggml-cuda/getrows.cu`, with production call sites at `src/src/models/qwen4exp.cpp`.

## Production shape

The current standard sparse QSA consumer builds the mask gather at `qwen4exp.cpp:2410-2419`:

- `kq_mask` is viewed as `[1, n_kv, n_tok]` (one scalar per cache-cell row).
- `top_k` is reshaped to `[width, n_tok]`; V34's current selected `width=2051` is not padded until after the mask gather, so `ne10=2051`, `ne11=1024`, `ne12=1` for the observed prefill shape.
- `ggml_get_rows` creates `[1,2051,1024,1]` and returns F32, even when the mask source is F16. This is explicit in `ggml.c:4180-4200` (`type=F32`, except I32 input).
- The graph then reshapes, adds/pads, and casts the attention mask to F16. Thus the gather is F16→F32 and its output has 2,100,224 scalar cells (`2051*1024`) before padding/cast.

The current standard per-stream helper uses the same core operation in `src/src/llama-qsa-standard-sparse.h`: it views the mask as `ne00=1`, gathers stream-local IDs, reshapes to `[width,1,1,queries]`, then casts to F16. The fixture includes both the observed V34 dimensions and small 1/2/4-column cases.

## Current V78 dispatch

`getrows.cu:312-370` implements `get_rows_cuda_float` for F16/F32 (and other scalar types). It first considers `k_get_rows_float_vec` for same-source/destination types; that vector route requires, among other conditions, a 16-byte-aligned row width, source/destination strides, pointers, and at least 128 vector blocks. If that does not apply, the generic kernel launches one block per row (`grid.x=ne10`, `grid.z=ne11*ne12`, `grid.y=ceil(ne00/256)`). For `ne00=1`, that means about 2.1 million tiny CTAs for the production mask shape. The narrow flattened path is absent from V78.

The P100 patch 37 is at commit `28ac0d68dc3f3d840a6d0437028c18e811492b05`, file `patches/37-getrows-narrow-batched.patch`. It flattens `ne00*ne10*ne11*ne12` output elements, unflattens in axis-0-fastest order (`i00`, `i10`, `i11`, `i12`), and computes:

```text
src1 index = i10*s10 + i11*s11 + i12*s12
src row    = id*nb01 + i11*nb02 + i12*nb03       (byte offsets)
dst        = i10*s1 + i11*s2 + i12*s3 + i00     (dst-element offsets)
```

It uses a 1D grid with 256 threads, reducing the production grid from roughly 2.1 million blocks to about 8.2 thousand. F16→F32 still goes through `ggml_cuda_cast`; this is a copy/conversion operation, not changed arithmetic.

## Port hazards

1. Preserve the existing same-type vector dispatch before the narrow branch. Otherwise a broad width threshold can steal eligible F16/F32 vector work. The fixture has a width-4 F32 vector control plus a small width-4 F32 case below the current 128-block vector threshold.
2. Keep units distinct: `nb01/nb02/nb03` are byte strides, `s10/s11/s12` are index-element strides, and `s1/s2/s3` are output-element strides. The production QSA path has padded `nb11`, and 2-axis test cases have non-tight index batch and stream strides.
3. Keep `ne11` and `ne12` in both index and source offsets. The production case has a large `ne11=1024` batch even though `ne12=1`; multi-stream cases exercise `ne12=2`.
4. Retain the existing PDL load/consumer synchronization prologue. Do not move it after bounds checks or omit it in the new kernel.
5. Guard zero/overflow cases before forming a zero grid or multiplying untrusted shape products. The test sizes are bounded; they do not validate generic huge-shape overflow handling.
6. The public donor reads `GGML_CUDA_GETROWS_FLAT_MAX` into a function-local `static` via `atoll`. The root port is planned under the distinct default-off name `GGML_CUDA_GET_ROWS_NARROW`; use the root spelling, and run generic/on modes in separate processes so static env caching cannot invalidate the comparison.
7. Environment presence is not dispatch proof. If runtime logs are available, assert the narrow branch was selected for eligible F16→F32 shapes and that the width-4 vector control stayed on vector dispatch.

The CUDA fixture checks exact outputs against a CPU scalar oracle, and verifies source/ID padding canaries. It must be built against the exact runtime after the root kernel port is frozen. Host-only API/syntax checks do not show CUDA dispatch or performance.

## V79 port review (read-only)

V79 build `20261016T070400.271089Z` adds `k_get_rows_float_narrow<WIDTH,src0_t,dst_t>` at `getrows.cu:11-31`, with the batched/source/output stride terms above, fast-divisor unflattening, and the original `ggml_cuda_cast`. The host path at `getrows.cu:363-381` is opt-in through `GGML_CUDA_GET_ROWS_NARROW=1`, requires `ne00` in `{1,2,4}`, and requires `ne10*ne11*ne12` in `[256, UINT32_MAX]`; it logs `get_rows_narrow selected=1` and returns after launch. The per-process static environment read is why the runner uses separate generic/opt-in processes.

Review finding for the V79 source snapshot: this new narrow gate appears before the existing same-type vector check beginning at `getrows.cu:384`. Therefore an opt-in F32→F32 width-4 case with enough vector blocks is also captured by the new kernel. The fixture's `f32_w4_vector_control_i129_b17_s2` is intended to make this visible in the dispatch log; unless the final source moves the narrow gate after the vector early return or excludes `can_vec`, do not claim that the existing vector route was preserved. This is a dispatch/performance concern, not a numerical-correctness failure; the flat copy still has exact output.

V80 moves the narrow gate below the same-type vector early return. The V80 source review at `getrows.cu:363-406` therefore preserves a qualifying width-4 F32 vector route before the opt-in narrow branch. V79 remains unchanged and its initial ordering concern is recorded above.

V80 DSO is frozen. The fixture was relinked to V80, passed the CPU oracle, then passed the bounded default-off/opt-in GPU3 matrix after the root's explicit grant. Exact hashes, timing samples, dispatch markers, safety state, and raw log are in `results/narrow-getrows-v80-gpu3-20261016.md`. The earlier V78/V79 CPU logs remain host-only evidence and must not be substituted for the V80 CUDA result.
