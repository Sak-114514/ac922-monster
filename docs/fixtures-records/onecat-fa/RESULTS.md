# Onecat Flash-V100 FA fixture results

Date: 2026-10-15. GPU: AC922 GPU0, Tesla V100-SXM2-16GB. This was an operator-only CUDA fixture; no model was loaded and no engine build or server was run.

## Patched defects

The standalone Onecat source patch is `fattn-volta-onecat.patch`.

1. The launch had two enumerators for GQA chunks: an outer host `chunk` loop and `grid.y=kv_chunks`. For QH24/KVH2, GQA=12 and this made the last launch address heads 24–29, beyond the 24-head Q and workspace. `grid.y` is now 1 so each host-selected six-head group is launched once.
2. Scratch sizing used `dst->ne[2]` as QH. Flash-attention output layout is `[D, heads_q, query_tokens, streams]`, so `dst->ne[2]` is query-token count. It now uses `dst->src[0]->ne[2]`. Before the fix S4/NKV512 Onecat scratch was 4,298,816 bytes; the corrected allocation is 4,441,600 bytes. A 128 KiB canary immediately after the reserved FA allocation was corrupted before the fix and remained intact after it.
3. The reduction used `dst->nb[2]` as the head stride, although heads are output dimension 1. It now uses `dst->nb[1]`.
4. Q conversion bounded work by `gridDim.y*heads*D`; the launcher uses a 1-D grid and `gridDim.y=1`, so only stream 0 was converted. The kernel now receives the exact host element count.
5. Onecat indexed a mask row for each query stream but admitted masks with fewer stream rows. Admission now requires a non-null mask to have `mask->ne[3] == Q->ne[3]`; broadcast masks fall back to stock attention.

## Verification

The fixture used QH24/KVH2, D256, T1, streams 1/4/8, and NKV512/8192. It ran stock and Onecat in fresh processes against the frozen V5 ggml CUDA libraries. Only the patched Onecat translation unit and standalone fixture were compiled; ELF binding checks confirmed V5 `libggml-cuda.so.0` resolved Onecat `supported`, `get_alloc_size`, and `launch` to the fixture executable’s patched definitions. The Onecat route marker appeared with `QH=24 KVH=2 T=1 S=4 n_kv=512`.

The fixture covers Q as a permuted view, gathered K/V as Q8_0 views with every other cache row selected, F16 masks with per-stream masked tails, contiguous K/V controls, a strided no-mask control, an output canary, and an S4→S8→S4 CUDA-graph A/B/A replay. The A output was byte-identical before and after B, and all canaries remained unchanged.

Stock-versus-Onecat output comparisons passed all eight files with a strict absolute elementwise gate of 0.004 (`rtol=0`). Maximum absolute error was 0.00394; maximum RMS error was 0.000924. The full comparison output is in `../../results/onecat-fa-fixture-20261015/compare-strict.txt`. Raw F32 output files are retained in `../../results/onecat-fa-stock/` and `../../results/onecat-fa-onecat/`; hashes are in `../../results/onecat-fa-fixture-20261015/sha256sums.txt`.

## Remaining gaps

- The actual server profile reported NKV2304; this fixture covers NKV512 and 8192, not 2304.
- No full-model request, MTP verification, server concurrency run, performance timing, or model-level semantic check was performed.
- The fixture covers matching per-stream masks; broadcast-mask shapes are deliberately rejected by the Onecat admission guard and were not exercised through the stock fallback here.
- `compute-sanitizer` was not installed. The explicit post-allocation canary passed for every fixture shape.
- Onecat remains an opt-in experimental path; this result establishes bounded FA parity and scratch safety for the tested synthetic shapes only.
