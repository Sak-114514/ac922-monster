# V87 SM70 packed decode chunks：只读审查

日期：2026-10-16  
范围：审查当前私有 CUDA 源码中的 default-off packed ordinary MMVQ 分支；未改 engine、未 build、未调用 GPU。V87 build 在途，不把静态审查写成 runtime 验收。

## 审查结论

新分支的索引变换在源代码层面与多次 T4（或 T8）MMVQ 等价：同一组 K 个量化权重被多个输出列组复用；每个输出列仍走原 MMVQ T4/T8 特化、原 Q8_1 行量化、原 K 方向累加和原 warp reduction。通道维只拆分独立输入列组，没有跨组求和、原子累加、权重格式转换或新精度路径。满足 gate 时，预期每组输出可与原分块路径逐元素一致。

该结论依赖现有输入权重布局和 stride 条件。它是源码级数值/lifetime判断；当前尚未跑 CUDA fixture，不能宣称 bit-exact runtime PASS。默认关闭，现有 tail、F16/BF16 路径保留。

## 源码版本锚点

本机与服务器相关文件 SHA-256 一致：

- ggml/src/ggml-cuda/ggml-cuda.cu: a428f41ba9ea2c10d234669d6aebd0cdb9b0f4f7dbbb87b6b4c256a9616215ca
- ggml/src/ggml-cuda/mmvq.cu: a83ed741f20619008d64839dcb2e3f40ddcc9054df396edda5afe5e6b47d0748
- ggml/src/ggml-cuda/mmvq.cuh: b37f8359f519ba90ddecd9cb3ca5f8c2ae05685bfae6813d435d75763c2bb006
- ggml/src/ggml-cuda/common.cuh: 602774b94fa20f4defbb8486d8c42298f1c9213d4c37d0b57b0201d0ef7c6553
- ggml/src/ggml-cuda/quantize.cu: bfa995ac91171345a9bfbba7873562d206e76912e3626aadff53574d7a32d445

## Dispatch contract

V87 adds the packed route inside the existing SM70 decode-linear chunk branch in ggml-cuda.cu:2684–2720.

The new environment switch is GGML_SM70_DECODE_LINEAR_PACKED, parsed as a static flag and true only for value 1. It has no effect unless the existing GGML_SM70_DECODE_LINEAR_CHUNKS setting selects 4 or 8. The ordinary outer branch also requires:

- NVIDIA Volta;
- weights on a CUDA buffer with WEIGHTS usage;
- quantized src0;
- src0 dimensions ne2=1 and ne3=1;
- src1/dst dimensions ne2=1 and ne3=1;
- the ordinary element strides expected by MMVQ;
- ne11 greater than one chunk and at most 64.

The packed sub-branch further requires ne11 divisible by chunk size, contiguous F32 src1 and dst, and ggml_cuda_should_use_mmvq(type, SM70, chunk) true. Otherwise it falls through to the preexisting per-chunk loop or ordinary dispatch.

This preserves safe edges:

- One chunk or fewer is not packed.
- A partial final chunk is not packed; the original loop retains its min(chunk, remaining) tail behavior.
- F16/BF16 are not packed because the new branch also checks ggml_is_quantized; their existing paths remain.
- Noncontiguous, batched/sample, or multi-channel source/destination shapes do not enter the packed view path.
- Types that MMVQ declines at the selected chunk width fall back unchanged. On Volta, the current chooser allows Q2_K at T≤4, Q3_K≤6, Q4_K≤5, Q5_K≤6, Q6_K≤7 and other supported quantized types through T8. Thus K-quant types outside those limits do not use packed T8.

## Indexing and arithmetic proof

Let C be the existing chunk width (4 or 8), G=ne11/C, K=src0.ne[0], and N=src0.ne[1]. The packed descriptors are:

- input: [K,C,G], input.nb[2] = input.nb[1] × C;
- output: [N,C,G], output.nb[2] = output.nb[1] × C;
- source weights remain [K,N,1,1].

Because the original src1 and dst must be contiguous and have ne2=ne3=1, their original third-dimension stride is nb1 × ne11. After the reshape, nb1 × C × G equals nb1 × ne11, so the unchanged nb3 remains correct. The branch checks ggml_is_contiguous on both synthetic descriptors before dispatch.

For ordinary dense MMVQ, ggml_cuda_mul_mat_vec_q computes:

- ncols_dst from dst.ne1, which remains C and therefore selects the same compile-time T4/T8 specialization;
- nchannels_dst from dst.ne2 = G;
- nchannels_x from weights.ne2 = 1;
- nchannels_y from input.ne2 = G;
- channel_ratio = nchannels_dst / nchannels_x = G.

In mmvq.cu:682–700, each grid-y channel d maps to channel_x=floor(d/G)=0, so every group reads the same 2D weight matrix, while channel_y=d selects its own activation group. The grid is built with grid.y=nchannels_dst in mmvq.cu:1070–1078. Each group writes to dst + d × stride_channel_dst. There is no cross-channel reduction.

Within each output column, the math remains the existing kernel:

- mmvq.cu:653–680 sets the same template width, K block loop, and per-thread accumulators.
- mmvq.cu:778–818 accumulates each quantized K block in the same order.
- mmvq.cu:820–859 performs the same within-block partial-sum and warp reduction.
- The channel index only changes input and output base addresses; it does not change the reduction tree or accumulation order.

For an old T4 call on group d, src1 begins at the original column offset d×C and dst begins at the matching output offset. In the packed call, channel_y=d advances by C Q8 rows, and channel_dst=d advances by C output columns. Thus the addresses for each j in [0,C) map to the same activation row, weight matrix, and output column as the old call.

This is an ordinary dense MUL_MAT branch with ids=null and fusion=null. It does not enter MoE routing, expert offsets, GLU, or the MoE Q8 output-producer path. The linear q8_1 cache producer is likewise not used here.

## Q8_1 quantization and cache

ggml_cuda_mul_mat_vec_q computes Q8_1 size from the product ne11×ne12×ne13 and padded K (mmvq.cu:1908–1945). The packed view has C×G=original ne11 rows, so the number of quantized rows and total Q8_1 bytes match the original full activation tensor.

The Q8_1 quantizer is row-wise and preserves flattened contiguous row order:

- quantize.cu:67–84 computes the flattened element index as ((i3×ne2+i2)×ne1+i1)×ne0+i0.
- quantize.cu:564–578 launches over the ne1 and ne2×ne3 dimensions.

For original [K,ne11,1,1], flattened row i1 is 0..ne11-1. For packed [K,C,G,1], flattened row is i2×C+i1. Since G×C=ne11 and each group is contiguous, these produce the same sequence of K-vectors; each row’s Q8_1 scales/sums are unchanged.

common.cuh:1488–1520 defines a two-entry activation cache. For contiguous tensors, matches() compares data pointer, ne0, and nrows rather than axis decomposition. That is safe for this reshape: K, total row count, byte range, and contiguous flattened values are unchanged. If an earlier compatible contiguous view populated the Q8 cache, its row order is still the same. The cache is cleared at graph evaluation boundaries in ggml-cuda.cu; non-cacheable inputs use the ordinary temporary pool path.

## Remaining verification gates

This review proves the candidate’s indexing and accumulation preserve the existing T4/T8 per-column operation at source level, but runtime verification is still required:

1. Compare packed output against the existing per-chunk path for the same F32 input and exact weight bytes, for T4 with 8/16 columns and T8 with 16/32 columns. Require exact elementwise equality and canaries around input/output buffers.
2. Keep Q8 cache both in its production setting and disabled for the local fixture; compare outputs and verify packed groups map to the same flattened Q8 rows.
3. Confirm path logs show packed dispatch for eligible shapes and fallback for non-divisible tails, F16/BF16, and MMVQ-rejected quant types.
4. Full-model semantic/performance validation is separate; no such result is established here.

No GPU or build was used for this review.
