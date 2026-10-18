# V113 virtual QSA cell-score TopK prototype

Objective: retain the standard block scores and exact stable per-cell selection while eliminating the full F32 expanded scores, full mask cast and full masked-scores intermediates. This stage retains the existing F16 dense mask; attention implicit-mask removal is separate.

Inputs: scores F32 [B,Q,S] after original bias; cell_block I32 [K,S] with every value in [0,B); original F16 mask [K,Q,1,S]. Each virtual score is exactly F32(scores[row,B + cell_block] + F32(mask[row,K+cell])). Selected IDs and threshold tie order must match current GGML_CUDA_TOPK_STABLE=1: greater-than-threshold IDs in physical order, then the earliest threshold-tie IDs. No forced whole-block selection or dropped routes.

Root owns CUDA header. Worker owns CPU oracle + eventual fixture. GPU kernels derive the existing fork radix/stable gather, replacing only operand loads. Workspace is caller/context-owned and zeroed on the same stream; no global static per-device counter ring. This also avoids introducing the existing global counter-ring ownership hazard, but no claim of a production race or repair is made yet.

Stage: local isolated source only. No engine integration, build, GPU run, numerical pass or performance claim. Do not heavy-compile/run during V54. Gate CPU corner oracle then current stock materialize+TOP_K vs virtual kernel CUDA exact IDs, A/B/A/multistream metadata, canaries/immutability, separate contexts' scratch and measured full preparation+selection. All raw timings remain recorded even on numerical differences. Only after operator gates and actual memory/speed result decide the private TOP_K ABI/support/all-child Meta graph seam. Existing standard path remains fallback.

Expected workspace elimination at K65536,Q4096: three F32 [K,Q] payloads, each1GiB. This is source-shape size, not a measured peak reduction. Remaining F16 mask512MiB/blockscore256MiB/TopK histogram256MiB retained. No allocation savings claim before actual proof.
