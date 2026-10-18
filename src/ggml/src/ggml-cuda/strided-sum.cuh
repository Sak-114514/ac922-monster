#include "common.cuh"

// A few strided 2d windows of one tensor summed in order, then an optional scale or same-shape bias:
//   GET_ROWS(q8_0) -> ADD(view, view) -> ADD(.., view) ... -> SCALE   (mean of block members)
//   RELU           -> ADD(view, view) -> ADD(.., view) ... [-> ADD(.., bias)]   (per-head indexer score sum)
// Views may go through a CONT first. Returns the number of nodes after i the fused group covers, 0 if none.
int ggml_cuda_strided_sum_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

// CONT(transpose of block scores) -> GET_ROWS(cells -> block) -> CONT(transpose back) [-> CPY f16 mask to f32] -> ADD(mask):
// each cell gets its block's score plus its mask value, in one kernel. Returns the nodes covered after i, 0 if none.
int ggml_cuda_expand_rows_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

// true if nothing outside nodes [i, end] reads one of roots (tensors computed in that range) or a view of one,
// and none is a graph output: the group may keep them to itself
bool ggml_cuda_group_private(const ggml_cgraph * cgraph, int i, int end, const ggml_tensor * const * roots, int n_roots);
