# MTP unrolled KV dependency fixture

Scope: synthetic bounded GGML graph only. The fixture checks whether a depth-1 KV cache read has a scheduler-visible dependency on depth-0 `SET_ROWS`, despite an in-place cache alias. It does not execute a model or prove a model-level MTP fix.

Modes:
- `cpu`: preflight and exact sequential-vs-unrolled comparison.
- `cuda`: one visible GPU.
- `meta3`: mirrored three-GPU Meta backend.

The sequential reference runs depth 0 and depth 1 as separate graphs. The unrolled variants compare model-order aliasing, a deliberate read-before-write node ordering, and a versioned cache tensor edge (`GET_ROWS(SET_ROWS0(...))`). Observed outputs are marked with `ggml_set_output` so allocator liveness reuse cannot corrupt post-run inspection.
