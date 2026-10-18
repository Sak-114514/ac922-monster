# Standard QSA per-head loop fixture

This fixture compares the current standard QSA score graph against the private
`llama_qsa_standard_headloop::build()` helper. It uses only F32 inputs and
ordinary GGML operations; it does not load a model or use the Lightning Indexer
fused operator.

The matrix covers `dim=128`, `heads=4`, `blocks={512,2112}`,
`tokens_per_stream={128,1024}`, and `streams={1,2}`. Each arm reports every
score's maximum absolute and relative delta, rejects non-finite values, and
requires exact selected cell ID arrays for two top-k widths: the model-shaped
`min(2048 + 4 - 1, blocks*4)` and a smaller nontrivial `min(512 + 4 - 1,
blocks*4 - 1)`. Block scores are expanded to four equal member-cell scores in
the same member-fast order used by QSA (`cell_id = block_id*4 + member_id`). A
causal-prefix-like visibility bias leaves enough candidates for both widths.

The score-only and selector graphs are computed in separate contexts. This
keeps readback from extending the score tensor's lifetime through top-k and
makes `score_graph_bytes` the allocator peak of the indexer score graph itself.
The selector graph evaluates two equivalent `ggml_top_k()` nodes per width;
their ordered I32 IDs must match each other and the other arm, so
repeated-member ties are covered by the exact-ID gate. Both runners set
`GGML_CUDA_TOPK_STABLE=1` and the binary records that setting. The fixture also reports the
separate selector graph allocation. The score gate is fixed at max absolute
delta `<=1e-5`; this is an operator-level F32 tolerance, not proof of model
semantic parity.

`build-hostlink-v78.sh` host-links against an already built runtime and does not
invoke NVCC. `run-cpu.sh` runs only the CPU backend. The CUDA runner refuses to
run without `QSA_HEADLOOP_GPU_WINDOW_GRANTED=1`, then checks temperature,
free memory, and active GPU processes. Do not use that variable without a
fresh explicit GPU window from the parent task.

The fixture binary requires the helper header to be present in the source tree.
No model-level performance or quality claim is made by this fixture.
