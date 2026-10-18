# V387 MTP vocabulary shard argmax candidate

Status: staging patch and unbuilt fixture only. No active source, model, build, or GPU was touched for this candidate.

## Candidate shape

The opt-in applies only to intermediate steps of the single-row Qwen4Exp MTP unrolled graph. Each of the four existing Q5_K vocabulary row shards still runs its unchanged `ggml_mul_mat`. Instead of concatenating 248,320 F32 logits, each shard runs the existing CUDA `GGML_OP_ARGMAX_SPLIT` unique-summary path, with explicit rank and global vocabulary offset. The four selected summaries have layout `(max, global_id, second, invalid)`, 16 F32 values total (64 bytes for one row), and are reduced by the existing `ggml_argmax_unique_finalize` into an I32 token.

The new result field is deliberately separate from `t_logits`. `t_logits` remains null only for an intermediate MTP step using this opt-in; the final unrolled step retains full logits. The final step, public/raw-logit requests, CPU samplers, non-MTP graphs, non-single-row shapes, non-four-shard shapes, and any failed eligibility check use the existing full-logit gather path.

Opt in with `LLAMA_MTP_VOCAB_SHARD_ARGMAX=1`. The parameter is set only for `step + 1 < unrolled_depth`; the Qwen4Exp graph builder also checks MTP context, unrolled step, GPU argmax, no sampler, one output, no LoRA, and the existing four-shard output-head geometry.

## Fixture

`argmax-shard-fixture.cpp` compares actual CUDA full-head `ggml_argmax` on a synthetic full F32 vocabulary row against four ranked shard reductions plus `ggml_argmax_unique_finalize`. Cases include a shard-boundary tie, negative maxima, mixed NaNs with a finite winner, all NaN, all negative infinity, positive-infinity tie, and equal negative values. It also has a `--oracle-only` CPU mode that checks expected IDs without initializing a backend or loading a model.

The GPU case uses the production shard devices `{0,1,3,5}` and does not load model weights. The fixture validates the candidate reduction's semantics and global offsets; it does not validate the model's four-shard scheduler/P2P concatenation integration or establish a full-model speedup. Root must run it only through the existing bounded GPU guard after build/runtime freeze.

`build-host.sh` is prepared for the server and requires an explicit `RUNTIME_DIR` plus `MTP_SHARD_ARGMAX_V387_BUILD_AUTHORIZED=1`. This turn did not execute the script or either fixture mode.

## Patch

Apply `candidate.patch` only to the exact source hashes in `source-pins.json`. The patch adds a private rank/offset constructor around the existing unique split op; it adds no CUDA kernel or quantization change.
