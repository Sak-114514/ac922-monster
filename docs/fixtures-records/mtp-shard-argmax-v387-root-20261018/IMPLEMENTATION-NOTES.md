# V387 staging review

## Exact source boundary

The patch is generated against the server files and SHA-256 values in `source-pins.json`. It touches only:

- `ggml/src/ggml-argmax-unique.h` and `ggml/src/ggml.c`: one private constructor for the existing unique split reduction, parameterized by shard rank and vocabulary offset.
- `src/llama-graph.h` and `src/llama-graph.cpp`: one internal intermediate-candidate field, reset/reuse identity, and a null-logits MTP-only finalizer branch.
- `src/models/qwen4exp.cpp`: opt-in for intermediate unrolled steps, per-shard candidate creation, and preserving full logits in the final step/fallback.

No active source tree was edited. No CUDA kernel, attention path, quantization, sampling, or public logits API was changed.

## API and fallback contract

The opt-in is `LLAMA_MTP_VOCAB_SHARD_ARGMAX=1`; absent/unrecognized values preserve the existing graph. It is set on an unrolled step only when `step + 1 < depth`. The Qwen graph branch additionally requires MTP context, an unrolled step, GPU argmax, no sampler, one output row, the exact four-shard `model.output` head, no output scale, no LoRA, and the expected hidden matrix shape.

The graph-result contract is strict:

- `t_logits` is always either a real full-vocabulary F32 matrix or null. The candidate buffer is a separate internal field and can never be exposed as logits.
- Candidate mode is allowed only with `output == false` while `active_mtp_step >= 0`; it finalizes to an I32 token consumed by the next unrolled step.
- The final unrolled step always builds the existing full logits and existing output argmax. All non-MTP, sampler, raw-logit, public-output, and ineligible cases retain the original full-logit gather.
- Candidate mode clears its private result pointer as soon as it is converted to the token op; normal full-logit argmax asserts that no candidate pointer remains.

Each selected per-shard summary is `(max, global ID, second, invalid)`. Reusing the existing unique split summary/finalizer preserves first-global-ID tie breaking and the current CUDA full-row argmax token for negative values, NaNs, and infinities. The finalizer's uniqueness flag is computed but unused by the intermediate MTP token path.

For a 248,320-token vocabulary divided into four 62,080-token shards, the current full gather moves 744,960 remote F32 bytes per one-row output (3 remote shards). The candidate representation is 16 F32 values total, 64 bytes; 48 bytes come from the three remote shard ranks. This is a transfer-volume estimate, not a decode-speed claim.

## Verification available for the root

- `candidate.patch` reproduces the five staged source copies exactly; `patch --dry-run -p1` succeeded against the pinned server snapshot.
- `argmax-shard-fixture.cpp --oracle-only` checks expected full-vocabulary IDs against independently merged shard summaries for equal/boundary ties, negative maxima, mixed/all NaNs, `-inf`, `+inf`, and equal negative values.
- `argmax-shard-fixture.cpp --gpu` compares the current CUDA full-head `ggml_argmax` with four actual `ggml_argmax_split_unique_rank` shard ops and the existing `ggml_argmax_unique_finalize`. It uses CUDA devices `{0,1,3,5}` and loads no model.
- `build-host.sh` requires the root's exact frozen `RUNTIME_DIR` and an explicit build authorization variable. Neither it nor either fixture mode was run in this preparation task.

The standalone fixture does not prove the real llama scheduler's peer-copy ordering, CUDA Graph capture selection, production hidden-state parity, or a full-model throughput gain. Those remain separate gates before the opt-in is considered for any serving profile.
