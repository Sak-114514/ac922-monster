# V18 P4 KV-layout qualification fixture

This is an independent host-side runner preparation derived from the frozen V17 semantic and exact-token gates. It does not alter V17 or the engine core. It is not built against a runtime and has not been run on the model/GPU yet.

## Accepted KV layouts

`run-v18.sh` accepts exactly two `ROOT_KV_UNIFIED:ROOT_BANK_CTX` pairs. The runner rechecks the pair before backend initialization and checks the actual target and MTP draft context values after construction using the public `llama_n_ctx`, `llama_n_ctx_seq`, and `llama_n_seq_max` APIs. The actual stream count is computed from `context_total / context_per_sequence` and must match the selected variant.

| Variant | `ROOT_KV_UNIFIED` | `ROOT_BANK_CTX` | `n_seq_max` | Actual context per sequence | KV streams per bank | Capacity interpretation |
|---|---:|---:|---:|---:|---:|---|
| `unified_132096` (default) | 1 | 132096 | 2 | 132096 | 1 | One shared 132096-cell pool per bank |
| `per_seq_128k` | 0 | 262144 | 2 | 131072 | 2 | Two independent 131072-cell streams per bank |

For `per_seq_128k`, four streams across two banks sum to 524288 cells. The result records explicitly state that this aggregate is not a per-request context guarantee. A mismatched pair, false variant label, or context query inconsistent with the selected shape fails before a run can be accepted.

Example selection:

```bash
# Preserve the V17 default layout
ROOT_KV_UNIFIED=1 ROOT_BANK_CTX=132096 ./run-v18.sh RUNNER MODEL OUTPUT

# Select two independent 128K sequences in each bank
ROOT_KV_UNIFIED=0 ROOT_BANK_CTX=262144 ./run-v18.sh RUNNER MODEL OUTPUT
```

## Preserved quality gates

The fixture retains the V17 native semantic checker, same-prompt exact token comparisons, full 256-token outputs, and MTP before/after logits diagnostics. It records target `llama_decode` plus natural synchronization time, MTP prefill processing time, diagnostic dump/logit-scan time, per-lane total, scenario prefill wall, and whole-scenario wall separately. The original cold 8K probe remains; a second same-prompt 8K probe runs after the serial real-token scenarios, clears KV with `bank.reset()`, and checks an exact 32-token prefix against the same serial reference to measure a same-process warm-graph prefill without prompt-cache reuse. No checker tolerance was relaxed. V18 adds strict layout parser tests and result checks for both variants, bad capacities, and incorrect stream metadata.

## Build boundary

The host runner compile script is parameterized for the planned V56 frozen runtime closure. It only links the runner; it must not start a model. The V56 runtime must be frozen and identified before linking. V18 has no authority to change `src/`, the BMC, fans, power limits, or the production service.
