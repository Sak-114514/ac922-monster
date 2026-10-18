# Native MTP D3 unrolled staging (2026-10-16)

This directory is preparation only. The V45 engine source was not edited, built, or run with D3. No GPU work, model payload read, full-model launch, or benchmark was performed by this fixture lane.

## Exact source base and integration patch

The patch was made from the remote frozen source at `/home/debian/ac922-monster/experiments/standard-opt-20261014/src`. `patches/source-sha256.txt` pins all five patched source files; those hashes matched the remote files when staged. The source directory is not a Git checkout, so the reviewable patch is `patches/mtp-d3-unrolled.patch`.

The patch changes only:

- `src/llama-mtp-lane-policy.h` — centralizes depth/lane eligibility. Depth 3 requires exactly two lanes and retains the existing `LLAMA_MTP_UNROLL_VARIABLE_LANES=1` opt-in. Depth 2 and 4 keep their prior lane rules.
- `common/mtp-fixed-prefix-plan.h` — accepts exact depth `3`; keeps unset depth at D2; allows six prefix rows for D3 while four rows stay D2-only.
- `common/speculative.cpp` — admits the D3/two-lane shape and logs an explicit D3 downgrade reason for three or four lanes.
- `src/llama-context.cpp` — accepts explicit depth 3 and checks the same lane/depth contract both when arming and when processing an unrolled batch.
- `src/models/qwen4exp.cpp` — keeps the model builder assertion aligned with the accepted depths and lane shape.

The patch dry-run passed on the copied source snapshot. It was also applied to a fresh copy; `cmp` confirmed all five resulting files matched the staged patched copies.

## Source audit

The current depth 2/4 restrictions were in the env parser, common selector, context arming gate, context batch validator, and `qwen4exp::build_mtp_unrolled()` assertion. Those were the behavioral hard gates.

The rest of the path is depth-generic: `llama_mtp_unrolled_layout` uses vectors; context reserves each dependent step in a loop; Qwen4exp builds each graph step in a loop and concatenates its token tensors; history is packed and reordered by the requested depth. The per-lane history allocation is `lanes * 32`, and end/readback guards cap depth at 32, so D3 fits. There are no D2/D4-sized fixed arrays in this path.

Fixed-prefix D3 uses six rows per lane: up to five real catch-up rows and one seed, with future rows filling the remaining slots. Existing planner checks enforce contiguous positions, one output seed, the row budget, prefix length, and context bounds. On graph failure the common path rolls memory back from each lane's `first_pos` and restores the saved original batch, removing synthetic rows. A D3 three/four-lane request is rejected by the shared shape gate and takes the existing fallback; the log says `depth-3-requires-two-lanes`.

For two lanes, three draft tokens plus the seed gives a planned target verification batch of `2 * (3 + 1) = 8` rows. The source's existing CUDA route has an eight-token small-batch limit and admits Volta gate/up/GLU fusion for small speculative verification batches. This is source eligibility evidence only; the D3 head fixture does not run target verification or prove that dispatch was selected.

The unrolled graph's current `n_kv` diagnostic prints steps 0 and 1. It still prints `depth=3`; the two-value `n_kv` field is diagnostic only and does not index by the requested depth.

## CPU regressions

Run from the experiment directory:

```sh
./fixtures/mtp-d3-prep-20261016/run-cpu-regressions.sh
```

The suite compiles the patched policy/planner/layout headers with `-Wall -Wextra -Werror -pedantic`. It covers exact D2/D3/D4 parsing, unchanged D2/D4 lane support, D3 two-lane admission and three/four-lane rejection, fixed-six parsing and D3 policy, fixed-prefix planning and rollback anchors, D3 row/position/history strides, token-history repacking, context/position overflow, and the planned eight-row target batch.

Verified output:

```text
PASS d3_shape_gates d2_d4_compat fixed_prefix6 d3_only_two_lanes
PASS d3_fixed_prefix_planner rows=6 lanes=2 max_catchup=5 rollback_pad_positions=checked
PASS d3_layout rows=6 input_stride reservation_stride history_pack=lane-major position_overflow=checked
PASS d3_verify_row_contract lanes=2 draft_plus_seed=8 planned_target_rows=8 sm70_small_batch_ceiling=8
```

## Real-weight head-only fixture

`head-only/mtp-d3-context-fixture.cpp` is adapted from the existing D2 fixture. It requires exactly 34 selected MTP/head tensors: block 48 plus shared token embedding/output tensors. The trunk and 26.8 GiB PLE table remain metadata-only. Its planned matrix compares serial D3 with one-graph D3 for sequence IDs `{0,1}`, `{1,3}`, then `{0,1}` reuse, checking all three steps' full logits, exact token IDs, final hidden, final KV positions, and fixed-prefix future-row rollback. The unrolled arm is restricted to two lanes.

The metadata model uses the regular tensor-split mode with equal split weights `1/1/1`. The fixture requires `ggml_ep_private::requested()` to be true and checks the real model split callback before reading any selected tensor payload. Every selected routed `*_exps.weight` with an expert dimension must report split axis 2, one contiguous segment, 512 total experts, local counts `170/171/171`, and offsets computed as prefix sums `0/170/341`. Payloads are written through the Meta backend using those model tensor descriptors; the fixture does not manually carve or TP-slice expert bytes. The shared/dense tensors continue to use the model's ordinary TP split policy. `LLAMA_EXPERT_PARALLEL` must be exactly `1`; the runtime parser rejects values such as `native=1`.

The earlier `fixtures/mtp-real-head-20261015/run-context-matrix.sh` used `LLAMA_EXPERT_PARALLEL=native=1`, so that D2 matrix cannot be cited as evidence for the production EP axis-2 layout. Its historical results are retained without relabeling.

The source compiled on AC922 against `runtime-perf-v47`. Compilation emitted two deprecation warnings for the same `llama_new_context_with_model` API used by the existing D2 fixture.

The bounded GPU runner completed under the root-assigned three-card window with `LLAMA_EXPERT_PARALLEL=1`. It verified the real model split state for block-48 routed down, gate, and up weights: axis 2, 512 global experts, local counts `170/171/171`, and prefix offsets `0/170/341`. It loaded exactly 34 selected MTP/head tensors (3,485,816,832 payload bytes), with trunk and PLE payloads absent.

All three two-lane D3 head cases passed. The serial and unrolled arms matched full logits and token history at all three steps, final hidden state, final KV positions, and future-pad rollback. Sequence-map rebind and same-context reuse also passed. Maximum absolute logit difference was 0; maximum final-hidden difference was `2.38419e-07` in the first case and 0 in the other two. The fixture records `planned_target_verify_rows=8`; it does not run target verification.

Run tag: `20261015T191651Z-32432`. Full output is retained at `results/mtp-d3-context-matrix-20261015T191651Z-32432.log`.

The GPU runner checks for existing processes, pins the same three head-placement GPU UUIDs as the established head-only fixture, aborts below 512 MiB free or at 58°C, limits runtime to 180 seconds, and disables PLE shard loading. Each run creates a UTC-tagged log and refuses to overwrite an existing log. The final guard observed 38–39°C and 16,116–16,126 MiB free on the three cards. This head-only result validates MTP head graph parity and MTP weight split metadata only; it does not prove target-model semantics, target verification dispatch, server quality, or performance.

The reproducible commands for the already verified V47 runtime are:

```sh
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v47 \
  ./fixtures/mtp-d3-prep-20261016/build-head-d3-fixture.sh
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v47 \
  ./fixtures/mtp-d3-prep-20261016/run-head-d3.sh \
  /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf
```

## Applicable instructions

No additional `AGENTS.md` was found under `monster-engine/standard-opt-20261014`. The workspace root instructions supplied with the task apply.
