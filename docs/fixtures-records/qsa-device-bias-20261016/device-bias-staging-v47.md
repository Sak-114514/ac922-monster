# V47 ordinary QSA device-bias staging

Date: 2026-10-16

This fixture is a source-only, default-off candidate against the V47 private source. No core files
were modified and no full engine build or model load was performed. A bounded tiny CUDA/Meta3
attempt reached backend setup but failed before candidate computation; details are in the r5 update.

## Activation and ordinary-path evidence

In `qwen4exp.cpp::build_qsa_top_k`, `blk_bias` is selected from the normal attention mask shape:
`kq_mask.ne[0]==n_kv`, `ne[1]==n_tokens/n_stream`, `ne[3]==n_stream`, causal attention, no ALiBi.
`llama-graph.cpp::dsv4_build_comp_inputs` constructs that mask with exactly those dimensions when
the DSV4 plan has KV cells. This path is independent of `get_blk()` and
`LLAMA_QSA_POOLED_MULTI`; it is usable by the current standard/no-persistent-pool route.

The feature requires literal `LLAMA_QSA_BIAS_DEVICE=1`. Unset, `0`, or other spellings retain the
existing host F32 bias inputs. Non-`blk_bias` cases also retain the existing per-cell host bias.

## V47-exact candidate

The upstream 9f5a320 contract includes an additional future-block term, but V47's actual host
`blk_bias` branch does not: it writes `-INFINITY` for absent/foreign blocks, `+1e9` when a valid
block start is at/after the query's partial-tail start, and `0` otherwise; a spare/dead block is
always `+1e9`. The staging implementation intentionally follows that V47 oracle instead of
copying the upstream future-mask formula.

With the feature enabled, the graph receives only compact host metadata:

- `blk_idx`: block start indices for valid owner-sequence blocks;
- `blk_add`: the spare-block `+1e9` value;
- `blk_dst`: valid physical destinations, otherwise unique scratch rows;
- `q_tail`: each query's `floor((q+1)/ratio)*ratio`.

On each layer's backend, the graph computes the `0/+1e9` tail values, fills an F32 base with exact
`-INFINITY`, then uses existing `ggml_set_rows(F32, F32, I32)` to scatter valid block values.
Foreign/absent entries target unique scratch rows beyond the visible block range, leaving their
physical rows at exact `-INFINITY`. The chain is pinned to `model.dev_layer(il)`. No CUDA source or
new kernel is added. The map key includes ratio, device, block count, tokens per stream, and stream
count to prevent stale graph-shape reuse.

## Artifacts and verification

- Patch: `qsa-device-bias-v47-r5.patch` (updated five-file source patch; includes the shared private graph/input helper). The earlier `qsa-device-bias-v47.patch` is retained as the initial snapshot.
- V47 baseline SHA-256: `llama-memory-hybrid-idx.cpp` `8a09c00826ca7e749e8b13607a822a146a46a23f20fd90c144450ea415d6b7f6`; `llama-memory-hybrid-idx.h` `101bc4ba17ffde3d7e8e855142ec7b837c7909e2f9fa4a6cbe69d161f01266e6`; `models.h` `28bbc4351fe62bf6d07c41ad4bbef4939497aafa54dce7a9140ad8f6737544b5`; `qwen4exp.cpp` `67adf511d7490a6a1f278a56c0d7fbfe813c5dce2d18e8bd703b86b3ad03b89f`.
- Baseline files and clean patch target are preserved under `rebase-v47/` and `apply-v47-r2/`.
- Patch dry-run/apply against the clean V47 fixture passed.
- `qsa-device-bias-cpu-fixture.py` r5 passed: pooled-score formula, default-off gate, exact
  `-inf/0/+1e9` bias values, sequence membership, tail, V47 no-extra-future-mask behavior, and
  `ggml_set_rows` compact-index shape/scratch contracts.
- `qsa-device-bias-ggml-fixture.cpp` uses the same staged private inline helpers as production,
  constructs the actual GGML arithmetic/FILL/SET_ROWS graph, and passes on CPU using the existing
  V47 GGML shared libraries. It verifies complete bias bytes, stable physical top-k IDs, final KQ
  masks, one/two streams, holes/foreign/dead/spare/tail rows, same-shape sequence rebind, shape
  keys, and legal positions through 262K. Its optional `cuda` and `meta3` mirrored modes print and
  assert scheduled backend placement plus `supports_op` for every device-bias node. The initial r4
  attempt ran both modes and failed before computation; see the r5 update.
- Runner: `run-qsa-device-bias-ggml-fixture-r5.sh cpu` runs the CPU fixture (verified). The r5
  runner uses the compact-index helper in `apply-v47-r5`; its `cuda` and `meta3` modes refuse to
  start unless `CUDA_VISIBLE_DEVICES` is set; use the assigned one-device or three-device UUID
  subset. The largest case has 65,536 block rows and no model weights; the
  temporary tensors are only a few MiB per device by shape accounting, below the 512 MiB/card cap.
- Syntax-only C++17 compilation passed for patched `llama-memory-hybrid-idx.cpp` and
  `qwen4exp.cpp` and the GGML fixture. An existing signedness warning in `qwen4exp_qsa_mblk_fast`
  remains.

## Boundary and merge notes

This patch was prepared on V47 independently from
`fixtures/qsa-pool-turbo-20261016/qsa-pool-turbo-phase-v47.patch`. They both touch
`llama-memory-hybrid-idx.{h,cpp}` and `models/qwen4exp.cpp`; the phase-ledger patch additionally
changes context/lifecycle and CELLSEL code, while this patch changes only ordinary QSA bias input
plumbing plus `models.h`. Do not apply the two patches blindly as independent rebases; merge their
method signatures and QSA input fields.

The CPU fixture is not a CUDA/Meta3 backend parity or prefill-speed test. Before any opt-in run,
use the staged modes on one CUDA backend and then a real three-member mirrored Meta3 backend. Verify
every logged node stays on the target and its backend supports the op, then compare the device bias
bytes/selected block IDs with the host oracle on real V100, including graph replay across shapes and
sequence rebind. Keep `LLAMA_QSA_BIAS_DEVICE` unset unless those gates pass.

## r5 update and bounded backend attempt

The shared helper retains the F32 `REPEAT` for block-index scores (the backend support log showed
that op is supported by CUDA) and removes the I32 destination-index `REPEAT`. `SET_ROWS` now receives
compact `I32[B,1,S,1]` indices, broadcast across query tokens by the existing GGML contract. The
actual CPU GGML graph fixture asserts this shape, rejects any I32 `REPEAT`, and passed exact bias
bytes, stable physical top-k IDs, and final masks for one/two streams and 65,536 blocks. The
standalone CPU contract fixture and clean V47 patch dry-run also passed. `qsa-device-bias-v47-r5.patch`
SHA-256: `581160fa670c31d1cbea147539f7f746e4c4ad48c4e57bfdd8f20109bdf06d01`.

The bounded r4 GPU attempt did not compute a candidate result:

- Single CUDA: all earlier operations were CUDA-supported, but the final I32 destination `REPEAT`
  was logged as `supports_target=0`; that node is removed in r5. This does not establish CUDA parity.
- Mirrored Meta3: backend initialization succeeded, then scheduler allocation aborted in
  `ggml_backend_meta_get_split_state`'s `cached()` assertion that each queried dependency has a
  Meta buffer. The fixture's host metadata inputs are graph leaves; likely the scheduler left at
  least one such leaf in a CPU buffer, but the failing tensor was not printed, so this remains a
  hypothesis. No post-r5 Meta3 retry was made.

Keep `LLAMA_QSA_BIAS_DEVICE` disabled. The r5 CPU proof is not a CUDA execution/parity proof, and
Meta3 placement/allocation remains unresolved. No GPU process remains from this fixture.
