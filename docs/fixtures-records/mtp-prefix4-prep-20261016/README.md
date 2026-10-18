# MTP fixed-prefix rows=4 preparation (2026-10-16)

Scope: isolated fixture work only. The live core source was not edited and no engine/model trunk was built. A host-only fixture binary was compiled against the frozen V41 runtime, then run under the assigned three-GPU head-only window; all runs exited and the GPUs were released.

## Proposed opt-in behavior

- `LLAMA_MTP_FIXED_PREFIX_ROWS=6` keeps the current six-row behavior for D2 and D4.
- `LLAMA_MTP_FIXED_PREFIX_ROWS=4` opts into four rows per lane only for D2; it covers up to three real catch-up rows plus the real seed, so there are no synthetic future rows at the observed maximum prefix.
- Unset, empty, or unsupported values keep fixed-prefix mode disabled, matching the current explicit-opt-in behavior.
- Four rows at D4 are rejected from the fixed-prefix path and use the existing non-padded path. The value `3` is intentionally unsupported: it would not cover three catch-up rows plus the seed.
- Existing lane, batch/ubatch budget, context end, sequence ID, contiguous position, output-seed, rollback, and future-padding guards remain in place. `make_plan` rejects catch-up-prefix count `>= rows`.

The staged patch is `patches/mtp-fixed-prefix-rows4.patch`. It is pinned to the source hashes in `patches/source-sha256.txt`; patch dry-run against the current source tree passed. Apply/review only in the root-owned integration step, not from this fixture directory.

## CPU planner verification

Run:

```bash
./run-planner-cpu.sh
```

This compiles only `planner4-test.cpp` against the staged policy/planner headers with `-Wall -Wextra -Werror`. It covers explicit 4 and legacy 6 parsing, unset/invalid env values, D2-only rows4, supported 2/3/4 lanes, max catch-up=3, lane-major ordering, seed-only output row, future seed-backed pads, row budget, prefix>=rows rejection, unsupported row rejection, and position/context boundary cases at seed positions 31, 32, 61, and 62. It does not build or link llama.cpp.

Expected current result:

```text
PASS planner4 rows=4 lanes=2,3,4 d2_only=1 legacy_rows6=1 unset_disabled=1 max_catchup3=1 future_pad=1 context_boundary=31,32,61,62 rejections=budget,prefix>=rows,unsupported-rows,position-overflow
```

## Head-only D2 correctness matrix design

Use the existing `fixtures/mtp-real-head-20261015/mtp-real-head-d2-context-fixture.cpp` and extend its row count/seed position setup under a new fixture-local copy. Use the frozen runtime V41 and the same real 34-tensor head-only payload; do not load the trunk or PLE. The following was the intended matrix for the assigned GPU window; the final V41 run stopped at the mixed-prefix parity failure recorded below:

1. Run the same real seed tokens and head state through serial D2 and fixed-prefix D2; compare each step's full logits and hidden state under the existing strict tolerance, exact token IDs, final hidden state, final per-sequence positions, and internal captured hidden outputs.
2. For lanes 2, 3, and 4, exercise sequence IDs `{0,1}`, `{1,3}`, `{0,1,2}`, `{0,2,3}`, `{0,1,2,3}`. Include per-lane catch-up counts `[3,...]`, mixed `[0,2,3]`, and `[0,1,2,3]`; compare only the actual real rows against serial D2. Under rows=4, expected graph row counts are 8/12/16 and the max-prefix case has zero pad rows. For legacy rows=6, preserve existing 12/18/24 row layouts and compare against its existing baseline.
3. Repeat `4 -> 2 -> 4` on the same contexts/slots to catch graph-signature or slot-reuse errors. At every transition, verify the same exact logits/token/hidden comparisons.
4. Check seed positions 31 and 32 with the same deterministic head inputs. Check the context edge with `n_ctx=64`: seed 61 is valid for D2 if all dependent positions and any padded rows fit; seed 62 must reject because the existing dependent-position guard requires `seed_pos + depth < n_ctx`.
5. Force a planner/graph rejection and verify rollback removes every synthetic future row, restores the original catch-up+seed batch byte-for-byte, and falls back. Include a case where a short prefix needs pads and a near-context-end case where the pads overflow.
6. Record graph row counts and logs proving the rows=4 arm was actually selected; an arm that silently uses serial or six-row mode is not a pass.

This head-only fixture validates MTP head graph correctness only. It cannot establish target-model quality, serving correctness, or a throughput gain. The 2-lane rows4 shape is 8 rows and is eligible for the existing Volta MMVQ batch-size threshold; lanes 3/4 produce 12/16 rows and are not expected to hit that <=8 MMVQ path.


## V41 head-only rows4 run status

The exact run artifacts are retained in `results/`:

- `prefix4-v40-20261015T175507Z.log`: first probe; max-prefix rows4 stepwise/unrolled parity passed, but the extra-sequence canary used a different initial KV state from its baseline and failed. Treat the canary comparison in this attempt as an invalid oracle.
- `prefix4-v40-20261015T175704Z.log` and `prefix4-v41-20261015T175800Z.log`: follow-up fixture-collection attempts; these exposed assumptions about one output event per variable-length batch and then an incorrect API index. These logs are preserved as fixture failures, not runtime parity evidence.
- `prefix4-v41-20261015T180049Z.log`: corrected output collection uses the official batch-token output indices and a canary/reference pair with identical canary initial state. The 2-lane seed31 max-prefix D2 case passed exact full-logit/token/final-hidden checks (max absolute logit and hidden error 0); same-state canary step0/step1, rollback, final positions, and unrelated-sequence preservation passed. The next 2-lane seed32 mixed-prefix `[0,3]` case failed the existing strict full-logit tolerance after token IDs had already matched. The current compare helper reports only a generic failure, so max-absolute error and failing D2 step were not captured. No retry was run after this failure.

The mixed-prefix result is a real unresolved rows4 parity failure in the current head-only fixture run; rows4 should remain disabled pending root investigation. It does not establish full-model quality or throughput. The run loaded only the 34 selected real tensors (payload 3,485,816,832 bytes; trunk and PLE weights zero), on fixed CUDA UUIDs 3/4/5. Post-run readback showed 38/38/39 C and about 16.1 GiB free per card, with no remaining compute process.
