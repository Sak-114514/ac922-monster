# V167 four-bank D4/D5 local preparation

V167 derives a single-load D4-then-D5 runner from the corrected, native4-checked V146 source. It prepares two independent four-bank context families, with the D4 family drained and destroyed before D5 contexts are constructed. One model object is retained across both families. Each family has its own four-case native canary followed by one four-bank 16K-prompt/512-output performance case.

## Capacity and MTP path

Each target and draft context remains 17,152 tokens per sequence. With 16,384 prompt tokens, 512 requested output tokens, depth D, and one extra guard token, the context slack is 251 tokens for D4 and 250 for D5. `LLAMA_MTP_PAD=6` holds depth plus seed for D5 exactly. The runner keeps the generic single-context GPU token chain enabled, keeps the fixed-prefix unrolled route off, and requires chain steps on all four banks with no fallback lanes in each scenario. The core's six-row fixed-prefix unrolled policy supports through D4 only, so D5 never selects it.

The profile targets the frozen V169 runtime (`20261017T013609.676118Z`; build-report SHA `d4451cb0…a46c7c0b`) and selects stable CPU input leafs plus allocation timing (`GGML_SCHED_ALLOC_TIMING=1`, `GGML_SCHED_ALLOC_TIMING_ALL=1`). It keeps `ROOT_UBATCH=256`, 16,384-token logical prefill batches, two decode threads and eight prefill threads per bank, four bank owners, shared model weights, and one serialized warmup frame per bank. It also requires force capture with structural-signature checks and path logging. The pinned V160 tiny gate observed compatible capture selected; V169's tiny25 off/on allocator gates are still pending. V146 requested draft `n_ubatch=512` but reported the actual draft context value as 256; V167 preserves the request and checks/records the observed 256 baseline.

## Scenarios and evidence

Each family preserves the four original native IDs and frozen prompts/checkers, but uses exact D4/D5 labels. The performance cases reuse the pinned real corpus source and its loaded-model token-count proof, with four disjoint 16K token slices. Raw performance outputs remain unscored.

The runner records both the existing post-warmup shared-window TG and an inclusive first-to-last emissions rate over all banks, including serialized warmup and tail. It derives `prefill_parallel` and `host_submission_parallel` from the count of nonempty banks, and explicitly makes no GPU overlap claim from host submission.

The V146 reference in `v167-profile.json` is retained for context only: native four-case canary passed; performance outputs were unscored; reported shared-window TG was 248.358 tok/s. Root's inclusive 2,048-emission reference was 148.067 tok/s over 13.831588 seconds including warmup and tail. These are separate metrics, and neither is a V167 result or a speed claim.

## Local checks

Run the preparation and CPU guards from the workspace root:

```sh
python3 monster-engine/standard-opt-20261014/fixtures/layer6-fourbank-depth-v167-20261017/prepare_v167_runner.py
python3 monster-engine/standard-opt-20261014/fixtures/layer6-fourbank-depth-v167-20261017/prepare_v167_scenarios.py
python3 monster-engine/standard-opt-20261014/fixtures/layer6-fourbank-depth-v167-20261017/prepare_v167_profile.py
python3 monster-engine/standard-opt-20261014/fixtures/layer6-fourbank-depth-v167-20261017/test_v167_cpu_guards.py
```

`build-v167-host.sh` is authorized for a one-translation-unit V169 host link, but requires `V167_HOST_BUILD_AUTH=ROOT_V167_HOST_BUILD`. This task did not run that script, load a model, build remotely, or touch a GPU. The profile keeps the GPU/model launch held until the V169 tiny25 off/on allocator gates and a separate root GO; it contains no V167 speed or quality claim.
