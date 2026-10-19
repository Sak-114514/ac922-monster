# L4: bounded context reserve and prefill pacing

2026-10-19. Same model, IQ3_XXS/Q8 formats, all experts, fixed D3, standard indexer and seven engine DSOs as L3. Only host worker/policies were recompiled. No hardware settings changed.

## What is added

Existing UB256 banks in a2–4 request wave may reserve the next8K capacity bucket, at most40960 cells, only when the exact plan already requires growth. Whole-pool check and all-six-card preflight apply. Every positive8K growth unit costs128MiB in the engineering budget, plus256MiB floor and256MiB reserve. Other banks, mapping and retained hits are unchanged. Insufficient budget uses the original exact decision. If successful allocation still leaves too little memory, all rounded additions are released and exact contexts restored before the existing floor check. That recovery is CPU-regression-tested, not forced by unsafe GPU memory exhaustion.

The existing prefill soft wait has an explicit61/57C profile. Legacy remains59/54C when the opt-in is absent; hardstop65C, sensor checks, cancel polling and120s bounded wait remain. Actual runtime thresholds are logged. No fan, GPU power or driver changes.

## Actual results

| Test | Result |
|---|---|
| Two conversation reserves18717→19009 tokens | First exact18944 rounded24576; next wave zero rebuild and10496cached tokens perrequest |
| First growth / next larger same-prefix complete wave | 9.568s /1.262s; correctanswers3179 and6842 |
| Early prefix edits | No false cache hits; correct7301 and4159 |
| Cached singleton/mixed profile/cancel-follow | Pass; unchanged existing geometry and correctanswer9274 |
| Four512 actual IPC / HTTP whole-window output | 193.49 /207.33tok/s |
| Exact no-cache32K later rebuilt / retained repeat | 3180.04 /4652.87tok/s |
| Four exact8192-token semantic inputs | 3455.57tok/s; soft-wait sum0; allanswerscorrect |
| Lowest observed free / hottest GPU | 705.375MiB /61C |

All2048 IPC token IDs and full texts, eight HTTP full texts and the cached512-token essay are byte-identical to reviewed L3 outputs. The essay ends at its512-token cap; no unlimited-length quality claim. Cache latency is the confirmed benefit. Raw rates are single samples with warm/order differences: L3HTTP205.47 and warm32K4709.32, so no stable raw speedup is established.300/6000 remain unmet.

CPU proof: growth13, originalretention9, thermal boundary, affinity, setup and prefix fixtures; API30+coordinator13; growth and affinity positive/negative mocks. Zero CUDA/engine TUs rebuilt. Onecat six-device proof reused because runtime libraries are identical.

## Test repairs and limits

The first private boot found a missing runtime output directory before context creation. It was archived, directories and frozen native scenarios provisioned, and a new startup preflight passed both missing-directory negative and restored positive tests. All real campaign gates ran in the subsequent single loaded model.

Growth's four-document seed primed the later cold-only affinity case. Nine functional hit/edit waves passed; after explicit cache-disabled native4 invalidation, only cold4 was repeated and passed. Original failure, repair receipt and actual order are preserved. Assertions were not weakened.

The full524288 allocation proof is inherited from unchanged L3 allocation paths; this campaign did not submit a524K input. A separate CPU audit confirmed a pre-existing near-pool active-floor crash also present in L3; C5 is repairing it. L4 is an interim release for qualified normal workloads, not the final completion of that boundary fix.

Target frozen release: `/home/debian/ac922-monster/production/releases/20261019-r444-banked-l4`, backend8005. Promotion uses the already-qualified worker. Next start disables full generated-text recording; the current acceptance process continues under its protected private run directory. Root timer and backup preserve L3.

Rollback: `sudo python3 /home/debian/ac922-monster/production/backups/growth-cache-20261018T184015/stage_lastwave.py --rollback /home/debian/ac922-monster/production/backups/growth-cache-20261018T184015 --force`.
