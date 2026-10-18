# V110: admit existing QSA sparse kernel for long prefill batches

Objective: remove the adapter-only 2048-query cap that forces UB4096 to the dense attention path. Original selector, selected IDs/order, Q8 KV bytes, padded-mask semantics and CUDA arithmetic remain unchanged. V52 log line2484 records `queries_per_stream=4096 selected=0 reason=invalid_shape`.

Change: private graph helper only. Explicit query limit8192 (grid.z<=8192, kernel block/warp work unchanged) and named rejection above that limit. No new environment, dependency, kernel, public ABI or model change. Existing old header is backed up.

Proof before retention: old helper rejects4096 graph; updated helper accepts4096/8192 and rejects8193. Actual CUDA4096 (and8192 if bounded) A/B/A attention versus chunked same-kernel reference, masks/duplicate padding/cache offset coverage, canaries and input immutability. Meta3 source/placement or actual test as prepared. Then one combined32K/64K full-model run with existing controls, log actual selected=1, semantic canaries and throughput separately. Existing V53 already authorized and will finish unchanged. No heavy build or GPU fixture during its run.

Rollback: restore backed-up header and previous frozen runtimeV108. Do not change shared source, drivers, kernel, BMC, fans, NUMA policy or safety limits.
