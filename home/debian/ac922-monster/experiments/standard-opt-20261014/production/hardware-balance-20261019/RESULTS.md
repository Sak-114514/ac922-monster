# Hardware-balanced L5 candidate: actual model results

2026-10-19, private worker PID577096. **Not promoted. Original L5 restore started; readiness tracked separately.**

## Scope

Seven original L5 engine DSOs and original R5 helper unchanged. CPU-only worker; full IQ3_XXS/expert set/Q8 KV, fixed D3, shared pool524288/request max262140 unchanged. Shared MTP bodyGPU1/outputGPU5, 16GiB hybridPLE quotas3968/128/3840/2816/2688/2944MiB, fixed draftUB128. The attempted wider target profile used2048 at32K and1024 at64K.

## Uncached prefill

| Case | Actual target UBatch | Cold tok/s | Retained tok/s |
|---|---:|---:|---:|
| 32K | 2048 | 1737.21 | 2003.04 |
| 64K | 1024 | 4224.24 | 2743.83 |
| 32K, explicit60000 reservation selects existing1024 profile | 1024 | 4403.86 | 2920.55 |

All rows use actual token slices, cache_prompt=false/cached_tokens=0. Retained rows have zero context rebuilds. These are combination measurements, not an isolated causal attribution to placement or batch size. Raw6000+ goal not reached.

## Multi-stream generation and correctness

Aligned V306 exact4x16384/512 corpus slices: all-emissions 225.979tok/s vs frozen V306219.944, +2.74% (insufficient to claim a real speedup). Common post-warmup window 248.513tok/s is separately derived from stored per-flow timing; do not substitute it for all-emissions or whole-request speed.

New MTP accepted/proposed 1390/1955 (71.10%). All2048 aligned output token IDs/texts exactly match V306; all2048 short-story output token IDs/texts also match L5. Four native cases and four8192-token native cases passed. 524288 actual reservation and262140 single reservation passed tiny-prompt semantic canaries. These capacity canaries are not a full262K-input test. Longest new uncached input65536.

## Prefix retention and rejection

UB2048 left device2 at315MiB, below the post-wave headroom threshold. The unchanged safety reclaimer replaced the active wide context, losing its prefix checkpoint; original qualification remains FAIL. No answer or CUDA fault was observed. Do not remove the memory guard to force a cache hit.

Same process UB1024 preserved the context and reused30720tokens of32768 with zero rebuilds. Prefix improves repeated-prompt latency; cache-hit logical input throughput is not raw prefill throughput.

NVML samples3102, per-device maxC [60, 59, 59, 57, 58, 57], minfreeMiB [865.375, 713.375, 315.375, 467.375, 478.0, 1635.875]. No thermal wait or hardware/fan/power change. Numerical engine unchanged.

## Startup evidence, without re-running existing outputs

Original18GiB-only worker admission and frozen loader15GiB whitelist mismatch were repaired using existing16GiB support. C3 answers were all correct; its initial boot gate failed only because draft128 was compared to stale expected256. The private gate changes exactly those two expected metadata values and retains all frozen prompts, answer/JSON checkers and hash checks; a wrong256 negative test passes. Existing C3 output CPU re-evaluation passed without another inference request.

Aligned diagnostic initially saved a complete successful model wave then its parser expected corpus metadata omitted by the slim live record. It was finalized offline from the existing wave/full token record and frozen boot corpus digest, with no repeat inference. Original failed qualification was never relabeled PASS.

## Disposition

Reject this combination for production: no robust generation gain, raw prefill regression, and UB2048 prefix eviction. Keep original L5 and preserved artifacts. Next bounded investigation: bulk-prefill graph preparation versus retained graph path while retaining decode graphs. No R441/parallelprepare repeat. No shutdown; latest instruction is continue.
