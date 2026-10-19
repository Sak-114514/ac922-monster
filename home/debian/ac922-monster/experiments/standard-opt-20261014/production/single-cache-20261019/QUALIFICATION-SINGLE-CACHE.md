# L3: single conversation cache routing

2026-10-19. Qualified on the six V100 AC922. Parent and independent reviewer read the new cached 512-token output; all four long-reference IPC outputs and eight HTTP outputs exactly match the reviewed L2 reference. Model, quantization, experts, fixed D3, standard indexer, engine libraries and safety configuration are unchanged.

## Added behavior

A single follow-up can reuse its original bank's complete checkpoint instead of always rebuilding/routing bank0. Require exact token prefix, at least75% input cached, at most8192 uncached tail tokens, enough existing context capacity, quiescent owners and six-card free-memory admission. All four bank capacities/UBatches remain unchanged. No state transfer or extra allocation. Otherwise use the existing bank0 cold fast path. Existing2–4 routing, independent progression and refill remain enabled.

## Actual results

| Workload | Result |
|---|---:|
| D/A/B/D singleton follow-ups, 10,817-token prompt | 0.798–0.816s per complete answer; each10496cached tokens |
| Mixed bank0 UB1024 and cached bank3 UB256 | 1.043s answer, zero rebuild and unchanged four-bank geometry |
| Cached 512-token coherent CPU-cache essay | 8.431s complete; ended at requested token cap |
| Original question after that essay | 0.776s, correct9274 and10496cached tokens |
| Insufficient existing bank capacity | Coldbank0 fallback, no false cache hit, correct9274 |
| SSE disconnected after first content | Partial generation stopped; follow-up bank3 hit10496 and answered9274 |
| Four512 IPC, first wave | 188.32 actual output tok/s |
| Four512 HTTP, later wave | 205.47 actual output tok/s; L2 measured210.00 |
| Uncached exact32K, early / later rebuilt / retained repeat | 3723 /4173 /4709tok/s; L2warm4754 |
| Minimum per-card free / maximum GPU temperature | 699.375MiB /60C |

These are individual runs with different warm-up/order. This patch improves repeated-conversation latency, and does not establish a raw compute speedup. Cached logical-token rates are not raw prefill. The earlier300decode/6000prefill objective remains unmet.

All ten distinct-document waves, early-edit misses, cold fallback, mixed profile, cancellation, native checks, HTTPauth/refill/SSE and full-output comparison passed. CPU30API+5coordinator/policy/setup/mock tests and worker identity passed. Only the worker TU was incrementally rebuilt (58.56s); zero CUDA/engine TUs. The six-device first-capture proof is reused because all seven DSOs are byte-identical L2.

Pool524288 and maximum request262140 are unchanged. Full-pool allocation proof is inherited from the unchanged L2 allocator; this campaign did not feed a524K input or reboot the server. See `MEASURED-SINGLE-CACHE.json`, `lastwave-acceptance.json` and `evidence/` for exact scopes and hashes.

## Operation

Frozen target release: `/home/debian/ac922-monster/production/releases/20261019-r444-banked-l3`, backend8004. Promotion keeps the qualified worker loaded and switches the gateway. Subsequent service startup uses the frozen worker/runtime, with output text recording disabled. Candidate control socket is sealed before public switch. Existing safety unit remains bound to inference and stops it on fault.

Rollback to L2: `sudo python3 /home/debian/ac922-monster/production/backups/single-cache-20261018T181309/stage_lastwave.py --rollback /home/debian/ac922-monster/production/backups/single-cache-20261018T181309 --force`.

No BMC, fan, hardware power, driver, kernel, network or shutdown changes. No shutdown scheduled; production stays available while the next compatible optimization is prepared.
