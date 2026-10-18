# V323 ngram-map-k4v + generic MTP D3 preparation

Status: V318 CXX hostlink passed. `launch_eligible=false`; no model load or GPU launch has occurred.

## Concrete diff from the V221 runner

Only `dual-context-v323-ngram-mtp-fourbank.cpp` is the new runner source. The old V221 source, scenario JSON, native checker, V221 KV policy, V306/V314 sources and results, engine files, and other agents' files are untouched.

- Each bank still uses the existing Qwen MTP model/context path. `params.speculative.types` now requests `ngram-map-k4v` followed by `draft-mtp`, `ngram_map_k4v.size_n=12`, `.size_m=3`, `.min_hits=1`, and `draft.n_max=3`. `common_speculative_n_max()` is required to remain 3. The existing `common_speculative_init()` order puts K4V ahead of MTP; no n-gram algorithm was added.
- The original four-bank, one-lane D3 shape, target batch 4096 / ubatch 256, MTP ubatch 512, PAD6, CPU threads 2/8, PLE NUMA 18 setting, real corpus source/slices, native four-canary gate, target row padding/verifier, full 2048-token output checks, and V318 runtime remain specified by the profile.
- `LLAMA_MTP_PREFILL_HISTORY_LAST_ONLY=0` is explicit. That optimization is admitted only when MTP is the sole implementation; with two strategies the constructor sets its internal `sole_implementation` argument false. `LLAMA_MTP_KV_ONLY_PREFILL=1` and `LLAMA_MTP_RELEASE_PREFILL_HISTORY=1` remain enabled.
- `LLAMA_MTP_UNROLL_SINGLE_D3`, single-lane unroll, unrolled graph, dedicated scheduler, layer cache rebind, and single-device scheduler inputs are all explicitly `0`. This stays on generic MTP D3.
- Existing scenario `require_gpu_chain=true` remains unchanged for provenance. V323 interprets that legacy assertion as “no active lane ends without any speculative proposal.” A K4V hit satisfies that proposal and can validly produce zero MTP draft steps. Output counters/log labels say `speculative_proposal`, not `GPU chain`, so they cannot be misread as MTP execution evidence.

## Per-type selection evidence

The runner emits `common_speculative_print_stats()` snapshots at each scenario boundary for each bank, using the existing library counters. Measured execution stays at `LOG_LEVEL_INFO`; only a stats snapshot temporarily raises the threshold to `LOG_LEVEL_TRACE`, then restores the prior value. The one pre-model-load TRACE interval remains solely for backend activation diagnostics. Existing per-type lines expose calls, `#gen drafts`, `#acc drafts`, `#gen tokens`, and `#acc tokens` for `ngram-map-k4v` and `draft-mtp`. Snapshots are cumulative; compare successive snapshots to isolate native-canary and real-corpus scenario deltas. These are the implementation's existing statistics, not new or inferred fields.

A hit is demonstrated only by `ngram-map-k4v #gen drafts > 0` in the real-corpus performance-scenario delta. If it is zero, report that the ngram path did not hit and make no ngram performance-gain claim. If MTP has zero generated drafts while K4V has generated drafts, that is an all-ngram proposal case, not a chain failure. If both methods yield no proposal for an active lane, the retained scenario guard fails. The original `common_speculative_print_stats` line determines which implementation generated and accepted drafts.

`ngram-cache` is deliberately absent: its common maximum-length resolution hardcodes 8, exceeding the D3 verify budget.

## Static and hostlink evidence

```sh
python3 fixtures/fourbank-d3-newruntime-v306-20261017/verify-v323-static.py
bash -n fixtures/fourbank-d3-newruntime-v306-20261017/root-host-link-v323.sh
python3 -m json.tool fixtures/fourbank-d3-newruntime-v306-20261017/v323-root-profile.json >/dev/null
```

The hostlink script has its own `V323_HOSTLINK_AUTHORIZED=1` gate and verifies V318 build ID/report SHA, exact source hash, type order, ngram value size, MTP depth, disabled sole-only/unroll options, and `launch_eligible=false`. It compiles a new binary into `build-root-v323`; it does not launch it.

`results/v323-hostlink-receipt.json` records runner SHA `66d5070bb63ab5eea37eea2ceca43e81e252e246832efc55b5433a55bd2f8ab6`, V318 report identity, resolved DSO hashes, and `model_loaded=false`, `gpu_touched=false`, `launch_eligible=false`. Static assertions and shell syntax passed after the scoped statistics-verbosity change.

The remaining gate is a separately released model/GPU run. No current speed or ngram-hit claim is made.
