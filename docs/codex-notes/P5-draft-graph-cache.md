# P5: bounded MTP draft graph-family cache

## Implementation

- Opt-in: `LLAMA_MTP_DRAFT_GRAPH_CACHE=1`; per-process diagnostics: `LLAMA_MTP_DRAFT_GRAPH_CACHE_LOG=1`. Both flags are read once at process start; the cache object is allocated only when the opt-in flag is true. Default behavior stays off.
- Cache eligibility is limited to decoder MTP graphs with 1–12 tokens and at least one output, outside pipeline parallelism, graph-reuse-disable mode, and hardware test-policy runs.
- The cache holds up to four entries. At capacity, a new family bypasses the cache; it never evicts an existing graph. Lookup requires both the 64-bit hash and the complete family vector to match.
- The family vector captures model/graph identity, scheduler and backend identities, all context-parameter values (including float bit patterns), adapter/cross-attention/MTP carry pointers, sampler identities, output layout, token/sequence shape and IDs, and the pooling sequence map. Dynamic KV/GDN and other input validity remains gated by `llm_graph_result::can_reuse()`.
- The cache keeps `llm_graph_result` metadata per family while the scheduler owns one active allocation layout. Restoring an inactive family synchronizes the scheduler, resets it, allocates that family's graph on the same scheduler, validates the final dynamic inputs, then refreshes its inputs.
- The allocation signature records tensor identity/type, backing buffer/data/extra pointers, view source/offset, dimensions, and strides. A changed family allocation signature advances the generation and prunes other entries; backing-buffer changes or incompatible buffer signatures also prune stale entries.
- Scheduler reservation, direct `graph_reserve()` scheduler rebinding, and a successful memory update invalidate cached families. These invalidations drain outstanding work before dropping graph metadata. Cache helper functions receive cache and active-result references explicitly; no free helper reaches into `llama_context` private members and no public API was added.

## Verification performed

- `fixtures/P5-context-mock.cpp`: compiled with `c++ -std=c++17 -O2 -Wall -Wextra`, then run with `CUDA_VISIBLE_DEVICES=`. Passed A/B/A selection, forced hash collision separation, dynamic-validity rebuild, generation invalidation, and four-entry bypass without eviction.
- `fixtures/P5-scheduler-aba.cpp`: compiled against the existing `standard-opt-20261014/build-reuse/bin` GGML libraries, then run with `CUDA_VISIBLE_DEVICES=` and `OMP_NUM_THREADS=1`. Passed: output A 42.0 -> 42.0 after A/B/A restore, B -165.0, same backing buffers, distinct family allocation signatures, and stable A allocation signature after restore.
- `llama-context.cpp`: one translation-unit `-fsyntax-only` check using the existing `build-reuse/compile_commands.json` flags with paths redirected to the isolated copy. Passed; emitted only pre-existing signed/unsigned comparison warnings in MTP helper/decode code.
- The complete `notes/P5-final.patch` passed `patch --dry-run -p1` against `hardware-next-20261014/source-before`.

The CPU scheduler fixture compile/run command from the experiment root was:
```sh
/usr/bin/c++ -std=c++17 -O2 -Wall -Wextra -I src/src -I src/include -I src/ggml/include -I src/ggml/src -I build-reuse/src fixtures/P5-scheduler-aba.cpp -L build-reuse/bin -Wl,-rpath,"$PWD/build-reuse/bin" -lggml-cpu -lggml-base -lggml -o fixtures/P5-scheduler-aba
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 fixtures/P5-scheduler-aba
```
The context-policy fixture uses `/usr/bin/c++ -std=c++17 -O2 -Wall -Wextra fixtures/P5-context-mock.cpp -o fixtures/P5-context-mock`, then runs with `CUDA_VISIBLE_DEVICES=`.

## Limits

- No full build, model, CUDA/GPU execution, server request, logits parity, or semantic-output check was run for P5.
- The context mock checks cache policy semantics independently, not the production class. The scheduler fixture exercises the production allocation-signature helpers and same-scheduler CPU A/B/A rebind; it does not prove CUDA or Meta-backend allocation stability.
- Enable this only for a later controlled runtime validation. Compare uncached and cached output on identical prompt sequences, exercise A/B/A under the actual draft context, and inspect cache build/restore logs before drawing performance conclusions.

## Patch artifact

- Complete source patch relative to `hardware-next-20261014/source-before`: `notes/P5-final.patch`.
- P5 fixtures are `fixtures/P5-context-mock.cpp` and `fixtures/P5-scheduler-aba.cpp`.
