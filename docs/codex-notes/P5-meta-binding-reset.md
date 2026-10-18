# P5 Meta descriptor restore by scheduler-owned binding reset

## Change

P5 still keys graph families by their complete family data and cache generation. On an inactive-family restore it requires the cached result, generation, backing-buffer set, allocation signature, and recorded scheduler bindings to match before touching the graph. It synchronizes and resets the scheduler, clears only graph tensor `buffer`/`data`/`extra` fields captured before the original allocation, and allocates the cached graph again. This reruns backend `init_tensor` and lets Meta materialize descriptors into the active compute generation.

The capture deliberately accepts a null buffer even when `data` is already non-null: GGML views inherit their source data pointer at construction. After allocation it records scheduler COMPUTE tensors plus Meta-backed graph views. Such views can inherit a WEIGHTS buffer while their per-graph Meta descriptors live in the rotating compute generation. Already-bound model weights and cache/KV input tensors are not candidates and remain untouched. The fix never dereferences saved buffer handles; it validates raw binding fields and clears only graph tensor fields after the scheduler is drained. If identity or generation checks fail, P5 declines restore and follows the normal graph-build path. The feature remains opt-in and defaults off.

## CPU-only Meta A/B/A fixture

`fixtures/P5-meta-aba.cpp` wraps two CPU backend instances in one Meta device and runs A, B, then restores A through the same scheduler. It checks the stale A descriptor state after B, applies the production binding-reset helper, checks that every Meta node has a simple tensor on both members, verifies the prebound input retains the same buffer/data/extra fields, and compares restored A with a fresh A graph.

Command:

```sh
/usr/bin/c++ -std=c++17 -O2 -Wall -Wextra \
  -I src/src -I src/include -I src/ggml/include -I src/ggml/src -I build-reuse/src \
  fixtures/P5-meta-aba.cpp -L build-reuse/bin \
  -Wl,-rpath,"$PWD/build-reuse/bin" -lggml-cpu -lggml-base -lggml \
  -o fixtures/P5-meta-aba
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 ./fixtures/P5-meta-aba
```

Observed result:

```text
Meta A/B/A descriptor restore: PASS (A 72.0 -> 72.0, B -110.0, fresh-A 72.0, stale-missing=1/6, stale-view-missing=2, cleared=3, view-binding=1, input-preserved=1, missing-after=0, view-missing-after=0)
```

The fixture uses distinct A and B view shapes over one prebound WEIGHTS Meta input. It confirms stale A descriptors after B (`1/6` graph-node/member mappings missing, including both ranks of A's view), then confirms rematerialization and exact output after clearing three scheduler-owned bindings, including the data-prebound view. It runs only CPU backends; it is not a CUDA, full-model, logits-parity, performance, or production-service validation. P5 remains disabled until the root runs the full MTP path and reviews the fixture result.

## Root build handoff

After the current fused candidate releases the GPUs and the root integrates this private-source patch, build only the changed dependency closure and freeze a fresh runtime with the existing isolated builder:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
python3 incremental-build.py --run -j2 \
  --runtime-dir runtime-p5-viewfix-20261015 \
  --report notes/P5-viewfix-build.json
```

The build and full MTP validation were not run in this task. Keep `LLAMA_MTP_DRAFT_GRAPH_CACHE` off until the root explicitly starts the actual-model validation; enable its log flag for that run so A/B/A restore and reset counts are visible.
