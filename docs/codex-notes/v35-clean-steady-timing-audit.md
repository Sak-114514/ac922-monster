# V35 clean steady timing and instrumentation audit

Date: 2026-10-16  
Scope: read-only inspection of the saved V35 one-bank/four-lane run and its exact runtime/source. No build, GPU work, or configuration change was made.

## Result

The V35 `clean` family did **not** record CUDA event spans, run CUPTI, or use Nsight Systems. `GGML_META_TIMING=1`, `LLAMA_STEP_TIMING=1`, `GGML_CUDA_GRAPH_STRUCTURAL_TRACE=1`, `GGML_CUDA_GRAPH_PATH_LOG=1`, and `GGML_OPTIMIZATION_PATH_LOG=1` were nevertheless still set. The first two are host timers; structural trace is software diagnostics, not a CUDA-event synchronizer. The separate path logger does add per-dispatch software counters under a shared mutex and periodic log output. These settings make “clean” mean no CUPTI/NSYS/full-logit/GPU-span instrumentation in this harness, not instrumentation-free.

The requested nested composition was **not eligible** in this run. `GGML_META_NESTED_COMPOSE=1` was present, but `GGML_META_NESTED_ASYNC` was absent. The Meta constructor only enables the asynchronous nested path when that opt-in, fragment cache, replay guard, context views, and the six-rank P2P plugin are all present. Composition eligibility then depends on that async state. There are no `meta-nested-compose` replay or summary lines in the clean-family log. Thus the trace flags did not disable composition; this run did not turn on the prerequisite async path.

The clean four-lane measurement itself shows a steady target graph-compute host interval around 150 ms, with about 1.3 ms in `set_inputs`; the representative all-four target interval had no graph rebuild or allocation. The timing is host wall around graph dispatch/completion, **not GPU kernel-busy time**. The saved data cannot assign the 150 ms to kernels, inter-GPU waits, or host scheduling without a valid GPU activity trace.

## Saved-run evidence

Run tag: `root-p4-v35-onebank4-v80-ep6-d3-16k-20261016t073508z`  
Runtime log: `fixtures/dual-context-v35-onebank4-combined-20261016/results/root-p4-v35-onebank4-v80-ep6-d3-16k-20261016t073508z/runtime.log`  
Clean results: same run directory, `copy1/clean/results.json` and `copy1/clean/run-manifest.json`.

- Clean scenario `v35-onebank4d3-copy1-clean-real8k-p4-1x4-clean`: 866 shared emitted tokens over 11.434217 s, `shared_tg=75.7375866` tok/s.
- The clean manifest says `profiled=false`, `cupti_window.phase=off`, `gpu_spans_requested=false`, and `clean_performance_eligible=true`.
- The exact V35 environment still sets `GGML_META_TIMING=1`, `LLAMA_STEP_TIMING=1`, `GGML_CUDA_GRAPH_STRUCTURAL_TRACE=1`, `GGML_CUDA_GRAPH_PATH_LOG=1`, `GGML_OPTIMIZATION_PATH_LOG=1`, `GGML_CUDA_GRAPH_STRUCTURAL=1`, and `GGML_META_NESTED_COMPOSE=1`; it does not set `GGML_META_NESTED_ASYNC`.
- Clean steady frames 4–72: 69 completed frames with four active lanes and 16 target rows. Frame 54 is a target-wall outlier at 306.337 ms. Across all 69 frames, draft phase (`draft_us`) median 12.234 ms, p90 12.550 ms; target wall (`full_target_us`) median 147.516 ms; full step wall median 159.717 ms. Excluding only frame 54, the means are 12.264 ms draft, 147.897 ms target, and 160.162 ms full step. The target/full-step completion timestamps include the explicit target synchronize in the harness.
- Target `LLAMA_STEP_TIMING` sample in the all-four, 16-token verify shape: `rebuilt=0`, `build=0`, `alloc=0`, `inputs=1315 us`, `compute=150319 us`, `total=151681 us`. In `llama-context.cpp`, `inputs` times `res->set_inputs`; `compute` times `graph_compute`. These are host elapsed times, not kernel durations.
- There is no same-shape, all-four steady MTP `inputs/compute` sample in this log. The closest MTP line (`inputs=99 us`, `compute=11669 us`, `total=16478 us`) is after the four-lane window, with variable token rows 1–10. It must not be used as an all-four draft breakdown. `draft_us` is the available direct all-four draft-phase wall measurement.
- The log contains no `meta-gpu-span` samples and no `meta-nested-compose` selection/replay/summary records. CUDA graph path summaries show `structural_active=1`, `env_disabled=0`, and actual replays; e.g. a clean-family target context reached 6,400 path passes with 4,869 replays. These path counters are cumulative by CUDA context, not a per-request measurement.
- At clean-context teardown the structural-graph stats reported about 6.18–6.33 s of `signature_build_us` and 0.46–0.52 s of `lookup_us` per target CUDA device across the **whole clean family**, not just the 11.4 s shared window. This is evidence of nontrivial CPU-side structural-graph work, but it is not attributable to the `TRACE` toggle alone: signature build/lookup and their counters are part of the enabled structural graph path. No A/B here quantifies their impact on `shared_tg`.

## Source gates and interpretation

- In `fixtures/dual-context-v35-onebank4-combined-20261016/dual-context-v35-onebank4.cpp:1078-1088`, the runner changes `GGML_META_GPU_SPANS` and its sample period to `0` before constructing each clean context family. `ggml-backend-meta.cpp:3809-3829` only resolves/initializes span callbacks for value `1`; its per-dispatch `ggml_meta_span_scope` at `6069-6070` has no begin/end callbacks in clean. CUDA event creation/record/query/synchronize is in `ggml-cuda.cu:3552-3593` and is reached only through those callbacks. This is why GPU spans are absent in clean.
- `GGML_META_TIMING` at `ggml-backend-meta.cpp:5442-5447` calls `ggml_time_us()` and accumulates host rebuild/dispatch durations; `log_timing` at `6048-6065` prints every 200 calls. It does not create CUDA events or call a synchronize API.
- `GGML_CUDA_GRAPH_STRUCTURAL_TRACE` is read in `ggml-cuda.cu:1140-1149` for the context-teardown statistics, and in `4150-4163` for bounded rejected-graph messages. It does not select the graph path. Structural signature checks are selected by `GGML_CUDA_GRAPH_STRUCTURAL=1` and contain their own timing/counter work. Separately, `GGML_CUDA_GRAPH_PATH_LOG=1` gates `ggml_cuda_graph_log_path` (`1086-1133`), which locks the structural registry, increments counters, and emits a summary every 256 path calls.
- The nested async prereq is read at `ggml-backend-meta.cpp:3768-3776`; `nested_async_enabled` requires async opt-in, fragment cache, replay guard, Meta context views, and P2P6 callbacks. Compose eligibility at `6161-6171` requires that state, a parent generation, the P2P6 handle, and six leaves. `GGML_META_NESTED_COMPOSE=1` alone is insufficient.
- `llama-context.cpp:3325-3363` brackets `set_inputs` and `graph_compute`; the host-side timing emission is at `3367-3377`. The harness records `draft_us` at `dual-context-v35-onebank4.cpp:702-731`, and target/full-step wall after `llama_synchronize` at `735-737` and `863-868`.

## Conclusion and boundary

The timing flags in the V35 clean run do not explain a 150 ms step by introducing CUDA event synchronization, and the available logs show graph replay remained active. Host instrumentation is still present, especially the separately enabled graph path logger and structural signature work; it could add CPU overhead, but this run does not measure the counterfactual without it. The absent nested-async opt-in is the concrete reason composed graphs were not selected. Do not attribute the V32 136.149 tok/s versus V35 75.7376 tok/s difference to these flags without matching the shape/topology and a logging-only A/B. No GPU-busy percentage is claimed.
