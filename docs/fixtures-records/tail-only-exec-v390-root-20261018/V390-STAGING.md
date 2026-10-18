# V390: captured-only device-tail graph launch (staged)

Status: **design and patch staged only**. Nothing in `src/` or the R389 runtime was modified; no V390 hostlink or GPU run was performed.

## Purpose

The V386 CUPTI sample recorded 60 main `cudaGraphInstantiate` calls (736 ms) and another 60 helper instantiations (751 ms) for the first 32K prefill chunk. V390 proposes eliminating only the unused standard main-graph executable on opted-in V100 prefill graphs. The R385 helper still owns and launches its child and controller executables. The fallback remains ordinary `cudaGraphInstantiate` + `cudaGraphLaunch`.

## Proposed integration

- `staged/ggml-cuda-v390-captured-only.patch`: CUDA backend caller changes against the exact V389 `ggml-cuda.cu` source hash recorded in `source-provenance.json`.
- `staged/ggml-cuda-v390-client-header.patch`: new private V2 dynamic-loader header. It has no `cudaGraphExec_t` parameter.
- `staged/bridge-api-v2.patch` and `staged/bridge-upload-r5.patch`: optional helper ABI V2, based on the V385/R385 upload-isolated helper. ABI V1 remains intact.
- `staged/ggml-cuda-v390-proposal.patch`: concatenated review packet; apply the per-file patches only after the owning source is reconciled with the then-current runtime.

The directory name `source-pin/r381` is historical. Its pinned `ggml-cuda.cu` hash is the exact V389 runtime source (`b575634b…`); the current local checkout still has the older R381 parent (`f878ce32…`). This proposal is deliberately not applied to that checkout.

The helper key is the **`ggml_cuda_graph *` object**, matching the launch path exactly; the raw scheduler `graph_key` is diagnostic only. A ready query never creates or launches work. A new or structurally changed capture passes `invalidate=1`, so a cached entry cannot be reused by address alone.

The candidate requires both `GGML_CUDA_GRAPH_DEVICE_TAIL=1` and the new `GGML_CUDA_GRAPH_DEVICE_TAIL_CAPTURE_ONLY=1`; it is limited to the existing prefill hint, Volta, CUDA graph, non-cold-cohort path. The new switch is opt-in and defaults off. V1 remains available for the existing MTP decode/default path.

## Bounded verification gates before a model run

1. Reconcile `ggml-cuda.cu` with the exact current build source and make a unified hunk merge; do not replace the TU. Verify all parent hashes and the V389 runtime manifest.
2. Hostlink the helper and caller against one frozen runtime. Confirm the new symbols and `ldd -r` closure before any model load.
3. Run the helper lifecycle fixture: cold ready=false; first launch prepares and queues captured child; same owner/key/capture ready=true and replay; changed capture invalidates/rebuilds; mode transition retires the prior entry; cap, unsupported node, and stream-change declines queue no model work; context drop drains the completion event. Keep the existing R385 upload/pending-consumer/exactness gates.
4. Run a guarded native semantic canary, then one bounded real prefill. Compare the qualifying main graph instantiate count against the V386 trace; helper child/controller instantiation is expected and must not be counted as a miss. Report the main and helper counts separately, along with completed-token/semantic result and all six temperatures.
5. On helper unavailability or admission decline, prove the standard executable is created and launched. A newly created standard executable is already instantiated from the current captured graph, so the proposal skips an immediately redundant `cudaGraphExecUpdate`; existing execs still take the normal update path when required.

No pressure test, kernel math change, graph-structure change, public API change, or `.cuh`/graph-layout edit is part of this proposal.

## Risks to resolve in integration

- This is a staged proposal, not compiled evidence. The captured-only path must be tested on the deployed CUDA 12.4/V100 stack before enabling it.
- The helper still instantiates child and controller graphs and owns their retirement. V390 does not remove those costs.
- The existing compose path may decline when `graph->instance == nullptr`; this is a performance path and must be checked against actual meta-backend readiness and stage scheduling. It must never make the result numerically different.
- Any asynchronous CUDA error must remain an error; only explicit helper admission declines may fall back to host graph launch.
