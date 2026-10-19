# V428 tail-stage timing candidate

This is a staging-only copy of the R5 captured-only helper with opt-in host
timing. `AC922_TAIL_STAGE_TIMING=1` must be set before process startup. With it
unset, the candidate takes no new clock readings and emits no V428 lines. The
pre-existing R5 `upload_isolated` line and CUDA behavior remain unchanged.

Enabled lines use monotonic start/end nanoseconds and identify API, owner, key,
captured graph, device, caller invalidation, cold/warm path, outcome, and status.
They report per-last-call and bounded aggregate sum/max microseconds for done
event retirement, child graph instantiation, controller graph instantiation,
upload drain, total prepare, and launch API host time. The process emits one
line for a slow phase/call (at least 1 ms) or every 64 calls; aggregate state is
global and fixed-size. No CUDA calls or their order are changed.

The source closure is one CUDA translation unit plus the ABI-v2 header. The
build helper is deliberately guarded and refuses to overwrite an output. It
uses the existing R5 SM70/relocatable-device-code/shared-runtime link shape,
including `cudadevrt`. This agent did not invoke it.

`host/verify_stage_timing.py` is a CPU-only format/provenance fixture. It checks
phase accounting, cold invalidation, warm replay, monotonic interval metadata,
and SHA256s. It does not build or execute CUDA code and cannot establish that
event synchronization explains the UB2048 host-time regression by itself.

## Evidence boundary

The copied baseline source hashes match the frozen R5 source reported for
`tail-only-exec-v390-root-20261018`; the pinned binary identity is recorded in
`source-provenance-v428.json`. No engine source, helper binary, model, or GPU
was changed or run here. R5 ABI-1 and ABI-2 exports remain source-identical.
