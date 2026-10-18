# P4 selective Meta replay guard: CUDA 3+3 fixture evidence

Date: 2026-10-15
Status: bounded CUDA correctness fixture PASS on frozen runtime-perf-v13. No model or performance test.

## Runtime provenance and topology

The fixture binary links only the frozen runtime-perf-v13 GGML libraries; its process map and ldd output are in fixtures/meta-replay-guard-cuda-20261015/run-001.log. The loaded driver is libcuda.so.550.54.15, CUDA runtime is 12.4. The exact explicit UUID order mapped to:

- island 0: PCI 00000004:04:00.0, 00000004:05:00.0, 00000004:06:00.0
- island 1: PCI 00000035:03:00.0, 00000035:04:00.0, 00000035:05:00.0

Four independent CUDA Meta contexts initialized successfully, each with in-graph P2P AllReduce enabled over its three-member island. Each context scheduler routed two active Meta splits, one per island.

## Test result

The fixture uses two separate contexts and schedulers with shared static weights, a small F32 add crossing the island boundary, and a scalar output. All output reads are guarded by host canaries.

- Fresh A and B graphs were not replay-ready at allocation or after the first deferred Meta dispatch.
- Their second serialized dispatch merged/captured the graph and made every active split replay-ready.
- A hot replay passed before the B shape change.
- B switched to a fresh 8-row view while a guarded second capture ran; A hot replay was submitted concurrently during that capture. Both outputs matched exactly and both canaries remained intact.
- Scheduler reset denied the old B graph. A fresh 16-row B graph was then allocated and captured. A still replayed exactly afterward.

Final line in run-001.log:

```
CUDA 3+3 shared-weight/two-context replay-guard fixture: PASS
```

Post-run nvidia-smi showed about 1 MiB used per GPU and temperatures of 40–41 C. The GPU grant is released.

## Limits

This is an isolated CUDA/Meta ownership and readiness fixture, not a full llama_context/server correctness proof or performance result. The outer runner guard remains in place pending root review and the full-model gate.
