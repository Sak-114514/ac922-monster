# V220 binding-slot gate on V218

**Result: PASS** for the authorized tiny four-copy scheduler gate. This is a fixture-level CUDA graph result, not a model or throughput result.

## Run

- Frozen runtime: `runtime-perf-v218`, build ID `20261017T063007.552385Z`.
- Runtime report SHA256: `f8137bb427a99f948013313d9964064d969d28650a558a8b48bb1811898fff5b`.
- `ggml-cuda.cu` SHA256: `e276dc36eab95aee34192a67294673f35a4d6c0cf74e7d333b24965daf1b4cd6`.
- Fixture source SHA256: `9127db266c59582dad2e2bb233954e01366cf5068a8d411f3288704509e9513a`.
- Host-linked binary SHA256: `37de332c684c09e65c0aec5f3461b96a762e419676aa113ddb5530c096d9e76a`.
- Run tag: `v220-v218-binding-slots-20261017T063446Z`, logical CUDA devices 0 and 1.

The binary completed with `V220_PASS`. The raw log shows 20/20 exact CPU-oracle outputs and intact canaries; one shared metadata context and stable graph/mm0/mm1 descriptor addresses across the 20 frames; 20 distinct scheduler graph UIDs; source overwrite after every asynchronous submission; four-key pools on both CUDA contexts; and 80 pinned input copies with zero declines. Each device recorded 4 graph captures and 12 replays, with 12 cross-descriptor replays and 20 structural signature builds.

The first runner invocation's checker stopped on an incorrect expected staging-copy count of 40. The raw log showed 80 (the graph stages three CUDA0 inputs and one CUDA1 input per frame). I corrected that assertion and reran the checker offline against the same raw log. Both the local and AC922 checker runs passed; CUDA was not rerun.

## Artifacts

- Raw run log: `v220-v218-binding-slots-20261017T063446Z.log` (SHA256 `d66e5dc7e2b028cf613442b6fedef21693a6b32a59f16f2a6861d38358e0be32`).
- Corrected offline checker output: `v220-v218-offline-checker.log`.
- Host link log: `host-link-v218-r2.log`.
- Runtime report, four-DSO `ldd -r` output, provenance hashes, and binary: `build-v218-r2/`.
- Initial checker failure output: `run-v218-supervisor-20261017T063446Z.log`.

## GPU preflight and postflight

Immediately before and after the run, all six cards were at 38–39°C with 16,109–16,124 MiB free, 0% utilization, and no compute processes. The test used only logical CUDA devices 0 and 1. No CUDA fault occurred.
