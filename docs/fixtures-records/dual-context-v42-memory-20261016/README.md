# V42 memory profile (host-only preparation)

V42 reuses the V41 target-UB4096/draft-UB512 harness binary while selecting the frozen V96 runtime. It adds leader-only Relay6 staging, unused-segment elision, and a CUDA graph variant cap of one. V42 is **not launchable** until its V96 AR fixture gate passes and root separately authorizes the bounded model window.

## Candidate settings

- Geometry: four lanes, D3 MTP, target ubatch 4096, draft ubatch 512; retain the V41 16K-per-sequence profile and all quality checks.
- Runtime: resolved from the selected build report (`V96-build.json` by default), not from a hard-coded runtime directory.
- Added environment: `GGML_CUDA_P2P_AR_RELAY_LEADER_STAGE=1`, `GGML_CUDA_P2P_AR_ELIDE_UNUSED_SEGMENT=1`, `GGML_CUDA_GRAPH_VARIANTS_CAP=1`.
- Existing safety bounds: one model load, no retry, maximum 900 seconds, 512 MiB free on every GPU, 58°C GPU/BMC ceilings, no reset. Fan/watchdog state is preserved.

## Host-only evidence

- `V96-build.json`: frozen runtime report.
- `ldd-v41-runner-under-v96.txt`: reused runner's dynamic dependencies resolve under the selected V96 runtime.
- `v41-runner-binary-sha256.txt`: fresh remote SHA-256 for the reused host runner.
- `v42-exact-env.json`: generated profile overlay.
- `v42-launch-manifest-dry-run.json`: exact environment and guard manifest; `model_loads=0`, `systemd_run_sent=false`, and `root_gpu_window_authorized=false`.
- `v42-micro-gates.json`: inherited V41 gate state plus the pending V96 relay-stage/segment-elision gate.
- `run-v42-flatmeta6.sh`: generated from the frozen V41 shell template using checked substitutions; the V96 flags are explicit, and runtime/report paths come from the selected profile.
- `run-v42-supervisor.py`: thin adapter that imports V41's V40 supervisor and rebinds the V42 profile, V96 runtime, and launcher. It retains the same monitor/supervisor gates.
- `write_run_manifest_v42.py`: imports the existing V41 manifest writer and adds V42 source and flag provenance.

The launcher is prepared locally but not yet synchronized to the server. The V96 gate is still pending, so the supervisor refuses launch until the profile is regenerated with the passing gate and the root authorization values are supplied.

## Regenerate

```bash
python3 monster-engine/standard-opt-20261014/fixtures/dual-context-v42-memory-20261016/make_v42_profile.py
```

The generator validates the frozen build report, the V41 runner's selected-runtime `ldd` closure and SHA, all required inherited V41 micro gates, and any V96 gate evidence. It never starts a model or touches the GPUs.
