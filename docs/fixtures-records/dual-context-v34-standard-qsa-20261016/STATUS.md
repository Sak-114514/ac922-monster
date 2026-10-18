# V34 standard QSA sparse run status (2026-10-16)

## Candidate and setup

- Frozen runtime: V78, build ID `20261016T061009.431203Z`; V34 host-linked runner uses the complete V78 library closure.
- Two banks × two lanes, D3, 16K per sequence, target UB1024, MTP UB512, target verify padding enabled.
- `LLAMA_QSA_INDEXER_IMPL=standard` and `LLAMA_QSA_STANDARD_SPARSE=1`; runtime markers confirmed `requested=1 selected=1` on decode and on the 8K probe.
- CPU async-copy maximum 10 MiB with the existing 64 MiB pinned budget; relay disabled.
- No Nsight, CUPTI, or `LD_PRELOAD` injection.
- BMC fan monitor state left as configured; watchdog active. Runtime telemetry is record-only for DDR; existing GPU 512 MiB, GPU/BMC 58°C, and fan safety guards remain active.

## Run result

One V34 model load completed. The 12 native semantic cases passed. The 8K diagnostic prefill probe completed and was compared with the V32 record in `results/V34-PREFILL-PROBE-V32-COMPARISON.md`.

The run started `real8k-p4-2x2`, then the supervisor requested a stop after a live GPU sample fell below the 512 MiB/card minimum (256 MiB minimum across the six cards). `systemd` reports exit status 0, but the supervisor records the safety guard, so the run is not a completed qualification. F32 preclean, exact-token/auxiliary gates, and clean-family results are absent. No clean performance result is claimed.

The final postflight returned all GPUs to idle with approximately 16.12 GiB free per card, 41–44°C, and no compute process. BMC reported chassis On, max temperature 45°C, masks OK, fan targets 3000 RPM, and no fan alarm/fault.
