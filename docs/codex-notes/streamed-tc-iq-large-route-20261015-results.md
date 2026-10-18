# Streamed TC high-route-density fixture results

Date: 2026-10-15  
Status: bounded GPU2 fixture passed. The Volta streamed-TC kernel remains slower than grouped MMQ at all tested route densities, though its relative gap narrows as assignments per active expert rise. Do not promote the current kernel to engine dispatch.

## Scope and setup

This is a standalone kernel fixture under `fixtures/streamed-tc-iq-large-route-20261015/`; it does not edit engine code or load a model. It uses the existing actual packed-row samples from `fixtures/volta-tc-real/`, with seven production quantized formats and the pinned MIT Strata schedule. The experiment compares the streamed weight-decode plus Volta MMA graph with grouped MMQ.

The synthetic routes use top-k 10 over all 512 experts at T=1024, 2048 and 4096:

| Tokens | Assignments | Balanced assignments/expert | One-hot expert 0 | Other experts |
|---:|---:|---:|---:|---:|
| 1024 | 10,240 | exactly 20 | 1,024 | 18–19 each |
| 2048 | 20,480 | exactly 40 | 2,048 | 36–37 each |
| 4096 | 40,960 | exactly 80 | 4,096 | 72–73 each |

The GPU fixture invokes the engine's `ggml_cuda_launch_mm_ids_helper` and tile-offset scans, then checks each generated route map against the CPU oracle. Gate/up-style broadcast cases use `quantize_scatter_mmq_q8_1_cuda`; down cases use `quantize_mmq_q8_1_cuda` with the forward compact-row map. Thus the MMQ end-to-end comparator includes engine route compaction, tile-offset setup and actual GPU Q8 preparation. The TC comparator includes its route compaction, TC tile-offset scan, F32-to-F16 activation conversion and TC graph. Both core-only timings are also retained.

## Verification

- Build passed with `-arch=sm_70`; emitted PTX contains Volta `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`.
- CPU-only route matrix passed exact count and forward/inverse-map invariants for all six route shapes.
- GPU run completed **60/60** cases with `FIXTURE_END status=PASS`, `rc=0`, and no errors.
- **30/30** packed-weight GPU decode checks matched the CPU GGML dequantize-then-RNE reference exactly, with zero mismatches across all seven formats and all sampled expert rows/shapes.
- **60/60** TC numeric checks passed the existing F16 and original-F32 tolerance gates. Maximum observed F16-reference NRMSE was `9.8468e-7`; maximum original-F32-reference NRMSE was `3.5081e-4`.
- **60/60** GPU route maps and offset arrays matched the CPU oracle exactly. All 60 Q8 preparation canaries, 180 output canary checks and 60 A/B/A graph replays passed; every A replay was bit-identical. Both output buffers were finite in every case.
- MMQ outputs were checked for finiteness, not full elementwise parity against an independent high-precision MMQ reference. Q8 values were produced by the engine's own quantizers and protected by canaries; they were not separately compared byte-for-byte with a CPU Q8 implementation.
- Watchdog: GPU2 UUID `GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb` only, 2048 MiB memory stop, 58 C temperature stop, 540-second timeout. Peak observed GPU2 memory was 1934 MiB; peak temperature was 43 C. All six GPUs returned to 0 MiB after the run.
- No model, server, GPU reset, system setting or engine source was changed.

## Timing summary

Values below are medians over the 10 format/shape cases at each token-count and route pair. A ratio above 1 means streamed TC takes longer than grouped MMQ. “Prep included” includes GPU route mapping, each path's offset scan, activation preparation and operator graph time; host-side fixture construction and synthetic input-ID transfer are excluded.

| Route | Tokens | Mean assignments/expert | TC/MMQ core median | TC/MMQ with GPU preparation median |
|---|---:|---:|---:|---:|
| balanced | 1024 | 20 | 5.54x | 4.73x |
| balanced | 2048 | 40 | 3.16x | 2.74x |
| balanced | 4096 | 80 | 2.74x | 2.35x |
| one-hot E0 | 1024 | 20 | 5.40x | 4.74x |
| one-hot E0 | 2048 | 40 | 3.26x | 2.97x |
| one-hot E0 | 4096 | 80 | 2.67x | 2.32x |

TC core lost all 60 comparisons. With preparation included, TC lost 59/60. The single apparent exception was the first `IQ2_XXS`, balanced 1024-token case: route setup measured 3.04 ms and Q8 preparation 8.19 ms, while subsequent medians were roughly 0.1 ms. This is an isolated first-case timing outlier; the one-shot run cannot determine whether it came from first-use initialization or another transient, so it is not evidence of a steady-state TC win.

The prior 512-token arm averaged 19.845 assignments per active expert and reported a 4.7–5.6x TC-core disadvantage. This higher-density matrix shows that disadvantage shrinking toward 2.7x at 40–80 assignments per expert, but no tested shape makes the current TC schedule faster than grouped MMQ. Its current dispatch integration is therefore rejected.

## Artifacts and reproduction

Remote artifacts:

- Fixture: `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/streamed-tc-iq-large-route-20261015/`
- Raw GPU log: `gpu2-large-route-run.log`
- Safety/temperature log: `gpu2-large-route-watch.log`
- CPU matrix: `cpu-prep.log`
- Build script: `build-fixture.sh`
- Bounded runner: `run-gpu2-bounded.sh`
- Prior low-density result, left unchanged: `/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/streamed-tc-iq-20261015-results.md`

Reproduction from the experiment root:

```bash
fixtures/streamed-tc-iq-large-route-20261015/build-fixture.sh
fixtures/streamed-tc-iq-large-route-20261015/streamed-tc-iq-large-route \
  fixtures/volta-tc-real --cpu-only --matrix
fixtures/streamed-tc-iq-large-route-20261015/run-gpu2-bounded.sh
```

The GPU runner selects GPU2 by UUID, refuses to start if any GPU has more than 64 MiB in use, and enforces its memory, temperature and wall-clock stops.
