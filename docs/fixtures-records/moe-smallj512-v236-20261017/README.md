# V236 T=512 small-J grouped MMQ MoE fixture

This is an isolated `GGML_OP_MUL_MAT_ID` fixture for the opt-in `GGML_CUDA_MOE_SMALL_J` path, exercising the new `GGML_CUDA_MOE_SMALL_J_MAX_TOKENS=512` eligibility extension. It compares separate fresh processes: baseline `GGML_CUDA_MOE_SMALL_J=0` with the normal global J32 setting implied by 5,120 assignments / 512 global experts, and candidate `GGML_CUDA_MOE_SMALL_J=1` plus `GGML_CUDA_MOE_SMALL_J_MAX_TOKENS=512` with the per-expert J16/J32 offset path. The baseline leaves `SMALL_J_MAX_TOKENS` unset. Both arms use `GGML_CUDA_MOE_GROUPED=1`; both explicitly leave `GGML_CUDA_MOE_GROUPED_J` and `GGML_CUDA_MMQ_MOE_J` unset. The old J32/J64 adaptive path is disabled.

The fixture reads original expert170/expert171 raw files from the existing V52 TC probe sample set. It takes the first 64 quantized rows from each actual expert tensor and alternates those exact bytes across a synthetic E=512 bank. This intentionally bounds memory and is not a full model or a diverse 512-expert bank. K remains the source tensor's real K (2,560 for gate formats and 640 for down formats); N is 64; activations are deterministic F32.

Each format process runs four route distributions at T=512, top-k=10, with unique IDs per token (5,120 assignments total):

| Distribution | Expected nonzero expert counts | Purpose |
|---|---|---|
| `uniform10` | 512 experts at 10 each | all active experts should be cold/J16; global baseline averages 10 assignments per expert |
| `skew_hot20` | 16 experts at 51 and 4 at 52; 492 experts at 8–9 | mixed cold and hot bins |
| `allhot10` | 10 experts at 512, 502 inactive | all routed experts should be hot/J32 |
| `boundary16_17_20` | 2 at 16, 1 at 17, 1 at 20, 479 at 10, 29 at 9 | checks the 16/17 boundary and hot20 stays J32 |

The fixture reports its CPU-computed expected cold/hot expert and tile counts as `CPU_EXPECTATION_ONLY`; it does not claim a GPU count readback. The candidate runner requires an actual `path=mmq_moe_small_j ... requested=1 selected=1` runtime marker. It saves every baseline F32 output and compares the candidate's entire output against that reference, reporting bit differences, max absolute/relative error, RMS, and non-finite values. No numerical tolerance or performance threshold is invented. Five warm full-graph wall samples include the `MUL_MAT_ID` ID helper, classify/prefix work, activation quantization, and grouped matmul.

Seven supported formats are sourced from the existing V52 raw sample loader output in `fixtures/monpack-ep-readiness-20261016/operator-fixture/tc-probe/samples/` (expert170 and expert171 subdirectories), certified by that fixture's `sample-manifest.json`: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, and Q2_0. No sample is requantized; no model is loaded; no OpenSSL or other new dependency is used.

Host-link and GPU launch are separately gated. Run only after the V235 runtime is frozen, the fixture is host-linked against that exact runtime closure, and a GPU window is explicitly granted. `run-v236-format-pair.sh` starts one baseline process and then one candidate process for a single format. `run-v236-all-once.sh` runs the seven pairs sequentially and stops on the first failure; it never retries.

```sh
V236_BUILD_AUTHORIZED=1 \
RUNTIME_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v235 \
  ./build-hostlink-v236.sh
```

```sh
V236_GPU_WINDOW_GRANTED=1 DEVICE=3 RUN_TAG=unique-v236-tag \
  ./run-v236-all-once.sh
```

## Recorded V236 run

Run tag `v236-t512-max512-v235-20261017T1845Z` completed on GPU3 with the V235 runtime loaded. The host binary was linked against the frozen V232 ABI, then run with V235 first in `LD_LIBRARY_PATH`; runtime `ldd` resolved `libggml-cuda.so.0` and `libggml-base.so.0` from V235. No CUDA object was rebuilt for the fixture.

All seven formats completed baseline and candidate, and each candidate logged `mmq_moe_small_j requested=1 selected=1` with an 8/8 selected summary. All 28 complete 327,680-element output comparisons were bit-exact with no non-finite values. The postflight found no compute PIDs; all GPUs were 38–39 C with about 16.1 GiB free.

The compact timing, per-log and raw-reference SHA-256 values are in `results/v236-t512-max512-v235-20261017T1845Z/gate-summary.json`; the full logs and baseline output references are in that same directory. These results validate the isolated synthetic-bank fixture only, not a full-model workload or model-level performance change.
