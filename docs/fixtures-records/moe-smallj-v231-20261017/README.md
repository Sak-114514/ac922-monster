# V231 small-J grouped MMQ MoE fixture

This is an isolated `GGML_OP_MUL_MAT_ID` fixture for the opt-in `GGML_CUDA_MOE_SMALL_J` path. It compares separate fresh processes: baseline `GGML_CUDA_MOE_SMALL_J=0` with the normal global J32 setting implied by 2,560 assignments / 512 global experts, and candidate `GGML_CUDA_MOE_SMALL_J=1` with the new per-expert J16/J32 offset path. Both arms use `GGML_CUDA_MOE_GROUPED=1`; both explicitly leave `GGML_CUDA_MOE_GROUPED_J` and `GGML_CUDA_MMQ_MOE_J` unset. The old J32/J64 adaptive path is disabled.

The fixture reads original expert170/expert171 raw files from the existing V52 TC probe sample set. It takes the first 64 quantized rows from each actual expert tensor and alternates those exact bytes across a synthetic E=512 bank. This intentionally bounds memory and is not a full model or a diverse 512-expert bank. K remains the source tensor's real K (2,560 for gate formats and 640 for down formats); N is 64; activations are deterministic F32.

Each format process runs four route distributions at T=256, top-k=10, with unique IDs per token:

| Distribution | Expected nonzero expert counts | Purpose |
|---|---|---|
| `uniform5` | 512 experts at 5 each | all active experts should be cold/J16 |
| `skew_hot20` | 20 experts at 25–26, 492 at 4–5 | mixed cold and hot bins |
| `allhot10` | 10 experts at 256, 502 inactive | all routed experts should be hot/J32 |
| `boundary16_17_20` | 2 at 16, 1 at 17, 1 at 20, remaining active experts at 4–5 | checks the 16/17 boundary and hot20 stays J32 |

The fixture reports its CPU-computed expected cold/hot expert and tile counts as `CPU_EXPECTATION_ONLY`; it does not claim a GPU count readback. The candidate runner requires the actual runtime marker `path=mmq_moe_small_j requested=1 selected=1 reason=gpu_prefix_J16_cold_J32_hot_original_dot_kernels`. It saves every baseline F32 output and compares the candidate's entire output against that reference, reporting bit differences, max absolute/relative error, RMS, and non-finite values. No numerical tolerance or performance threshold is invented. Five warm full-graph wall samples include the `MUL_MAT_ID` ID helper, classify/prefix work, activation quantization, and grouped matmul.

Seven supported formats are sourced from the existing V52 raw sample loader output in `fixtures/monpack-ep-readiness-20261016/operator-fixture/tc-probe/samples/` (expert170 and expert171 subdirectories), certified by that fixture's `sample-manifest.json`: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, and Q2_0. No sample is requantized; no model is loaded; no OpenSSL or other new dependency is used.

Host-link and GPU launch are separately gated. Build/launch only after V231 runtime freeze and an explicit GPU window. `run-v231-format-pair.sh` starts one baseline process and then one candidate process for a single format. `run-v231-all-once.sh` runs the seven pairs sequentially and stops on the first failure; it never retries.

```sh
V231_BUILD_AUTHORIZED=1 \
RUNTIME_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v231 \
  ./build-hostlink-v231.sh
```

```sh
V231_GPU_WINDOW_GRANTED=1 DEVICE=3 RUN_TAG=unique-v231-tag \
  ./run-v231-all-once.sh
```
