# V270 dualhalf operator fixture results

Date: 2026-10-17  
Status: bounded operator measurements complete for seven formats at least once; full small matrix and full-N pilots intentionally incomplete after the first IQ4_NL correctness failure.

## Scope and runtime

- Device: GPU3, Tesla V100-SXM2-16GB, SM70.
- Frozen runtime: `/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v266`, build report SHA-256 `cedc737412da0e04b3468ad08eab8c0be0153aa13eed02805c0595d0af550a78`.
- Final fixture binary SHA-256: `8adf172f4b50fda1616f1579923ca3b292c49320b33a6deec8ef0c86fdb27965` (compiled after the final IQ4_NL decoder fix).
- Final custom CUDA object SHA-256: `92ec68acef1fa3487b9a737f336cfea1ef9e2fd1f9ef07fc76c135c2c27062da`.
- `ldd` resolved `libggml-cuda.so.0`, `libggml-base.so.0`, `libggml.so.0`, and `libggml-cpu.so.0` from runtime-perf-v266. CUDA 12.4 libraries resolved from the system CUDA path.
- No model was loaded. Inputs are synthetic F32 vectors with 32 deterministic patterns repeated by token. Weights alternate exact raw rows from real expert170/expert171 samples; each of the seven sample files matched its V236 manifest shape, byte length, and SHA-256.
- Each valid case required the real `mmq_moe_grouped selected=1` log, a finite baseline and custom outputs, and custom RMS error below grouped MMQ versus the CPU FP32 reference. Correctness was checked before timing for each case. The first failed case stopped the matrix.

All timings below are milliseconds. Grouped MMQ is full `MUL_MAT_ID` graph wall time, including dispatch/preparation and matmul. Custom timing is CUDA-event time for the kernel only; CPU route preparation is excluded, so this comparison favors the custom kernel. These are operator-fixture results, not model speed.

## Valid cases

| Format | T | Route | MMQ full graph | Custom 3-term kernel | Custom 4-term kernel | FP32 RMS: MMQ / 3-term / 4-term |
|---|---:|---|---:|---:|---:|---|
| IQ2_XXS | 256 | uniform10 | 0.5501 | 1.2513 | 1.2452 | 6.5606e-4 / 3.3311e-7 / 3.3314e-7 |
| IQ2_XXS | 256 | allhot10 | 0.1541 | 0.4864 | 0.4844 | 6.5606e-4 / 3.3311e-7 / 3.3314e-7 |
| IQ2_XXS | 512 | uniform10 | 0.5961 | 1.3558 | 1.3455 | 6.5606e-4 / 3.3311e-7 / 3.3314e-7 |
| IQ2_XXS | 512 | allhot10 | 0.2403 | 0.9390 | 0.9339 | 6.5606e-4 / 3.3311e-7 / 3.3314e-7 |
| IQ2_XS | 256 | uniform10 | 0.5977 | 1.2554 | 1.2493 | 8.0521e-4 / 3.7684e-7 / 3.7677e-7 |
| IQ2_XS | 256 | allhot10 | 0.1613 | 0.4884 | 0.4854 | 8.0521e-4 / 3.7684e-7 / 3.7677e-7 |
| IQ2_XS | 512 | uniform10 | 0.6334 | 1.3609 | 1.3548 | 8.0521e-4 / 3.7684e-7 / 3.7677e-7 |
| IQ2_XS | 512 | allhot10 | 0.2546 | 0.9400 | 0.9339 | 8.0521e-4 / 3.7684e-7 / 3.7677e-7 |
| IQ2_S | 256 | uniform10 | 0.6010 | 1.2698 | 1.2657 | 6.4444e-4 / 3.0982e-7 / 3.0995e-7 |
| IQ2_S | 256 | allhot10 | 0.1620 | 0.4905 | 0.4884 | 6.4444e-4 / 3.0982e-7 / 3.0995e-7 |
| IQ2_S | 512 | uniform10 | 0.6405 | 1.3742 | 1.3681 | 6.4444e-4 / 3.0982e-7 / 3.0995e-7 |
| IQ2_S | 512 | allhot10 | 0.2519 | 0.9452 | 0.9411 | 6.4444e-4 / 3.0982e-7 / 3.0995e-7 |
| IQ3_XXS | 256 | uniform10 | 0.5623 | 1.2524 | 1.2483 | 6.4915e-4 / 2.7566e-7 / 2.7566e-7 |
| IQ3_XXS | 256 | allhot10 | 0.1555 | 0.4864 | 0.4844 | 6.4915e-4 / 2.7566e-7 / 2.7566e-7 |
| IQ3_XXS | 512 | uniform10 | 0.6006 | 1.3578 | 1.3507 | 6.4915e-4 / 2.7566e-7 / 2.7566e-7 |
| IQ3_XXS | 512 | allhot10 | 0.2434 | 0.9380 | 0.9359 | 6.4915e-4 / 2.7566e-7 / 2.7566e-7 |
| IQ3_S | 256 | uniform10 | 0.5791 | 1.2790 | 1.2728 | 6.2557e-4 / 2.5737e-7 / 2.5728e-7 |
| IQ3_S | 256 | allhot10 | 0.1549 | 0.4905 | 0.4895 | 6.2557e-4 / 2.5737e-7 / 2.5728e-7 |
| IQ3_S | 512 | uniform10 | 0.6175 | 1.3885 | 1.3804 | 6.2557e-4 / 2.5737e-7 / 2.5728e-7 |
| IQ3_S | 512 | allhot10 | 0.2485 | 0.9492 | 0.9431 | 6.2557e-4 / 2.5737e-7 / 2.5728e-7 |
| IQ4_NL | 256 | uniform10 | 0.2325 | 0.4076 | 0.3983 | 3.2221e-4 / 6.6196e-8 / 6.6193e-8 |
| Q2_0 | 256 | uniform10 | 0.2276 | 0.3901 | 0.3768 | 3.0999e-4 / 4.4186e-8 / 4.4186e-8 |

The IQ2/IQ3 custom kernel was about 2.1–2.3× slower than grouped MMQ for uniform routing and 3.0–3.9× slower for allhot routing. IQ4_NL and Q2_0 follow-up cases were about 1.7× slower. The custom result was much closer to the unquantized-activation CPU reference, but it did not improve operator latency at these tested small shapes. No engine integration or speed claim is justified by this fixture.

## Stopped and corrected attempts

| Run | Outcome | Preserved evidence |
|---|---|---|
| `v270-small-v266-20261017-r1-01-IQ2_XXS-n64-t256-uniform10` | Process exited `-11` before fixture output. Root cause was a host array sized for 2 inputs but indexed by 32 activation patterns. | Its empty runtime log and one-sample guard remain in `results/`; no compute PID or timing. Fixed array length to `kInputPatterns`; AC922 ASan+UBSan CPU-reference-only run passed for the exact IQ2_XXS/T256 case (`elapsed_ms=88.163`, `cuda_initialized=0`). |
| `v270-small-v266-20261017-r2-01-IQ2_XXS-n64-t256-uniform10` | Baseline ran and stderr showed `mmq_moe_grouped selected=1`, but a harness callback gate did not see the direct `fprintf(stderr, ...)` marker and stopped before custom correctness/timing. | Preserved log and guard; no custom kernel or timing. Removed the unnecessary in-process logger gate; Python now validates the real captured stderr line and records invalid/no-speed-claim if absent. |
| `v270-small-v266-20261017-r3-21-IQ4_NL-n64-t256-uniform10` | Grouped MMQ was finite and selected; first custom IQ4_NL output had nonfinite values. Correctness gate stopped before any timing and cases 22–28 did not run. | Preserved log/guard. Root corrected the IQ4_NL 32-value block decoder, then the single IQ4_NL/T256 follow-up passed. |

The corrected matrix did not rerun the 20 successful IQ2/IQ3 cases. The two authorized follow-ups covered IQ4_NL and Q2_0 at N64/K640/T256/uniform10. Therefore, 22 valid cases cover all seven formats, but the 28-case matrix is partial: IQ4_NL T256/allhot and T512 both routes, plus Q2_0 T256/allhot and T512 both routes, were not run. No N=640 up or N=2560 Q2_0 pilot was run.

## Safety and verification

- Each case used the 58 C, 512 MiB free, competing-PID, 2 GiB allocation, and 120-second guards. Across 47 guard samples: maximum observed GPU temperature was 39 C, minimum free VRAM was 16,118 MiB, and no competing compute PID appeared.
- Final postflight: all six GPUs were 37–38 C with at least 16,106 MiB free and no compute-app PIDs.
- CPU route test passed for T=256/512 and both route patterns, including exact association coverage, J16 limits, duplicate rejection, and input/output association consistency.
- All fourteen expert170/expert171 raw files matched the sample-manifest SHA-256, byte length, and shape checks.
- AC922 host C++ syntax-only compile passed with `-Wall -Wextra -Wpedantic -Werror`; runtime-perf-v266 host-link and `ldd` resolution passed. Hostlink and CPU reference tests did not load a model.
