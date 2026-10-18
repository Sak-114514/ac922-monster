# Narrow batched GET_ROWS V80 bounded GPU3 comparison

- Runtime: `runtime-perf-v80` (`20261016T070400.271089Z`), GPU3 only, no model loaded.
- Fixture source SHA256: `10283557e8892637918c64de3ab5f3eb961b755083c967cd42d287b607d69877`.
- V80-linked fixture SHA256: `9ee493c0a25ba4008956631cf293320c6582a77b9a73df2700b54f963df2e3e2`.
- `libggml-cuda.so.0` SHA256: `5255753a5896650c18d2cef49c974b5fd39e65e8bef5daec8be1c56d406b0f9d`.
- Raw log: `narrow-getrows-gpu3-v80-20261016T071728Z.log`, SHA256 `902e71f6917795de960c51004b61a03b7307b6c1bd5f049fe732027698f86e7d` (local and AC922 copies match).

## Correctness and dispatch

The default-off CPU source oracle passed all seven shapes. The GPU generic arm and narrow opt-in arm each passed all seven shapes, one correctness warmup per shape, and three timed repetitions per shape. Every output was bitwise equal to the independent scalar cast/copy oracle, finite, and had intact source/ID padding canaries. Both arms emitted `GPU_ARM_MATRIX PASS` with `cases=7 warmups=7 timings=21`.

Default-off reported no `get_rows_narrow` marker. The opt-in process logged `path=get_rows_narrow selected=1`; its summary was `requested=10 selected=10` (host dispatch/capture observations, not replay kernel counts). V80's narrow gate is below the same-type vector early return; the width-4 F32 vector-control case had no narrow marker. The production mask shape was F16 source `[1,2112,1024,1]`, IDs `[2051,1024,1]`, F32 output `[1,2051,1024,1]`, 2,100,224 values.

## Timings

CUDA graph-compute plus synchronize milliseconds, one warmup then three recorded samples. These are tiny operator-fixture timings, not model prefill or production performance claims.

| Case | Generic off: 3 samples (mean ms) | Narrow on: 3 samples (mean ms) |
|---|---:|---:|
| Production F16→F32, width 1, 2051×1024 | 3.750 / 3.501 / 3.479 (`3.577`) | 0.369 / 0.128 / 0.108 (`0.201`) |
| F32→F32, width 1 | 0.148 / 0.075 / 0.076 (`0.100`) | 0.101 / 0.033 / 0.032 (`0.055`) |
| F16→F32, width 2 | 0.093 / 0.049 / 0.048 (`0.063`) | 0.063 / 0.032 / 0.032 (`0.042`) |
| F32→F32, width 2 | 0.097 / 0.053 / 0.055 (`0.068`) | 0.063 / 0.031 / 0.032 (`0.042`) |
| F16→F32, width 4 | 0.082 / 0.041 / 0.040 (`0.054`) | 0.063 / 0.031 / 0.031 (`0.042`) |
| F32→F32, width 4, below 256-row gate | 0.121 / 0.050 / 0.047 (`0.073`) | 0.063 / 0.035 / 0.036 (`0.044`) |
| F32→F32, width 4 vector control | 0.114 / 0.061 / 0.060 (`0.078`) | 0.081 / 0.053 / 0.052 (`0.062`) |

For the synthetic production-shape mask gather, the mean was 3.577 ms generic vs 0.201 ms opt-in (about 17.8× in this bounded fixture). This does not predict end-to-end prefill speed.

## Safety and scope

Pre-run gate used the active read-only file `notes/boot-watch-20261016T003420Z/bmc-telemetry.jsonl`: latest pre-run row had host Running/chassis On, targets all 3000 RPM, fan RPM 2856/2886/2853, no alarm/fault, masks valid, max/air 41°C. After the fixture, the same active JSONL showed targets 3000 RPM, fan RPM 2849/2873/2851, no alarm/fault, masks valid, max/air 41°C.

Post-run `nvidia-smi`: all six GPUs idle at 39°C, 3–8 MiB used; no fixture process remained. No model load, core change, BMC write, fan-control change, or GPU reset occurred.
