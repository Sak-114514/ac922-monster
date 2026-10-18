# DFlash F32 encoder-FC CUDA consumer probe

This fixture loads the DFlash GGUF with the existing mmap CPU loader, validates
the real `fc.weight` tensor metadata, and transfers only that one tensor to a
single V100. It does not load the target model or copy the other 57 DFlash
tensors to GPU.

## Tensor and runtime gates

- Real tensor name: `fc.weight`; GGML dimensions `ne0=12800`, `ne1=2560`, type
  F32, contiguous, exactly 131,072,000 bytes.
- CUDA visibility is pinned to physical GPU2 UUID
  `GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb`.
- The fixture uses the selected runtime's `ggml_backend_load_all_from_path`,
  requires `MUL_MAT` output to be assigned to CUDA, and checks `/proc/self/maps`
  to ensure `libllama`, `libggml-base`, and `libggml-cuda` all came from that
  runtime directory.
- Expected token widths: T=1, 7, 28, and 512. The inputs are deterministic,
  nonzero F32 values.

## Build and bounded run

```sh
bash fixtures/dflash-f32-fc-20261015/build.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v25/lib

bash fixtures/dflash-f32-fc-20261015/run-bounded.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v25/lib \
  GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb
```

The runner refuses a busy GPU, pre-existing allocation over 128 MiB, or GPU
temperature at/above 58 C. It profiles each width separately, samples
temperature while running, records the Nsight CUDA kernel summary, and leaves
the target model untouched.

## Validation performed (2026-10-15)

Each width passed 27 sampled output checks. The fixture computes CPU F64 and
serial CPU F32 dot products for nine output rows and up to three token columns;
GPU F32 error must be within `max(2e-6, 2 * measured_CPU_F32_error + 2e-6)`.
Each width also passed exact zero-input output, bitwise A/B/A replay, and host
output-copy leading/trailing canaries.

| Tokens | Replays | Host wall avg | Nsight kernel | GPU kernel time |
| ---: | ---: | ---: | --- | ---: |
| 1 | 8 | 184.032 us | `mul_mat_vec_f<float,float,T=1>` | 161.627 us |
| 7 | 8 | 279.120 us | `mul_mat_vec_f<float,float,T=7>` | 258.364 us |
| 28 | 8 | 274.586 us | `volta_sgemm_128x32_sliced1x4_tn` | 240.093 us |
| 512 | 3 | 2,584.382 us | `volta_sgemm_128x128_tn` | 2,548.376 us |

One-time upload of the FC matrix took approximately 30.4 ms. Maximum GPU-vs-F64
sampled absolute error was 1.72e-7, 3.54e-7, 6.95e-7, and 5.49e-6 for T=1, 7,
28, and 512 respectively. All runtime mappings resolved to `runtime-perf-v25`.
GPU2 was 38–39 C during the probe and returned to 0 MiB / no compute process.

Result artifacts:

- T=1: `/home/debian/ac922-monster/experiments/standard-opt-20261014/results/dflash-f32-fc-20261015T132729Z/`
- T=7: `/home/debian/ac922-monster/experiments/standard-opt-20261014/results/dflash-f32-fc-20261015T132808Z/`
- T=28 and T=512: `/home/debian/ac922-monster/experiments/standard-opt-20261014/results/dflash-f32-fc-20261015T132852Z/`

This establishes that the isolated F32 FC operator executes on SM70 and gives
local kernel/host timings. It does not establish whole-DFlash encoder latency,
MTP acceptance, or model-level quality.
