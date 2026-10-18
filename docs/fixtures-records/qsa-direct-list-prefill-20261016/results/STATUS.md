# QSA direct-list and head-warps fixture results

Date: 2026-10-16. All GPU measurements used physical GPU3 (`GPU-fadb0459-4d30-907b-289a-862b3014f165`), one visible CUDA device, and no model weights. No core source was edited by this fixture task.

## V81 direct-list

Runtime: `runtime-perf-v81/lib/libggml-cuda.so.0`, SHA-256 `2f22091738127c31d1bdef499d9b55ec2e21e4cedac911fbce9e61923bdc15e6`. Fixture binary SHA-256 `b231c8dc398f259f862ec0279e71439b4b4cb2434f22b2c89ae2a0e661753d0a`.

The full five-case comparison passed on 2026-10-16 07:49:32 UTC. Direct-list was selected for 128/1024-token QSA cases and reported `auxiliary=0`; the 32-token case kept shared-list staging (`auxiliary=1664`). Full F32 outputs for both input seeds were bitwise identical between shared and direct arms. The shared-list arm's independent eight-token/all-head FP64 samples passed; both arms passed finite-output, repeat-exact, changed-input, and Q8 cache-tail canary checks. Path marker and summary counts matched host dispatch/capture behavior (two records per process, not per graph replay).

| Shape | Shared-list mean | Direct-list mean | Direct/shared |
|---|---:|---:|---:|
| GQA4, 32 tokens (fallback control) | 0.236 ms | 0.240 ms | 1.017× |
| GQA4, 128 tokens | 0.613 ms | 0.646 ms | 1.054× |
| GQA8, 128 tokens | 1.127 ms | 1.155 ms | 1.025× |
| GQA4, 1024 tokens | 5.810 ms | 3.882 ms | 0.668× |
| GQA8, 1024 tokens | 8.048 ms | 6.929 ms | 0.861× |

These are isolated operator graph-wall timings, not a model-prefill result. The 128-token cases were slightly slower, while the 1024-token cases were faster. The current `n_tok>=64` direct-list gate is therefore broader than the measured win; evaluate a more selective threshold before enabling it in production.

V81 logs and full-output binaries remain on AC922 at:
`/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/qsa-direct-list-prefill-20261016/results/qsa-direct-list-gpu3-v81-20261016T074932Z/`.
The first 07:48:39 run is also preserved there. It stopped because the fixture runner incorrectly expected five logger records; the actual logger counts dispatch/capture, so the observed count was two. The first tested kernel completed warmup, replay, changed-input, FP64, and canary checks. The runner was corrected and the successful full retry above passed; this was not a kernel failure.

## V82 head-warps

Runtime: `runtime-perf-v82/lib/libggml-cuda.so.0`, SHA-256 `2d457e002dd5f164fb5a709d77180fce1772e142e9e9ac6fc49fbe7927fa1ca1`. Fixture binary SHA-256 `c896ca92996ea9d0a818302a53fb73bc976771eb2f23515aececbb05cd4e433a`.

The complete comparison passed on 2026-10-16 07:53:23 UTC with direct-list fixed on. It covered GQA4/8 at 128 and 1024 tokens plus the production-shaped GQA4/256 case, using `GGML_CUDA_QSA_PREFILL_HPW=0,1,2`. Every requested path marker matched its selected HPW and warp count. Full outputs for both seeds were bitwise exact across HPW modes; FP64 samples at HPW0, finite/replay/change-input checks, and Q8 tail canaries all passed. The HPW0 32-token case retained the original grouping, direct-list fallback, and exact full output parity with V81.

| Shape | HPW0 | HPW1 | HPW2 | HPW1 / HPW0 | HPW2 / HPW0 |
|---|---:|---:|---:|---:|---:|
| GQA4, 128 tokens | 0.627 ms | 1.205 ms | 0.836 ms | 1.922× | 1.333× |
| GQA8, 128 tokens | 1.166 ms | 2.285 ms | 1.531 ms | 1.960× | 1.313× |
| GQA4, 256 tokens | 1.058 ms | 2.136 ms | 1.953 ms | 2.019× | 1.846× |
| GQA4, 1024 tokens | 3.906 ms | 7.912 ms | 6.051 ms | 2.026× | 1.549× |
| GQA8, 1024 tokens | 7.104 ms | 15.760 ms | 10.515 ms | 2.218× | 1.480× |

HPW1 and HPW2 were slower in every tested shape. Keep the HPW path default-off and do not use it for the current GQA4/8 workload.

V82 logs and full outputs remain on AC922 at:
`/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/qsa-direct-list-prefill-20261016/results/qsa-head-warps-gpu3-v82-20261016T075323Z/`.

## Safety and verification

- Both windows used the existing BMC watcher; no watcher was created or reconfigured. Pre/post samples showed target 3000 RPM, fan tach above 2840 RPM, no alarms/faults, masks intact, and maximum temperatures about 41–42°C.
- GPU3 was idle before both runs, below 40°C, and had over 16 GiB free. Postflight showed no compute process and only about 10–13 MiB of driver allocation.
- No model, server, or engine-wide load test was run. No CUDA source, kernel, runtime configuration, or BMC settings were changed by this fixture work.
- Local and AC922 parser tests: 8/8 pass. Host C++ `-Wall -Wextra -Werror -fsyntax-only` passed with and without `GETROWS_ENABLE_CUDA`; V81/V82 host links, shape plans, and CPU oracle plans passed.
