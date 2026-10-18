# Fixture preparation status

- Local Apple Clang 21 `-Wall -Wextra -Wpedantic -Werror -fsyntax-only`: PASS.
- AC922 POWER9 `/usr/bin/c++` host-link against `runtime-perf-v78`: PASS;
  fixture binary SHA-256 `2138d3629d621572ecbe413ce7549047f9c2df3f18467196b0067d4872189bdb`;
  see `ac922-hostlink-v78.log` for resolved shared libraries.
- Helper header SHA-256 on Mac and AC922: `060896c3680d5313a362e5dd98cec1b9ee2df1731d1ff1893b5c4f1e58467009`.
- The fixture is F32-only, contains both the stock batched score graph and the
  new per-head helper graph, uses the production four-cell expansion order,
  and compares the raw unscaled scores used by the current standard path.
- AC922 CPU matrix (`numactl --membind=0,8`, 60 s cap): PASS for all 8 shapes;
  `score_max_abs=0`, exact ordered cell IDs at both widths, duplicate top-k
  repeatability, and no non-finite scores. See `cpu-v78-20261016T071156Z.log`.
- AC922 CUDA matrix (`GPU3`, V78, 60 s cap, `GGML_CUDA_TOPK_STABLE=1`): PASS
  for all 8 shapes with the same exact score and cell-ID gates. See
  `cuda-gpu3-v78-20261016T071213Z.log`.
- An earlier CPU attempt falsely failed because the fixture read back a score
  intermediate after its buffer had become reusable. The fixture now computes
  score-only and selector-only graphs separately; the original failed log is
  retained as `cpu-v78-20261016T070823Z.log`.
- The score-only graph allocation fell from 2.35× to 3.45× across the tested
  shape matrix; at `blocks=2112, ntps=1024, streams=2`, it fell from
  144,769,024 bytes to 42,008,576 bytes (3.45×). The selector graph was equal
  at that largest shape (174,735,360 bytes per arm). These are fixture
  allocator peaks, not model-level prefill or throughput results.
- The six-GPU postflight showed no compute processes, 38–39 C GPUs, and over
  16 GiB free per card. The BMC watcher reported target 3000 RPM, fan RPM above
  2840, zero alarms/faults, masks intact, and maximum temperature 41 C.
- No model was loaded, no NVCC/runtime build occurred, and no core source was
  changed.
