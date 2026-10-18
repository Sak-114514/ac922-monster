# P5 CUDA Meta A/B/A fixture

`fixtures/P5-meta-aba-cuda.cpp` is the three-rank CUDA counterpart to the CPU Meta fixture. It creates a mirrored Meta device from CUDA registry devices 0–2, adds CPU last as required by `ggml_backend_sched_new`, allocates a tiny prebound fake-weight input, then constructs A/B/A graphs. It checks the old A descriptor lookup after B, applies the production scheduler-owned binding reset, checks all graph-node descriptors on the three Meta members and view sources, and compares restored A with a fresh A graph. It uses only a dozen F32 input values and one small graph per shape; there are no loops or pressure workload.

Compile only; do not run while another GPU candidate is active:

```sh
/usr/bin/c++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-function \
  -I src/src -I src/include -I src/ggml/include -I src/ggml/src -I build-reuse/src \
  fixtures/P5-meta-aba-cuda.cpp -L build-reuse/bin \
  -Wl,-rpath,"$PWD/build-reuse/bin" -lggml-cuda -lggml-cpu -lggml-base -lggml \
  -o fixtures/P5-meta-aba-cuda
```

The fixture compiled successfully. Its linked `libggml-cuda.so.0` bytes match the active `runtime-clear-order/lib/libggml-cuda.so.0` (`sha256 323da630e4a062215bf6aa54aea3c92a07b42f66597db43aa6c1950f57097854`); both link against CUDA 12.4 runtime, the installed `libcuda.so.1`, and NCCL. The active server maps the same CUDA DSO hash. No CUDA device was initialized and the fixture was not run. The CPU-only Meta fixture remains the current behavioral proof; this fixture needs a short GPU execution after the main candidate releases the devices.
