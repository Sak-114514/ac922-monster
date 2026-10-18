# V243: 512-byte MTP host-buffer UVA feedback fixture

Host-link r1 failed because the command omitted CUDA 12.4 headers; the original compiler log is preserved at `results/v243-host-link-r1.log`. R2 adds the CUDA include path and writes to a separate build directory. No GPU feedback run has started yet.

This tiny fixture tests the exact host feedback path for the next MTP head/body placement experiment: a 512-byte head buffer on GPU5 is filled with a different byte each round, copied asynchronously to the actual GGML CUDA host buffer allocated while CUDA device0 is current, then copied asynchronously to a 512-byte body buffer on GPU0 and read back for exact comparison. It repeats 100 rounds with nonblocking streams. There is no model load, full body copy, graph, or compute kernel.

The fixture requires the six fixed visible CUDA UUIDs, a 64-bit process, and `unifiedAddressing == 1` from each physical device. It reports the GGML host buffer name, `cudaPointerGetAttributes`, and `cudaHostGetFlags`; it does not assume a portable host-allocation flag. Successful GPU5 D2H → GPU0 H2D exact feedback is the test.

The build uses host `g++` against frozen V235 GGML/CUDA runtime libraries and CUDA 12.4 runtime APIs. It does not invoke NVCC. The guarded runner checks all six GPUs before and after for temperature below 58°C and at least 512 MiB free, and applies a 60-second timeout.

```sh
DEVICE_PROPERTIES_BUILD_AUTHORIZED=1 V235_FROZEN=1 \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v235 \
  ./build-v243-host.sh

MTP_HEAD_UVA_GPU_AUTHORIZED=1 V235_FROZEN=1 \
CUDA_VISIBLE_DEVICES=<six-fixed-UUIDs-in-order> \
  ./run-v243-guarded.py
```

The result establishes only the small UVA/page-locked-host feedback path, not performance or full-model correctness.

## Completed bounded run

`v243-uva-20261017T103647Z` passed all 100 exact round trips in 6.414 ms. The CUDA host buffer reported `cudaPointerGetAttributes.type=Host`, `cudaHostGetFlags=0x2`, and `portable_flag_set=0`; the GPU5→host→GPU0 copies still returned exact bytes without assuming the portable flag. All six devices reported UVA enabled and 64-bit pointers. Guard samples were 38–39°C with at least 16,108 MiB free, well inside the 58°C/512 MiB bounds. The run log has a nonfatal `ggml_backend_load_all` warning that `GGML_BACKEND_PATH` was a directory; the linked CUDA backend initialized and the test completed successfully.

Result artifacts are in `results/`; the V235 binary hash and resolved dependency list are recorded beside the run JSON/log.
