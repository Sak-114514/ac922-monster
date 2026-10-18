# V227: CUDA device-properties API microbenchmark

**Preparation state:** harness, scripts, comparator, and static checks are ready. Nothing has been built or run. Wait for the root agent to freeze V226 before building, and wait for a separate CUDA run handoff before benchmarking.

This uses only the ordinary GGML backend API on six visible CUDA devices. It does not load a model, allocate weight buffers, build a graph, or launch compute kernels. It records each device's name, description, type, PCI ID, capabilities, total memory, and initial free memory. It then calls `ggml_backend_dev_type()` 1,000 times per device and `ggml_backend_dev_get_props()` 20 times per device, checking that type and immutable property fields stay consistent. The `memory_free` samples are retained as live readings and are not compared as static fields.

Run **two separate processes** so the V226 static environment choice is read once at process start:

```sh
DEVICE_PROPERTIES_GPU_AUTHORIZED=1 V226_FROZEN=1 \
RUNTIME_DIR=/path/to/the/frozen-V226-runtime \
CUDA_VISIBLE_DEVICES=<six-fixed-UUIDs> \
  ./run-v227-once.sh off

DEVICE_PROPERTIES_GPU_AUTHORIZED=1 V226_FROZEN=1 \
RUNTIME_DIR=/path/to/the/frozen-V226-runtime \
CUDA_VISIBLE_DEVICES=<same-six-fixed-UUIDs> \
  ./run-v227-once.sh on
```

`off` removes `GGML_CUDA_CACHE_DEVICE_PROPERTIES` from the child process environment; `on` sets it to `1`. The C++ program never changes the environment. Each run writes one JSON document to `results/` and backend diagnostics to a separate stderr log.

Compare serialized native properties and print the per-device timings with:

```sh
python3 compare-v227.py results/v227-device-properties-off.json \
  results/v227-device-properties-on.json
```

The comparator requires the static property fields to agree and keeps live free-memory samples outside that equality check. Timing shows only the backend property API cost, not model loading or serving speed.

## Host build

After root freezes V226:

```sh
DEVICE_PROPERTIES_BUILD_AUTHORIZED=1 V226_FROZEN=1 \
RUNTIME_DIR=/path/to/the/frozen-V226-runtime \
  ./build-v227-host.sh
```

The build uses host `g++` and links `libggml-base`; it does not invoke `nvcc` or compile CUDA kernels.
