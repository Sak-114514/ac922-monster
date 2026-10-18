# V135 loader streaming-page-reclaim fixture

This fixture targets the new stream-reclaim path through the real
`llama_model_loader::load_all_data()` implementation. It creates a tiny valid
Qwen4Exp-tagged GGUF at runtime; it never opens a model GGUF. The GPU tensor is
allocated in a real mirrored Meta3 buffer, so the loader's GPU-buffer
classifier and Meta3 synchronous set path are exercised. A following CPU tensor
shares its final 64-KiB reclaim granule with the GPU tensor.

The binary has two modes. `off` leaves the stream flag unset and checks that a
fully owned GPU source page is still resident at the callback immediately
before the future CPU context is loaded. `on` sets
`LLAMA_LOAD_RECLAIM_GPU_PAGES_STREAM=1` and checks that the same page has been
evicted before that CPU load, while the GPU-side bytes in the shared boundary
granule remain resident and are not retired. Both modes compare all three Meta3
copies and the CPU tensor byte-for-byte, then check that the GGUF file itself is
byte-identical. After final cleanup the fixture remaps the file and verifies
that the CPU-shared boundary page remains resident.

The generated GGUF alone is `fsync`'d after creation, before the byte snapshot
and loader run. This keeps the tiny fixture's page-cache test about clean
file-backed pages; no real model file is opened for writing or synced.

The source also runs host-only planner contracts for 4-KiB and 64-KiB ranges,
including inward-aligned edges, an excluded future CPU alias, and invalid or
overflowing ranges. `--planner-only` exits before loading backend registries or
initializing CUDA.

## Build after the V135 runtime is frozen

The fixture includes the private loader C++ header, so source headers and the
runtime DSO must come from the same V135 build. The loader methods were visible
in `nm -D -C` on the older V126 runtime; the build script checks those dynamic
exports again against the requested runtime.

```sh
./build-v135-fixture.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/src \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v135
```

Build the small fixture TU only after the root marks V135 source/runtime frozen.
The script does not run the binary or initialize CUDA.

## Run after an explicit root GPU window

The runner requires `ROOT_LOADER_STREAM_RECLAIM_GPU_WINDOW_AUTHORIZED=1`, uses
only the visible first three CUDA devices for a tiny Meta3 copy, and bounds each
mode to 60 seconds. Run it only after the root has frozen the runtime and
assigned the GPU window.

```sh
ROOT_LOADER_STREAM_RECLAIM_GPU_WINDOW_AUTHORIZED=1 \
  ./run-v135-gpu-fixture.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v135 \
  /tmp/loader-stream-reclaim-v135-hostbuild/loader-stream-reclaim-v135
```

`LLAMA_LOAD_PREFETCH=0` is fixed for this test so the mapping does not receive a
separate WILLNEED prefetch before the copy checkpoint. The fixture sets the
existing final reclaim flag in both modes; therefore the `off` assertion is
made before aggregate final cleanup and demonstrates the old final-only timing.

## Scope boundary

This fixture covers the mmap-backed, synchronous `ggml_backend_tensor_set`
path, one Meta3 GPU tensor, one future CPU tensor, and the planner's page-range
contract. It does not exercise non-mmap uploads, event-ring async uploads,
`check_tensors`, lazy reads, hybrid PLE, or Monpack sidecar sources. Those
routes remain untested here and must continue using the existing final-only
fallback until separately tested; this fixture does not claim they are safe or
streamed. It also does not test a full model, output quality, load-time
headroom, or a performance gain.
