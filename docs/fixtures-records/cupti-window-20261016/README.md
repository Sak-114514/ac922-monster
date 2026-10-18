# Bounded CUPTI activity window collector

Revision 2, with a larger fixed record table and the callback-issued-buffer drain gate, is in [`r2/README.md`](r2/README.md). The files at this directory root are the preserved first host-only build.

Revision 3, which requires CUPTI preparation before CUDA/context/graph initialization and filters warm-graph records by CUPTI timestamps, is in [`r3/README.md`](r3/README.md).

This fixture is a small `LD_PRELOAD` library for collecting one explicit request window. It does not hook CUDA calls, initialize a CUDA context, launch work, synchronize a GPU, or collect CPU stacks. The runner must load it and call the exported markers around a request that has already naturally completed:

```cpp
using start_fn = int (*)(const char *output_path);
using stop_fn = int (*)();
auto start = reinterpret_cast<start_fn>(dlsym(RTLD_DEFAULT, "ac922_trace_start"));
auto stop = reinterpret_cast<stop_fn>(dlsym(RTLD_DEFAULT, "ac922_trace_stop"));
if (!start || !stop || start("/path/to/trace.jsonl") != 0) { /* fail explicitly */ }
run_one_probe();                  // no added CUDA synchronization
const int trace_status = stop();  // disable, forced flush, then write
```

Passing a null or empty path uses `AC922_CUPTI_TRACE_PATH`. One trace window per process is supported; repeated starts fail explicitly. `ac922_trace_stop()` returns nonzero if CUPTI reports dropped records, the fixed record/buffer pool overflows, callback parsing fails, or the output cannot be written. A failed/partial trace is still written when the callback drain completed and is marked `trace_valid: false` in the first JSONL record. The runner should retain both its return code and that metadata record.

## Collected records

- `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` as CUDA 12.4 `CUpti_ActivityKernel9`, including name, device/context/stream, timestamps, correlation ID, grid, and graph IDs.
- `CUPTI_ACTIVITY_KIND_MEMCPY` as `CUpti_ActivityMemcpy5`.
- `CUPTI_ACTIVITY_KIND_MEMCPY2` P2P records as `CUpti_ActivityMemcpyPtoP4`. With CUPTI V1 predefined records, enabling `MEMCPY` also enables the `MEMCPY2` P2P records; the callback parser handles both. The collector does not enable the ordinary `CUPTI_ACTIVITY_KIND_KERNEL` kind because CUPTI documents that kind as serializing kernel activity.
- `CUPTI_ACTIVITY_KIND_RUNTIME` as `CUpti_ActivityAPI`, with callback ID/name, host timestamps, thread ID, result, and correlation ID.
- `CUPTI_ACTIVITY_KIND_DEVICE` to map CUPTI device IDs to GPU names, UUIDs, and compute capability.

Timestamps use `cuptiGetTimestamp()` and the matching Activity clock. The stop marker is taken before disabling activities. The caller must stop after request work has naturally completed; this collector does not synchronize to force that condition. This avoids silently expanding the measurement window.

The collector uses a fixed pool of eight 1 MiB callback buffers and a fixed cap of 65,536 copied records. CUPTI is configured for host-pinned activity buffers, 3.2 MB per internal activity buffer, pool limit 4, and preallocation 2. These attributes are set before activity kinds are enabled. CUPTI documents some allocation attributes as applying to newly created contexts; if the application has already created its CUDA contexts, the collector cannot claim the limits changed those existing allocations. CUPTI 12.4's header says host-pinned allocation is supported and is the default since CUDA 11.2. Any failure to set the requested attributes is reported and aborts start.

## Build on AC922 (host-only)

```sh
CUDA=/usr/local/cuda-12.4
CUPTI="$CUDA/extras/CUPTI"
g++ -std=c++11 -O2 -Wall -Wextra -fPIC -shared \
  -I"$CUDA/include" -I"$CUPTI/include" collector.cpp \
  -L"$CUPTI/lib64" -Wl,-rpath,"$CUPTI/lib64" -lcupti -pthread \
  -o libac922-cupti-window.so
nm -D libac922-cupti-window.so | grep 'ac922_trace_'
```

This only compiles and links. Do not `dlopen` or run the library as part of the host-only check. Runner usage after a separate approved GPU window:

```sh
LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
LD_PRELOAD=/path/to/libac922-cupti-window.so ./runner
```

The CUPTI callbacks use the V1 async-buffer interface. They only copy normalized fields into bounded process memory and return the CUPTI buffer. The stop path disables the selected kinds, requests a forced flush from the control thread, waits up to five seconds for callbacks to drain, then queries the V1 global dropped-record counter. It does not call flush from a CUPTI callback.

## CPU-only parser

```sh
python3 -m unittest -v test_trace_summary.py
python3 trace_summary.py /path/to/trace.jsonl
```

The parser clips intervals to the marker window. Kernel duration sums are labeled separately from per-device interval unions; per-device unions and pairwise overlap are direct Activity timestamp calculations. Pairwise overlap values can overlap each other when three or more devices run together. Runtime API duration is a sum across host threads, not wall time. No throughput is inferred from token counts or tok/s.
