# CUPTI window collector r3

R3 is the profile-only, early-prepare variant of the collector. It fixes the late-attach limitation for warmed CUDA graphs: call `ac922_trace_prepare()` before `llama_backend_init` and before any CUDA context or graph is created. Do not inject or call prepare in non-profile processes.

## Exported C ABI

```cpp
using prepare_fn = int (*)();
using start_fn = int (*)(const char *output_path);
using stop_fn = int (*)();
using last_error_fn = const char *(*)();

auto prepare = reinterpret_cast<prepare_fn>(dlsym(RTLD_DEFAULT, "ac922_trace_prepare"));
auto start = reinterpret_cast<start_fn>(dlsym(RTLD_DEFAULT, "ac922_trace_start"));
auto stop = reinterpret_cast<stop_fn>(dlsym(RTLD_DEFAULT, "ac922_trace_stop"));
// last_error_fn is exported as ac922_trace_last_error().
if (!prepare || !start || !stop || prepare() != 0) { /* fail explicitly */ }
llama_backend_init(); // prepare must precede CUDA context/graph creation
warm_graphs();
if (start("/path/to/trace.jsonl") != 0) { /* fail explicitly */ }
run_one_probe_and_wait_for_its_natural_completion();
const int trace_status = stop();
```

`prepare()` configures CUPTI buffer attributes, registers the V1 asynchronous activity callbacks, and enables DEVICE, CONCURRENT_KERNEL, MEMCPY, and RUNTIME. It must be called before CUDA initialization/context creation so the memory allocation and device-buffer settings apply. It does not initialize CUDA, launch work, or synchronize a device. Start/stop each return 0 on success and nonzero on an invalid or incomplete window. Only one prepare/window per process is supported. `start()` accepts a path; a null/empty path uses `AC922_CUPTI_TRACE_PATH`. `ac922_trace_last_error()` returns the latest error string.

Activity collection stays enabled during model initialization and graph warmup. Completion callbacks parse but discard all non-device activity outside the selected window before reserving an event-table slot. DEVICE records are kept in a small separate table so the output can still map GPU IDs to names/UUIDs. Late-arriving warm-graph records are discarded using CUPTI activity timestamps. For boundary-crossing events the parser clips their intervals to the selected window. Stop records its timestamp, disables activity, forces a flush, and waits until callbacks are idle and every issued buffer has returned. On timeout it fails and retains the callback pool through process exit.

The collector only enables `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` (CUDA 12.4 `CUpti_ActivityKernel9`), never ordinary `CUPTI_ACTIVITY_KIND_KERNEL`, which CUPTI warns can serialize kernel activity. V1 MEMCPY enables P2P `CUPTI_ACTIVITY_KIND_MEMCPY2` records implicitly; the parser decodes them as `CUpti_ActivityMemcpyPtoP4`. Runtime events include API name, timestamps, thread, return value, and correlation ID.

## Drop counters and validity

The V1 global dropped-record counter resets on each read. R3 reads it after prepare, immediately before the selected start timestamp, and after stop's forced flush. It accumulates all returned counts in `dropped_read_total` and reports separate prepare, pre-window, and selected-window phases. Pre-window drops are reported but do not invalidate the selected request. Any selected-window global drop, overflow, parser error, callback-pool starvation, untimed selected event, missing query, or missing concurrent-kernel record makes the trace invalid.

The V1 completion callback supplies `(context=null, streamId=0)` global buffers, and `cuptiActivityGetNumDroppedRecords` requires an opaque `CUcontext` handle. Activity `contextId` values are not handles and are never cast as such. R3 records the actual observed `(device_id, context_id, stream_id)` tuples from selected events and queries the authoritative global `(nullptr, 0)` counter. It marks per-context/per-stream dropped-record counts as unavailable under this no-CUDA-API-hook design.

## Bounded memory

The fixed event cap is 1,048,576 records. `trace_meta` records the exact event size, BSS table capacity, name-field size, and name truncation counts; every kernel/device row also has `name_truncated`. CUPTI callback buffers remain fixed at eight 1 MiB buffers. The event array is static zero-initialized BSS, so only written pages are backed during a trace. CUPTI is configured before enabling activity for host-pinned data, 3.2 MB internal activity buffers, pool limit 4, and preallocation 2.

## Host-only build and CPU tests

```sh
CUDA=/usr/local/cuda-12.4
CUPTI="$CUDA/extras/CUPTI"
g++ -std=c++11 -O2 -Wall -Wextra -fPIC -fvisibility=hidden -shared -Wl,-z,defs \
  -I"$CUDA/include" -I"$CUPTI/include" collector.cpp \
  -L"$CUPTI/lib64" -Wl,-rpath,"$CUPTI/lib64" -lcupti -pthread \
  -o libac922-cupti-window.so
nm -D --defined-only libac922-cupti-window.so | grep ac922_trace_
g++ -std=c++11 -O2 -Wall -Wextra test_buffer_drain.cpp -o /tmp/test-buffer-drain && /tmp/test-buffer-drain
g++ -std=c++11 -O2 -Wall -Wextra test_window_filter.cpp -o /tmp/test-window-filter && /tmp/test-window-filter
g++ -std=c++11 -O2 -Wall -Wextra test_drop_ledger.cpp -o /tmp/test-drop-ledger && /tmp/test-drop-ledger
python3 -m unittest -v test_trace_summary.py
```

The collector must not be loaded or exercised as part of a host-only check. The output parser is `trace_summary.py`; it separates kernel duration sums from per-device unions and pairwise overlap, and it does not infer timings from tok/s.
