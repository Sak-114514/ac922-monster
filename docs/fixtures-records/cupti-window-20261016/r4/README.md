# CUPTI window collector r4

R4 preserves the R3 ABI and trace schema while increasing the fixed V1 callback-buffer pool and removing callback-path pool mutex contention. It is a new directory; R3 remains unchanged.

## Bounded memory budget

- User callback pool: 64 buffers × 4 MiB = **256 MiB host virtual allocation**. The buffers are allocated before CUDA context creation and reused without allocation inside request/completion callbacks.
- Event table: 1,048,576 × `sizeof(Event)` in static BSS (verify the exact size from the built symbol). Pages are demand-backed as selected-window records are written; the entire table is not pre-touched.
- CUPTI internal activity-buffer attributes remain the R3 values: host-pinned allocation, 3.2 MB per internal buffer, pool limit 4, preallocation 2. No internal buffer size or GPU HBM allocation was increased.
- Medium qualification graph storage is 1,000 kernel nodes per graph, three cards, one counter and one pinned host integer per card. GPU data payload is only a few bytes per card; it is not a model or load test.

## Callback changes

The request callback claims a free preallocated slot with atomic compare/exchange and a rotating scan; completion returns the slot with an atomic store. No callback takes the former global pool mutex. Both callback kinds remain counted in-flight, and stop still waits for in-flight callbacks and all issued slots to return before freeing the pool. Timeout retains the buffers until process exit. Kernel names are copied only after a timestamp passes the active-window filter; runtime API names are resolved in the stop/writer path, outside CUPTI callbacks. Context/stream tuple deduplication moved to the post-drain writer so each activity record does not contend on a global context mutex.

V1 dropped-record accounting still reads the global `(nullptr, 0)` counter at prepare, start, and stop, accumulating each reset-on-read value. A nonzero selected-window count, buffer starvation, event overflow, parser error, untimed event, or empty kernel trace invalidates the trace.

## Host build and CPU tests

```sh
CUDA=/usr/local/cuda-12.4
CUPTI="$CUDA/extras/CUPTI"
g++ -std=c++11 -O2 -Wall -Wextra -fPIC -fvisibility=hidden -shared -Wl,-z,defs \
  -I"$CUDA/include" -I"$CUPTI/include" collector.cpp \
  -L"$CUPTI/lib64" -Wl,-rpath,"$CUPTI/lib64" -lcupti -pthread \
  -o libac922-cupti-window.so
g++ -std=c++11 -O2 -Wall -Wextra test_buffer_drain.cpp -o /tmp/r4-drain && /tmp/r4-drain
g++ -std=c++11 -O2 -Wall -Wextra test_window_filter.cpp -o /tmp/r4-window && /tmp/r4-window
g++ -std=c++11 -O2 -Wall -Wextra test_drop_ledger.cpp -o /tmp/r4-drops && /tmp/r4-drops
python3 -m unittest -v test_trace_summary.py test_verify_medium_trace.py
/usr/local/cuda-12.4/bin/nvcc -std=c++11 -O2 medium_graph_probe.cu -ldl -o medium_graph_probe
```

No host-only test loads the collector or calls CUDA.

## Authorized medium GPU qualification

On an explicitly assigned free GPU0–2 window only, first record `free -b` and `nvidia-smi --query-gpu=index,uuid,memory.used,temperature.gpu --format=csv,noheader`. Restrict visibility to UUIDs 0,1,2. `medium_graph_probe` prepares CUPTI before CUDA initialization, captures and warms three 1,000-kernel graphs, then replays each 34 times in one window. It verifies each GPU's exact output is 1,000 and requests at least 102,000 kernel activity rows. The intended max window is 45 seconds; stop on timeout, do not reset/reboot.

Afterward run `verify_medium_trace.py` on the JSONL and re-record host/GPU memory. A qualification passes only when stop returns 0, exact outputs pass, trace integrity reports zero selected-window drops/starvation/overflow/parse errors, at least 100,000 graph-kernel records are present, and JSONL rows match metadata. Keep the trace and memory snapshots with the result. This qualification says nothing about model speed or clean-profile performance.
