# V35 prefill event-query fixture — 2026-10-16

## Result

The bounded fixture passed against frozen `runtime-perf-v35` for simple CUDA on GPU0 and Meta3 fanout on GPUs3/4/5. No model was loaded. The runner's in-process polling used 250-µs sleeps and a 2-second timeout; every event first returned pending (`0`) after its record was enqueued, then ready (`1`). A2 reused event A at generation 2 after A1/B1, and the fixture never queried an unrecorded event.

| Check | First query | Final query | Pending polls | Elapsed | Per-device tensor delta |
|---|---:|---:|---:|---:|---:|
| CUDA0 A1 | 0 | 1 | 40 | 12.580 ms | 576 MiB |
| CUDA0 B1 | 0 | 1 | 41 | 12.952 ms | 576 MiB |
| CUDA0 A2 (event re-record) | 0 | 1 | 40 | 12.882 ms | 576 MiB |
| Meta3 A1 | 0 | 1 | 39 | 12.742 ms | 576 MiB × 3 |
| Meta3 B1 | 0 | 1 | 41 | 13.476 ms | 576 MiB × 3 |
| Meta3 A2 (event re-record) | 0 | 1 | 43 | 14.049 ms | 576 MiB × 3 |

Null event query returned `-1`. CPU and null devices reported query unsupported, and a synthetic unsupported CPU-device event returned `-1`. Simple CUDA and Meta3 reported supported; Meta3 had three CUDA children. All 13 JSON checks passed. GPU guards observed 38–40°C and over 15 GiB free on selected devices; the post-run check showed no compute PIDs.

## V35 provenance

Runtime manifest hash: `1b604c428f8fabd7edb99d60c3b912aed7a61002fdf1dde0fb779138d8ada893`.

- `libggml-base.so.0`: `3b02064f687d28793d26a60992933688af590f8484ccbc57135279731df960d0`
- `libggml-cuda.so.0`: `614cc8685ba9d8c2f5a998a34af330609d7a372d7e764d07052c4ef3d808e7d1`
- Private API header: `01bf5fd2a7c44f08bdac8445756e5a886eca5b2ca069af6063fb2cae7c43846f`
- Fixture source: `632f3d3d11b424cb3e66999c8fc70da5b4c75665147275f5427987eb2deb6ecb`
- Executable: `2cac977daf4e6ea768dd4f31419a5fdbc18a7605f5ba78a05dda4b6374c0245d`

`ldd` resolved ggml libraries from V35. The library exported `ggml_backend_prefill_event_query` and `ggml_backend_prefill_event_query_supported`.

## Artifacts and scope

- Fixture: `fixtures/prefill-event-query-20261016/event-query-fixture.cpp`
- Build/run: `fixtures/prefill-event-query-20261016/build.sh`, `run.sh`
- Structured per-check output: `fixtures/prefill-event-query-20261016/results/event-query-results.json`
- Runtime hashes, `ldd`, event symbols, build log, event log, and GPU safety samples are alongside the JSON.

The fixture allocates eight 4096×4096 F16 matmul outputs, with measured delta 576 MiB on each active device. It changes no core/system/BMC/driver/firmware files, changes no power/fan settings, loads no model, and measures no inference performance. The runner has a 58°C cutoff and a 512 MiB free-memory floor.

The fixture itself printed PASS. The first wrapper invocation stopped after the GPU work because its JSON validator used stale `cuda_simple/...` names; I corrected those names and added `validate-only`. The corrected validator passed the existing JSON without rerunning GPU work, and the selected devices were checked afterward for temperature/free memory and PIDs.
