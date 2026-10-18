# Prefill event-query fixture

This small V35-only fixture validates the private nonblocking event-query API without loading a model or changing core code. It checks simple CUDA and a three-member mirrored Meta backend; GPU0 runs the simple case and GPUs3/4/5 run Meta3. Each per-device workload allocates eight 4096×4096 F16×F16 matmul outputs (about 576 MiB of tensors before backend workspace), and the fixture asserts the measured allocation delta stays below 1 GiB per device.

For each backend, two event objects are used in an A/B/A sequence. The fixture enqueues an asynchronous graph, records the event, and only then publishes that generation to its query helper. It never queries a newly allocated or re-recording event before the corresponding record call. For each published generation it requires a first query of `0` (pending), then `1` (ready), with a bounded 2-second timeout and 250-µs sleeps between queries. Meta query readiness is checked through the V35 aggregate API, which must wait for all three child events.

Invalid-path checks require `-1` for a null event and for a synthetic event on an unsupported CPU device. Query support must be false for null/CPU devices and true for CUDA and Meta3. V35's `GGML_META_FIFO_PIPELINE=1` gate is set before event creation.

The runner pins `LD_LIBRARY_PATH` to `runtime-perf-v35/lib`, records hashes for the V35 manifest and `libggml-base.so.0` / `libggml-cuda.so.0`, restricts visibility to GPU0 and GPUs3/4/5, verifies library resolution, and continuously enforces <58°C and >512 MiB free on each selected GPU. It refuses pre-existing or lingering compute PIDs. Results are written to `results/event-query-results.json` plus a plain-text fixture log.

Run on AC922:

```bash
bash fixtures/prefill-event-query-20261016/build.sh
bash fixtures/prefill-event-query-20261016/run.sh
```

To validate an existing JSON/log set without launching any CUDA workload, use `bash fixtures/prefill-event-query-20261016/run.sh validate-only`.
