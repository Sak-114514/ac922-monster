# AC922 CUDA stream-sync call-site audit

This fixture builds a small Linux `LD_PRELOAD` interposer for `cudaStreamSynchronize`. It is intended to locate host call sites behind the synchronization time observed in a CUPTI trace. It does not modify the engine or V16, does not link libcudart, does not initialize CUDA, and does not load the model.

The interposer is transparent by default: unless `AC922_SYNC_AUDIT=1`, it creates no file and directly forwards each call exactly once. When enabled, set:

- `AC922_SYNC_AUDIT=1`
- `AC922_SYNC_AUDIT_PATH=/absolute/path/sync-audit.jsonl`
- optionally `AC922_SYNC_AUDIT_EVERY=128` (default 128; sample every Nth call)
- optionally `AC922_SYNC_AUDIT_AGGREGATE_EVERY=1024` (default 1024)

For sampled calls, the shim resolves the real function first, then captures a caller PC and up to 16 raw stack PCs with image base/offset metadata **before** the measured interval. It records `CLOCK_MONOTONIC` begin/end timestamps, Linux TID, stream pointer, return code, caller image/base/offset, and per-frame image/base/offset. It calls the real function once between those timestamps, then formats and appends the record after the call. Thus `dlsym`, `backtrace`, `dladdr`, JSON formatting, and file I/O are excluded from the reported synchronization interval.

Non-sampled calls only perform the call counter and total-time/error atomics plus two monotonic clock reads. An aggregate JSONL row is written every configured aggregate interval. Writes use one append-only file descriptor and no locking mutex; concurrent rows can arrive out of call-number order, so use the recorded call number and timestamps when sorting.

Image offsets allow offline symbolization, for example `addr2line -Cfipe /path/from/image /offset/from/record`. A missing `RTLD_NEXT` target returns CUDA's `cudaErrorUnknown` numeric value 999; the normal production path should resolve libcudart after it is loaded by the process.

## CPU-only qualification

`./run-cpu-tests.sh` uses a fake libcudart DSO. It verifies disabled mode creates no file, the wrapper forwards the stream and error code with exactly one real invocation, sampled records contain bounded call-site/backtrace/timestamp fields, and concurrent calls complete without deadlock or dropped sample/aggregate records. It uses `g++ -shared -fPIC -ldl`; no NVCC, CUPTI, CUDA initialization, GPU, or model is involved.

## Production usage shape

After host compilation, a diagnostic process can be launched with:

```bash
LD_PRELOAD=/path/to/libcuda-sync-audit.so \
AC922_SYNC_AUDIT=1 \
AC922_SYNC_AUDIT_PATH=/path/to/sync-audit.jsonl \
AC922_SYNC_AUDIT_EVERY=128 \
AC922_SYNC_AUDIT_AGGREGATE_EVERY=1024 \
  /path/to/runner ...
```

The trace changes host overhead and is diagnostic-only. This fixture does not start a model or claim a performance result. Use it only in a separately authorized diagnostic run; the clean V17 run can remain CUPTI/logit-shim off if call-site attribution is not needed in that run.
