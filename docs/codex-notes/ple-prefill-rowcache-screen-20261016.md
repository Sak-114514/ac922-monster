# PLE prefill row-cache screen

Date: 2026-10-16  
Scope: read-only source and saved V35-log screen. No build, model run, or GPU work was done; V36 remains undisturbed.

## Finding

The current 8 GiB HBM + 18.82 GiB DDR PLE hybrid layout is **not admitted by the CPU PLE row-cache implementation**. The hybrid loader copies PLE into one anonymous mapping (`MAP_PRIVATE | MAP_ANONYMOUS`), then wraps that pointer in a CPU backend buffer. The row-cache admission path is file-backed: it scans `/proc/self/maps`, requires one mapping covering the full table, an absolute pathname, and a regular file with matching device/inode/size/mtime. It therefore rejects the hybrid PLE pointer before row lookups reach its cache or page-batch code.

This is corroborated in the saved V35 log: hybrid placement completed (`8.00 GiB` on GPU NUMA nodes, `18.82 GiB` on DDR), while PLE prefetch initialization emitted `row-cache identity or row geometry mismatch` (lines 2040, 2105, 168007, and 168062). The full V35 log contains no `ple-page-batch` lines, despite `GGML_CPU_PLE_PAGE_BATCH=1` and `GGML_CPU_PLE_PAGE_BATCH_LOG_PERIOD=1`. The mismatch identifies a failed cache/prefetch admission; the missing batch counters are consistent, but do not by themselves count or time CPU GET_ROWS calls.

When cache acquisition fails, `ggml_cpu_ple_cached_rows` returns false and the ordinary CPU quantized GET_ROWS path dequantizes rows from `table->data`. That fallback still partitions output rows across CPU tasks. The evidence does **not** show 32 threads serially processing all prefill rows. It does show a per-operation setup cost: thread 0 performs the failed cache acquisition before the helper’s barriers, so the other workers wait at that op boundary. There is no V35 timer isolating that setup, PLE row conversion, or CPU row-cache time.

## Source path

- V35’s loader reports the hybrid PLE placement in `fixtures/dual-context-v35-onebank4-combined-20261016/results/root-p4-v35-onebank4-v80-ep6-d3-16k-20261016t073508z/runtime.log:1932`: 8.00 GiB on the six GPU NUMA nodes and 18.82 GiB strictly on DDR nodes 0/8. Its table is 27,465.95 MiB. The saved PLE manifest gives 320,001,536 rows at 90 packed bytes per row.
- `src/src/llama-ple-numa.h:361-363` allocates the hybrid region with `mmap(..., MAP_PRIVATE | MAP_ANONYMOUS, ...)`; `src/src/llama-model.cpp:2291-2308` wraps this storage in a CPU buffer and attaches it to the PLE tensor. This is not the source GGUF file mapping.
- The CPU cache’s `mapped_source()` in `src/ggml/src/ggml-cpu/ple-row-cache-api.h:42-72` rejects a region that is not wholly contained in a single mapping, has no absolute file path, is not a regular file, or does not match the source file metadata. `acquire()` calls that check before its singleton mutex (`ple-row-cache-api.cpp:5-14`). The anonymous hybrid pointer fails these file-source requirements.
- The row-cache wrapper is gated by type/shape/contiguity and the two opt-in variables at `src/ggml/src/ggml-cpu/ple-row-cache.h:10-33`. Its thread 0 calls `acquire`, then the task team crosses barriers at lines 41-45, 88, 92, and 96. A null cache makes the wrapper return false so `ops.cpp:4994` takes the regular quantized GET_ROWS loop.
- The regular quantized loop obtains the type dequantizer and distributes rows by task (`src/ggml/src/ggml-cpu/ops.cpp:4980-5022`). Scheduler task count for the large PLE tensor is capped by `min(requested_threads, pool_threads, n_rows/32)` (`ggml-cpu.c:2295-2312`). With a 1,024-row operation and 32 requested/pool threads, that is 32 tasks, about 32 rows per task; with 131,072 rows in one operation, it is still at most 32 tasks, about 4,096 rows each.

## Lock, miss, allocation, and page-batch behavior if admitted

These facts describe the file-backed cache path; they are not evidence that V35 used it for hybrid PLE.

- `acquire()` has one process-wide mutex around checking/replacing its singleton cache pointer. The `/proc/self/maps` scan and `stat()` occur before that mutex. It is one lookup per eligible GET_ROWS op from task 0, not one global critical section per row.
- Rows are assigned to 64 hash shards with four-way sets. `find()` and `insert()` take an exclusive mutex only for the selected shard; a hit updates LRU age under that lock. Thus hot hits can contend within a shard, but the implementation does not globally serialize every row.
- The default capacity is 262,144 rows: 1,024 sets × 64 shards × 4 ways. The only accepted overrides are 262,144 or 524,288 entries. Against 320,001,536 PLE rows, those caps cover about 0.082% or 0.164% of the table. With the 144-byte `entry` layout, the corresponding cache vectors are about 36 MiB or 72 MiB. The default is selected because V35 does not set `GGML_CPU_PLE_ROW_CACHE_ENTRIES`.
- `GGML_CPU_PLE_PAGE_BATCH=1` switches each worker to `read_many` only when its local row count is at least 8 (`ple-row-cache.h:55-85`). At a 1,024-row GET_ROWS op split over 32 tasks, that condition would be met **if** cache admission succeeded. The batcher handles at most 64 indices at once, sorts misses and deduplicates repeated rows. Request arrays are stack allocated; the miss path creates one 16 KiB aligned scratch buffer per 64-row batch, not one heap allocation per row. It can merge adjacent aligned file spans into O_DIRECT reads up to 16 KiB (`ple-row-cache-api.h:325-489`).
- A cache miss calls `mincore` to choose resident mapped copy versus aligned O_DIRECT `pread`; failed/unsupported direct I/O falls back to mapped copy. Those source paths do not apply after the current hybrid admission failure. The normal GET_ROWS fallback still dequantizes each requested IQ4_NL row to FP32; the cache would not remove that conversion cost.

## V35 timing evidence and limits

The V35 log has aggregate `prefill-lane-timing` rows (target `llama_decode` sync and MTP time) but no PLE-specific CPU timing, row counts, cache hit/miss totals, or page-batch stats. Its `GGML_CPU_PLE_*` thread settings are present in the exact environment manifest, but that does not prove the row-cache fast path ran. In particular, V35’s PLE prefetcher uses the same `ggml_ple_io::acquire(table)` API (`src/src/llama-ple-prefetch.cpp:248-261`), and the saved runtime explicitly reports its identity mismatch; `LLAMA_PLE_PREFETCH=1` therefore did not make the bounded prefetch worker active for this hybrid table.

The PLE page-batch/cache path could have been a CPU cost in a file-backed layout, but the concrete V35 hybrid evidence instead points to the ordinary parallel dequantization path plus a failed per-op cache admission. No data here establishes how much of prefill wall time that consumes. A CPU fixture result, if reused, must retain this distinction:

- Existing `fixtures/P1-ple-page-batch-fixture.cpp` exercises the file-backed row-cache API with a synthetic 256 KiB file, exact byte checks, sparse/dense reads, and 16 concurrent readers. It does not cover the anonymous hybrid buffer or full-model GET_ROWS timing.
- Existing `fixtures/P6-ple-numa-residency.cpp` and its manifest use a bounded 64 MiB slice of the original PLE GGUF to inspect NUMA placement/residency. They do not exercise GET_ROWS or cache admission. No existing CPU-only fixture was found that combines original PLE bytes, the hybrid anonymous pointer, and production GET_ROWS/cache dispatch.

## Bounded next evidence, after V36

Do not change row-cache admission or claim a speed win based on this screen. If an isolated CPU investigation is wanted after V36 releases, the shortest useful gate is to take the already bounded original-byte P6 fixture model and extend a CPU-only fixture to check the production `mapped_source`/`acquire` result for (a) the original file mapping and (b) an anonymous hybrid-style mapping containing the same packed bytes, then run one exact production GET_ROWS graph with 0 and 32 CPU workers while recording the existing cache admission, `read_many`, and wall-time counters. Keep the data slice bounded (64 MiB), compare output bytes exactly, and do not run that during V36. This would validate the admission/fallback conclusion and quantify the barrier/dequant path without touching the model, CUDA, or system settings.
