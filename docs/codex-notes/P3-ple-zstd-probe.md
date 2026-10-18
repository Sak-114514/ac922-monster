# P3 PLE zstd sample and prefetch wiring assessment

Date: 2026-10-15. CPU-only, no source/header edits, no model rewrite/compression, no GPU work. A model load began after sample extraction; all compression/decompression after that used only the saved 16 MiB sample files.

## Machine and tool facts

- `/proc/cmdline`: `libata.force=noncq` is active.
- `/sys/block/sda/device/queue_depth`: `1`.
- `/usr/bin/zstd`: CLI v1.5.4 (`zstd -1 -T1` used).
- `libzstd.so.1` v1.5.4 was already present; Python `ctypes` used the installed runtime library for block-size accounting. No packages were installed. The zstd development header is absent.
- Model is 78,615,622,080 bytes. PLE metadata: 320,001,536 IQ4_NL rows × 90 bytes = 28,800,138,240 bytes (26.8222189 GiB), file offset 47,039,862,208.

## Sampling and exactness

Four equally spaced, row-aligned 4 MiB ranges were sampled from the actual PLE tensor (total unique sample coverage 16 MiB):

| Sample | File offset | Raw SHA256 | `zstd -1` compressed bytes | Ratio | Decompressed SHA exact |
|---|---:|---|---:|---:|---|
| 0 | 47,039,862,208 | `954e8ea3e08d962a42bf145cb5c24610f0c1451c275df6b5788d7bbb9087f697` | 4,110,104 | 0.979925 | yes |
| 1 | 56,638,510,168 | `1e928c1d2d96201eb06fbef729112d2d93b83e3a61e3f4ff349c9d26778f960a` | 4,109,898 | 0.979876 | yes |
| 2 | 66,237,158,128 | `e6ec050723a6bb613e950cdccf799c74701b61603d1df40807a5f14132e185a7` | 4,106,601 | 0.979090 | yes |
| 3 | 75,835,806,088 | `36ead8bac4a3e009ac00aaea01b8b944dbdec6a929e96d48ec040981c691000e` | 4,097,491 | 0.976918 | yes |

Across the four CLI frames: 16,777,216 raw bytes → 16,424,094 compressed bytes, weighted ratio **0.978952** (about 2.10% savings). Each decompressed sample SHA256 exactly matched its source.

The independent random-access accounting compressed each 4 KiB or 16 KiB block as its own zstd frame using installed `libzstd.so.1` v1.5.4 at level 1, then decompressed each block and byte-compared it to source. Every sample/block reconstructed exactly and every reconstructed sample SHA matched.

| Independent frame size | Raw sample bytes | Compressed bytes | Ratio | Result |
|---|---:|---:|---:|---|
| 4 KiB | 16,777,216 | 16,810,163 | 1.001964 | 0.20% expansion |
| 16 KiB | 16,777,216 | 16,494,805 | 0.983167 | 1.68% savings |

The four 16 KiB sample ratios ranged 0.980786–0.984524. Four 4 KiB sample ratios ranged 1.000531–1.002441. Compression used only the local sample files after extraction; no further model-file reads occurred.

Sampling caveat: an initial `dd bs=1` attempt was inefficient and interrupted. Its output showed 2,961,670 bytes written from sample 0 before termination, and it briefly began sample 1; the exact additional bytes read from sample 1 were not captured. Those reads were in the same four selected ranges, not additional tensor regions. Thus unique sample coverage is 16 MiB, but total underlying reads were slightly above 16 MiB due to the partial repeated read. The final samples were extracted with four 4 MiB `pread` calls. No page-cache eviction/advice, broad scan, or model write was used.

## Full-table RAM estimate for independently addressable chunks

The following estimate extrapolates the measured block ratios to the full table; it is a planning estimate from 16 MiB of sparse, equally spaced samples, not a full-table compression measurement. Random access requires per-block offsets; 8-byte offsets are assumed. The ledger includes a chosen 64 MiB decompressed-block LRU, the current 262,144-entry row cache (36 MiB entry storage), the existing 64 MiB max staging budget, and one decompression scratch block per 32 workers.

| Chunk size | Block count | Payload at measured ratio | 8-byte offset index | Decode LRU | Row cache | Max staging | 32 scratch | Estimated total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 KiB | 7,031,284 | 26.8749 GiB | 53.64 MiB | 64 MiB | 36 MiB | 64 MiB | 0.125 MiB | **27.0876 GiB** |
| 16 KiB | 1,757,821 | 26.3707 GiB | 13.41 MiB | 64 MiB | 36 MiB | 64 MiB | 0.5 MiB | **26.5445 GiB** |

A single 4 MiB frame per sample has the best measured ratio (0.978952) but is too coarse for efficient random-row access and still extrapolates to 26.26 GiB before indexes/caches. For the requested 20–22 GiB target, the full resident representation needs a ratio no worse than about 0.746–0.820 before metadata/cache overhead; measured independent-frame ratios are 0.983 (16 KiB) and 1.002 (4 KiB). **This does not justify a compressed resident PLE cache aimed at the 20–22 GiB target.** The 16 KiB case saves only about 0.45 GiB payload before indexes and decompression state; 4 KiB chunks expand the payload.

## Async prefetch status and safe wiring shape

Current source is under `standard-opt-20261014/src/src/llama-ple-prefetch-internal.h` and `llama-ple-prefetch.cpp`:

- It already has a bounded queue with one active job and one replaceable pending job. Replacing a pending job cancels the older pending job.
- `collect_rows()` hashes token/history and returns a sorted unique row set. The worker then iterates those sorted rows with `cache->warm_rows()`. This ordering is useful for a bounded chunk, but a whole-prompt job sorts rows across temporal chunks and can warm rows unrelated to the next GET_ROWS consumer first. Its current bounds are 8,192 tokens, 16,384 unique rows, 64 MiB direct-I/O budget.
- A source-wide search of `.cpp`/`.h` found only the prefetch API declarations/definitions; no llama context/server request path initializes the prefetcher or submits jobs. It is implemented but not connected to serving.

Recommended wiring (preparation only; not implemented):

1. Submit only the next actual scheduler ubatch, never the whole prompt. Preserve ubatch/request order and allow at most one future ubatch in the pending slot; do not stream all chunks into the replaceable queue.
2. Carry the exact `n_gram - 1` predecessor tokens for each sequence at that ubatch boundary, using the same KV-history/token-null/EOS rules as `llm_graph_input_ple::set_input`. A new request at position 0 has no external history; missing predecessor values become EOS. If exact history is unavailable or a sequence boundary is ambiguous, mark the job incomplete and skip it rather than guessing.
3. To overlap safely, release the worker for ubatch `k+1` only after ubatch `k` has completed its PLE GET_ROWS operation, so prefetch I/O does not race the currently demanded rows. At most one next chunk should be in flight; if it is not ready when its consumer starts, normal GET_ROWS remains the source of truth.
4. Current submit shape carries one token stream and one history, while server ubatches can contain multiple sequence lanes. For multi-stream serving, represent a pending physical ubatch as lane segments, each with its own ordered tokens and two-token history, then deduplicate/sort only the row union for that one ubatch. The current 16,384-row cap fits up to two 512-token/16-head lanes by raw row count; more lanes need per-lane jobs or a bounded larger aggregate.
5. The current warmer uses scalar `read()` per unique row. If the hook is integrated, feed its one-ubatch sorted unique rows through production `read_many()` to preserve page batching/coalescing; no row indices or output values should be reordered in the consuming GET_ROWS path.

## Artifacts

- Fixture directory: `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/P3-zstd-samples/`
- `sample-0..3.bin`, `.zst`, `.dec`: original samples, CLI-compressed frames, exact decompressions.
- `block-ratios.log`: independent 4 KiB/16 KiB frame ratios and exactness.
- `zstd-block-ratio.py`: sample-only block measurement script using system `libzstd` via Python stdlib `ctypes`.
- This report: `/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/P3-ple-zstd-probe.md`.
