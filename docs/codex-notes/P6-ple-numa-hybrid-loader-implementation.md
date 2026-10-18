# P6 hybrid PLE loader prototype

## Scope

Private, opt-in loader path for Qwen4Exp's single lazy `per_layer_token_embd.weight` tensor. The environment variable `LLAMA_PLE_NUMA_HYBRID=1` is required; default behavior is unchanged. On the expected AC922 UUID/PCI order, the raw IQ4_NL bytes are copied without conversion into one contiguous anonymous mapping: an 8 GiB prefix is striped over GPU NUMA nodes 255/254/253 and the remainder is interleaved across CPU DDR nodes 0/8. CPU GET_ROWS continues to use the unchanged tensor type and generic CPU path.

The helper reserves 32 MiB staging and streams from the PLE's validated file offset. It uses aligned O_DIRECT reads, a small buffered tail for a non-aligned final range, and a `memcmp` check for each chunk. Only after a chunk compares exactly does it `MADV_DONTNEED` and `POSIX_FADV_DONTNEED` fully covered system pages inside that PLE interval; shared first/last pages are retained. No full-file/global page-cache eviction is issued. CPU capacity accounts for free plus clean reclaimable `FilePages`, subtracting `Shmem`, `Dirty`, and `Unevictable`, and keeps a 3 GiB reserve. It also checks per-node CPU capacity, CUDA free memory, and GPU-NUMA node availability.

At completion, `/proc/self/numa_maps` must show every page exactly on the requested three GPU NUMA nodes or CPU nodes 0/8. UUID/PCI mismatch, insufficient capacity, syscall/copy/placement failure, or row validation failure discards the anonymous allocation and restores the existing lazy GGUF path. The feature is disabled when mlock mode or no-alloc mode is selected.

## Ownership and files

- `src/src/llama-ple-numa.h`: policy, bounded raw copy, source-page cleanup, and actual placement checks.
- `src/src/llama-model.cpp`: defer only the one-tensor lazy PLE context, perform capacity-gated placement, attach non-owning CPU buffer and keep mapping owner alive longer than buffer/context wrappers; restore original buffer route on failure.
- `src/src/llama-model-loader.h` / `.cpp`: borrowed PLE pointer/buffer handoff and attach the already copied exact bytes without invoking the file-backed row path.
- Before-edit snapshots are in `/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/P6-before-hybrid-loader/`.

The model SHA was already validated by the existing experiment artifacts; this loader does not hash the whole GGUF again. Exact per-chunk `memcmp` is the copy gate.

## Verification performed

- Compile-only helper fixture: passed with AC922 `/usr/bin/c++ -std=gnu++17 -Wall -Wextra -Wpedantic -fsyntax-only`; it performs no runtime allocation or CUDA call.
- `llama-model-loader.cpp` and `llama-model.cpp`: passed `-fsyntax-only` using the prepared `build-reuse/compile_commands.json` flags with private source/include paths inserted.
- Existing 64 MiB P6 placement fixture was preserved; its prior root-run result is in `notes/P6-root-fixture.log` and `notes/P6-root-probes.json`.

No object/library/server build, CUDA initialization, model load, inference, or performance test was run for this patch. Runtime enablement, exact real-model page placement, quality, and speed remain unverified. The PLE suffix uses generic CPU GET_ROWS; no row-cache acceleration is claimed.
