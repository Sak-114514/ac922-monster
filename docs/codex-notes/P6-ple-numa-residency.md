# P6: PLE on GPU NUMA memory

Status: root ran the first build once after stopping the server. Both file-backed and anonymous phases reported `mbind(node=255,maxnode=256) = -1/EINVAL`; both reference comparisons passed exactly, while `numa_maps` showed the pages on `N0`, so this did not validate GPU-NUMA placement. No CUDA calls were made. Root's read-only topology check shows nodes `0,8,250-255` online, with memory, and in the effective cpuset. The fixture has since been updated to retry the historical `get_nodes()` maxnode off-by-one form (`maxnode=node+2`) only when the documented form (`node+1`) returns EINVAL. Upstream [linux-next `mm/mempolicy.c`](https://kernel.googlesource.com/pub/scm/linux/kernel/git/next/linux-next/+/refs/heads/master/mm/mempolicy.c) still shows `--maxnode` at the beginning of `get_nodes()`; the AC922 version of the kernel source is unavailable locally, so the workaround remains a probe rather than a diagnosed kernel defect. The updated fixture compiles with AC922 GCC 12.2 and `-Wall -Wextra -Werror` to ELF64 PowerPC64 at `/tmp/P6-ple-numa-residency` (BuildID `b12a7b91af8ba687dc67b8e69a1d972125b38fc3`, SHA-256 `ef1b6c1579e6e8e532735b8b226c24f9d1409fc8a201dc27e7c88d213f68945f`). It has not been run.

The fixture reads `data_offset`, `table_rows`, and `row_bytes` from the existing `hardware-next-20261014/ple-operator-evidence/model-params.json` manifest. The copied fixture manifest records the same values: offset `47039862208`, `320001536` rows, and `90` bytes per IQ4_NL row. It does not scan tensor contents to find the offset.

The C++ program has no CUDA runtime or driver calls. Each placement phase is capped at one 64 MiB virtual range. In file mode it maps exactly one page-aligned range from the actual model file, calls `mbind(MPOL_BIND|MPOL_F_STATIC_NODES)` before touching the mapping, reads every byte, and compares against an aligned `O_DIRECT` read using both FNV-64 and `memcmp`. It then reports both `maxnode` conventions (the documented `node+1`, and the historical-kernel probe `node+2` if the first call returns EINVAL), plus `mincore` residency and the matching `/proc/self/numa_maps` line. It supports disjoint 64 MiB PLE slices so the CPU-node control cannot warm the same file pages as GPU-node probes. If `--phase auto` finds zero pages on the requested node after file-backed placement, it unmaps that view and tests one 64 MiB anonymous mapping on the same node by copying the same exact PLE bytes. It does not evict page cache, call global `fadvise`, or call `mlock`.

The file-backed test may find pages already cached on CPU NUMA memory. `mbind` changes policy for future page allocation; it does not migrate those cached pages without an explicit migration request. A partial GPU-node result is reported as such. The anonymous phase proves that an anonymous mapped buffer can be placed on the GPU NUMA node and CPU read/write it; it does not prove persistence against reclaim or model-loader integration.

## Compile and run later

Compile on AC922 as a host-only binary; compilation itself does not allocate GPU memory:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/P6-ple-numa-residency.cpp \
  -o /tmp/P6-ple-numa-residency
```

After the active model benchmark is stopped, run the same fixture on distinct tensor slices: node 0 is the syscall control; nodes 250 and 255 test GPU-NUMA placement. The slices avoid reusing file pages touched by the previous node's test.

```sh
/tmp/P6-ple-numa-residency \
  --model /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf \
  --manifest /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/P6-ple-numa-residency-model-params.json \
  --node 0 --slice-index 1 --phase file --wait-seconds 5
```

```sh
/tmp/P6-ple-numa-residency \
  --model /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf \
  --manifest /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/P6-ple-numa-residency-model-params.json \
  --node 250 --slice-index 2 --phase file --wait-seconds 5
```

```sh
/tmp/P6-ple-numa-residency \
  --model /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf \
  --manifest /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/P6-ple-numa-residency-model-params.json \
  --node 255 --slice-index 3 --phase auto --wait-seconds 5
```

During each `snapshot window` line, inspect the matching GPU's `nvidia-smi` memory and `/sys/devices/system/node/node255/meminfo`. The process releases each 64 MiB mapping when its phase ends. File-backed page-cache pages may outlive `munmap`; the test intentionally does not force their eviction.

## Loader path assessment (read-only)

The lowest-change first attempt is to range-bind the existing lazy PLE file mapping before first-touching its rows, then pre-populate a bounded range. This retains the same tensor pointer and exact IQ4_NL bytes. The current row cache already checks mapped-page residency with `mincore` and then copies resident rows from `info.base`; its cold path opens the GGUF with `O_DIRECT` and reads into CPU staging, so cold rows bypass any mapped-page placement. PLE pages placed on GPU NUMA nodes therefore matter only when they are resident and the existing mapped-copy branch is selected. `mbind` does not pin file-cache pages, and the model's current `RLIMIT_MEMLOCK` is about 3.95 GiB, below an 8 GiB range.

A stable hybrid alternative is one contiguous anonymous mapping of the full PLE tensor (`28,800,138,240` bytes, about `26.82 GiB`): apply node-specific policies to an 8 GiB GPU-NUMA prefix (for example, 2.67 GiB each on nodes 255, 254, 253) and an 18.82 GiB CPU-NUMA remainder (nodes 0 and 8), then copy the raw GGUF bytes into that mapping without dequantizing or changing any model value. The tensor's virtual layout and row indices stay contiguous. `llama_model::impl` already owns mappings and backend buffers; an anonymous mapping owner placed before `ctxs_bufs` would outlive its non-owning `ggml_backend_cpu_buffer_from_ptr` wrapper because members are destroyed in reverse order. The row-cache source detector currently accepts only a file-backed mapping, so an anonymous mapping would fall through to generic CPU GET_ROWS unless the cache is taught to register this owned range with no file/direct-I/O source. That fallback remains numerically exact but its speed is unknown.

The hybrid option has not been implemented or tested. Current free VRAM is nearly exactly the full PLE size in aggregate, with substantially less free space on GPUs 3–5, so full-table GPU placement is not viable under the loaded server. An 8 GiB placement leaves more GPU headroom but consumes most of currently free CPU DDR for the remaining table; `mbind` does not pin anonymous pages either. A model-level attempt needs a fresh per-node capacity snapshot and must retain ordinary-file/O_DIRECT fallback if target-node allocation or placement fails.
