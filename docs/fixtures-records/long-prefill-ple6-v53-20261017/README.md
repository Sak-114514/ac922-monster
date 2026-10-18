# V53: PLE GPU prefix 6 GiB total

V53 is a host-only candidate for the next long-prefill attempt. It keeps the V50/V52 model, all experts, quantization, CPU PLE lookup math, V108 native QSA mask and block-bias paths, and the 32K/64K scenario geometry. Its single placement change is `LLAMA_PLE_NUMA_GPU_GIB=6` across three GPU NUMA nodes: 2 GiB per node on 255/254/253.

The planning comparison favored the A-like placement: Meta weights 7.44 GiB/GPU plus a 4 GiB/GPU PLE prefix, versus B's 9.34 GiB/GPU weights and no GPU PLE. V53 retains PLE on those first three GPU nodes while returning 2 GiB/GPU relative to A; the figures do not establish the 64K workspace fit.

The V52 64K case failed a 4,305.92 MiB allocation on device 0. The placement estimate moves CPU PLE residency from 14.82 GiB to 20.82 GiB. Against approximately 31.57 GiB physical DDR on nodes 0 and 8, simple subtraction leaves 10.75 GiB before upload-pool use and the rest of the process. The upload pool can allocate up to 512 MiB. These estimates do not prove fit; actual per-node DDR, swap, file-cache, pressure, page-fault, and runner RSS/PSS observations are required. They are record-only; no extra DDR threshold is introduced.

The runner source is derived from the pinned V50 source SHA-256 `766a555bffb20631d7ce4d3020ca41d452ba05ec888c757ddf610793b2a7b965`. Its only behavioral change is the harness assertion for the new 6 GiB total / 2 GiB per-node PLE setting; it also emits those values in the startup and result records. The model, GGUF, token paths, and lookup arithmetic remain unchanged.

V53 remains launch-blocked until the single C++ runner is built against the frozen V108 libraries, its seven-library closure is verified, fresh BMC/fan/GPU/DDR observations are recorded, and root authorizes one bounded run. No engine/CUDA rebuild, remote sync, model load, GPU use, or memory-management changes are included in this host-preparation step.

## Runtime memory observations

The supervisor's read-only sample records NUMA node 0 and node 8 `MemFree`/`MemAvailable` and file-cache pages, `/proc/meminfo` `Cached` and `SwapFree`, `/proc/vmstat` page/swap counters, memory PSI, and model-runner `smaps_rollup` when the runner exists. It stores before/after deltas. The sample never calls `drop_caches`, changes swap/sysctls, or alters BMC, fan, power, driver, or kernel settings.
