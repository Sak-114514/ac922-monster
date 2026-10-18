# V324 direct CUPTI preparation

## Reason for switching away from Nsight import

Root reports that the V320 R2 native canary and 64K long-prefill run completed, but both the Nsight Systems 2023.4.4 importer and a manual import attempt failed with `CUDA API Wrong event order`. No repeat capture or importer experiment is planned.

NVIDIA's official Nsight Systems release notes say versions starting with 2024.2 no longer support PowerPC: <https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html>. The official ppc64le redistribution index lists 2023.4.4.54 as the last available ppc64le build: <https://developer.download.nvidia.com/compute/cuda/redist/nsight_systems/linux-ppc64le/>. NVIDIA HPC SDK 24.3 notes say the Nsight Systems 2024.1 `Wrong event order` report-generation failure is fixed in 2024.2: <https://docs.nvidia.com/hpc-sdk/archive/24.3/hpc-sdk-release-notes/index.html>.

## Bounded adaptation

V324 keeps the V319-qualified V320 R2 model/runtime, scenario order, UB512, D2, context geometry, and all inherited runtime flags. It replaces only the NSYS marker path with the already-built V223 direct-CUPTI Activity collector. The collector is reused by exact remote SHA; it is not rebuilt or modified.

The only CUPTI capture is target prefill `seq=0`, `chunk_index=1`, with exactly 32,768 rows from the 64K request. Start is called in the persistent bank owner immediately before that target `llama_decode`; stop is called after the runner's existing `llama_synchronize(target)` and before MTP prefill. There is no added CUDA synchronization and no Nsight importer. The runner logs the owner's Linux TID from `syscall(SYS_gettid)` at CUPTI start/stop and requires both markers to run on that same TID so offline analysis can isolate CUPTI runtime APIs for the actual submit owner.

Collector bounds pinned from its frozen V223 source/build:

- 4,194,304 event records (1.1875 GiB at the source's 304-byte `Event` size)
- 64 callback buffers × 4 MiB = 256 MiB
- zero overflow, dropped selected-window records, callback starvation, parse/bad-buffer errors, untimed records, and device/context-table overflow are required for a valid trace
- preserve at least 4 GiB host DDR free before starting the one-model run; root's existing in-band guard owns runtime resource/thermal checks

This is a diagnostic trace, not a profiled throughput benchmark. It does not claim that CUPTI activity alone attributes CPU PLE or all host scheduling time.
