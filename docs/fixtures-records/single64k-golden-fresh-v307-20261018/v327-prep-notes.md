# V327 CPU-only preparation

V327 derives from the V307 golden D2 runner and keeps its one-lane MTP2/UB512 64K workload, native Paris canary, and inherited fast-path environment. It changes only the six layer split weights and the explicit V326 PLE stripe quota map. The six CUDA-ordinal tensor-split weights are `9,9,8,8,8,8` (sum 50); the PLE GPU-prefix quotas are `3456,4160,3776,2880,2752,1408 MiB` (sum 18,432 MiB). The last CUDA ordinal retains the MTP/output placement share implied by the selected split.

Profile metadata and emitted result records store the six explicit stripe values, their total, and the tensor split. They do not infer equal per-GPU quotas from the legacy `LLAMA_PLE_NUMA_GPU_NODES` / `LLAMA_PLE_NUMA_GPU_GIB` total fields. V326's PLE admission code remains fail-closed; this CPU-only hostlink does not establish that the quotas fit at runtime. The root-owned capacity note remains pending.

Runtime identity pinned for this prepared runner: V326 build `20261017T190236.603887Z`, report SHA256 `c4319af21469d5a1b275b3a372f3e175b866562f7505be1ffd7b3105d3df39f2`. No model was loaded and no GPU was used during preparation.
