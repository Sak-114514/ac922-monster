# V41 target ubatch 4096 candidate

V41 is a thin V40/V94 overlay that raises only the target `n_ubatch` from 1024 to 4096. It uses chunkless wholegraph mode (`GGML_META_CHUNK_NODES=0`) with the V94 cold prepare cohort enabled and forced graph capture disabled (`GGML_CUDA_GRAPHS_FORCE=0`). It retains one bank, four lanes, D3 MTP, 16K context per lane, `n_batch=8192`, the original integrated IQ3_XXS + Q8 MTP model, standard QSA indexer, V40's diagnostic/clean semantic and token gates, and all existing fan/temperature/GPU-free guards. Draft `n_ubatch` remains 512.

The launcher is generated from the frozen V40 launcher with exact source-hash and replacement-count assertions. It changes the V40/V94 labels and paths, target-only 1024→4096 values, chunking 512→0, and ordered Relay6 maximum 10→40 MiB. The supervisor imports and runtime-patches the frozen V40 supervisor in memory; it does not carry a second 500-line copy. The result comparator similarly adapts only the target-ubatch expected value while retaining every other V40 validation.

## Capacity and transport gate

V40 has one target context and one MTP draft context live per context family. Diagnostic and clean families run sequentially; scope exit drains and destroys each pair before the next family. V40 reserved target compute buffers of 238.44 MiB Meta + 20.54 MiB CUDA Host and draft buffers of 249.28 MiB Meta + 41.28 MiB CUDA Host. Minimum observed GPU free was about 1,989 MiB. A purely linear target-buffer estimate for 4× ubatch adds about 715 MiB, but that estimate excludes graph growth, capture allocations, fragmentation, and route-dependent temporary tensors. The 512 MiB per-GPU stop floor stays in force.

At V40's 1024 target ubatch, the largest selected AllReduce was exactly 10,485,760 bytes (10 MiB). At 4096, the corresponding payload is 41,943,040 bytes (40 MiB). The 10 MiB Relay6 gate cannot be retained without falling back on the larger messages. V41 therefore stays launch-blocked until the bounded 40 MiB relay gate passes for the V94 runtime.

## Launch state

`v41-exact-env.json` is generated from the frozen V94 build report. `v41-micro-gates.json` keeps V40 evidence and adds the V94 six-rank cold-cohort and 40 MiB Relay6 gates. The cold-cohort gate must show exact masks for votes, graph votes, preparation, launch start, and launch return on all six devices with no abort. `launch_eligible` must remain false until those gates are artifact-verified and root explicitly grants a bounded GPU window. The runner performs no retry. The launch monitor keeps the existing 58°C ceilings, 512 MiB GPU free floor, fan target/tach checks, and no-GPU-reset policy.

Build the host harness only with `build-v41-runner-v94.sh`. It links the V41 harness to the immutable V94 library closure and never builds engine/CUDA objects or launches the model.
