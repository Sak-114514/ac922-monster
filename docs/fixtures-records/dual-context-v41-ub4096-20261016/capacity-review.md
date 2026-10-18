# V41 capacity and transport review

## Live context accounting from V40

The harness constructs one `bank` per context family. Each bank owns one target context and one MTP draft context concurrently. The C++ family scope drains and destroys both contexts before moving from diagnostic to clean, so diagnostic and clean reservations are sequential rather than additive.

V40 runtime log observations:

| Context | `n_ubatch` | Meta compute reservation | CUDA Host compute reservation |
|---|---:|---:|---:|
| Target | 1024 | 238.44 MiB | 20.54 MiB |
| Draft | 512 | 249.28 MiB | 41.28 MiB |

The target reservation recurs across diagnostic/clean context families. That is not evidence that both target reservations coexist. Minimum observed per-GPU free memory was approximately 1,989 MiB after model/context setup. A linear target-arena estimate for 4096 is 953.76 MiB, or +715.32 MiB over V40; that would leave roughly 1,274 MiB before considering other UB4096 graph/capture growth. This is a planning estimate only. It does not establish that the full workload fits.

V40 logs `lazy-prefill-workspace ... max_ubatch=1024 ... KV_backing=65536`; V41 must show `max_ubatch=4096` while retaining the same KV backing. A full-model run must stay above the existing 512 MiB free floor on every GPU, below 58°C, and pass all original semantic, F32, token, and clean-prefill checks.

V41 selects `GGML_META_CHUNK_NODES=0` so each eligible Meta graph remains whole. This is admitted only with `GGML_META_COLD_PREPARE_COHORT=1` on V94, and `GGML_CUDA_GRAPHS_FORCE=0` keeps normal graph warmup. The V94 local F32 gate must demonstrate complete six-rank preparation and launch masks; the UB4096 full-model run must then prove graph activation and memory headroom under the live guard.

## AllReduce size

V40 selected 10,485,760-byte AllReduce dispatches (10 MiB). The shape is 2,560 F32 elements per row × 1,024 rows × 4 bytes. At target ubatch 4096 the corresponding payload is 41,943,040 bytes (40 MiB). Existing ordered Relay6 admission is capped at 10 MiB, so it would not serve this message. V41 sets the maximum to 40 MiB only after the bounded V94 40 MiB gate passes.

The larger compact Relay6 reservation is estimated to increase 120→480 MiB per main communicator (+360 MiB) and add 80 MiB for the segmented buffer. These are additional planning costs; actual allocation must be checked in the bounded relay gate and full-model free-memory telemetry.

## Decision

UB4096 is the highest-potential single target-ubatch probe because it reduces the 8K prefill from eight 1024-token target tiles to two 4096-token target tiles and raises expert-row coalescing. It is not launchable with the current V40 10 MiB relay bound. Required before a model load: V94 cold-cohort fixture PASS, 40 MiB ordered Relay6 fixture PASS, and V41 host harness linking against the exact V94 closure. The full run remains one model load, no retry, no stress test, and uses all existing safety/quality gates.
