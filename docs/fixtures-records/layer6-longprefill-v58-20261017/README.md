# V131 plain-CUDA layer6 long-prefill preparation

This fixture prepares one derived runner/config from the pinned V50 single-model `single32k`/`single64k` harness. The V135 closure is frozen (`20261016T213259.513817Z`, report SHA256 `75a0a3a651a0f8a1b1fbd115a391e19289de10ffcca0959497c4e52b65c2b519`). The single-TU host link completed; the runner has not been launched, no model weights were read, and no GPU was touched.

## Candidate

- `LLAMA_SPLIT_MODE_LAYER` over six visible plain CUDA devices, expert parallelism off, no Meta island split, and no AllReduce selection.
- V122 byte-balanced split weights `[10, 10, 9, 7, 8, 6]`; the loader’s cumulative `upper_bound(layer_index / 50)` rule yields blocks `0–9`, `10–19`, `20–28`, `29–35`, `36–43`, and `44–48` plus output on device 5.
- One lane (`n_parallel=1`), original Q8 KV, unchanged original model weights, QSA standard sparse/head-loop, narrow GET_ROWS, grouped/fused-T16 MoE, Volta GDN, SM70 BF16, graph-cache, and packed decode selectors retained where eligible, plus the dedicated single-lane MTP GPU chain. Multi-lane/unrolled MTP remains off.
- `n_batch=32768`, target `n_ubatch=1024`, draft `n_ubatch=512`; the 32K prompt uses one logical batch and the 64K prompt uses two ordered 32K batches. The native F16 compact mask remains opt-in from the pinned V50 profile; its 65,536×1,024 size estimate is 128 MiB, subject to runtime dispatch/shape confirmation.
- PLE requests a 12 GiB prefix across six GPU NUMA nodes (2 GiB each), with the remaining table on DDR NUMA nodes 0 and 8.
- The pending V135 load profile sets `LLAMA_LOAD_PREFETCH=0`, streams page reclamation with `LLAMA_LOAD_RECLAIM_GPU_PAGES=1` and `LLAMA_LOAD_RECLAIM_GPU_PAGES_STREAM=1`, and releases MTP prefill history with `LLAMA_MTP_RELEASE_PREFILL_HISTORY=1`.

The V122 ledger gives a largest assigned layer-weight share of 8.069 GiB on GPU0, leaving 3.931 GiB after the existing 4 GiB reserve and before context/workspace allocations. Historical workspace estimates are 238 MiB at target UB1024 and 249 MiB at draft UB512. The previous V50 64K case OOMed on device0 during a 4,561.92 MiB allocation with target UB4096. These values motivate UB1024; they do not prove this profile fits.

## PLE placement gate

Root has updated the current `src/src/llama-ple-numa.h` page-placement loop to use `plan.gpu_count`, matching the three- or six-node planner and bind loops. The source emits `ple-numa: gpu_stripes_verified=%zu per_stripe_exact=1`; the six-card gate requires the exact `gpu_stripes_verified=6 per_stripe_exact=1` line after all six per-device actual placement checks. Six-card placement remains runtime-unverified. A later run must also retain all 859 exact-copy chunks and the whole-table `memcmp_exact=1` result before claiming the PLE stripes passed. The fixture does not edit core.

## Prepared artifacts

- `prepare_v131_layer_runner.py` verifies V50 source SHA256 `766a555bffb20631d7ce4d3020ca41d452ba05ec888c757ddf610793b2a7b965`, validates the V55 immutable reuse record and V122 byte ledger, and writes the derived C++ runner, unified patch, profile JSON, and copied scenario arrays.
- `build-v131-host.sh` uses one runner translation unit and the frozen V135 DSOs, binds the compiler to CPU NUMA node 0 and memory nodes 0,8, and records the binary, DSO, `ldd`, report, source, and compile-log hashes. The linker produced the binary and the actual seven-DSO `ldd` closure passed. The first wrapper exit was 1 because it also demanded `libllama-server-impl.so`, which is not a dependency of this runner; that optional-DSO check is corrected in the script. The earlier source attempt that assigned nonexistent `common_params::offload_kqv` is preserved at `results/host-build-v135/initial-invalid-common-params.log`. The corrected build did not build engine/CUDA objects or execute the runner.
- `dual-context-v131-layer-longprefill.cpp` is the one derived runner source. No supervisor was copied.
- `prepared-output/copy0/single32k/scenarios.json` and `prepared-output/copy0/single64k/scenarios.json` contain the V50 canary-plus-long-case sequences with V131 labels.
- `test_v131_host_prep.py` checks the loader-boundary calculation, source/profile settings, scenario geometry, pipeline-parallel source condition/copy constant, and the six-node PLE verification gap. It is CPU-only and is not model or runtime evidence.
- `v131-layer-profile.json` records the frozen V135 identity, linked binary and closure hashes, exact candidate, and expected runtime gates. Full build evidence is archived in `results/host-build-v135/`.

## Runtime evidence required later

Before reporting fit or overlap, a future authorized run needs the frozen V135 runtime closure and actual logs showing all six CUDA model devices, layer owners for indices 0–49, no CPU expert-weight fallback (CPU global buffers limited to the base embedding and PLE sources), automatic pipeline-parallel enablement with no reserve retry, Q8 KV, the original standard kernel selections, all 859 PLE copy chunks with exact `memcmp`, `gpu_stripes_verified=6 per_stripe_exact=1`, the semantic canaries, and the 32K/64K outcomes. The scheduler source creates four graph copies when automatic pipeline parallelism is enabled; only actual runtime enablement and complete workload execution can establish that the candidate used that path. This host preparation does not establish overlap, memory fit, semantic parity, or throughput.
