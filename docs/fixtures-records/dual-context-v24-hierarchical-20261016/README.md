# V24 hierarchical EP6 full-model launch preparation

**Prepared only; no model or GPU was launched for V24.** This reuses the already host-linked V23 owner-loop runner and its native semantic, full-F32 preclean, exact-token, and clean-family gates. It changes only the dynamic library closure to frozen V61 and applies the explicit hierarchical EP6 / six-node PLE environment.

## Runtime and runner

- Runtime: `runtime-perf-v61`, build ID `20261016T002602.434235Z`.
- V23 runner: `fixtures/dual-context-v23-prep-20261016/build-runtime-v23/dual-context-v23`, SHA-256 `df017b118af9669a8f3ea7c1e604f88791b85a3b1fe61c30a0f720e20de2d72d`.
- A read-only `ldd` check with `LD_LIBRARY_PATH=runtime-perf-v61/lib` resolves the runner's `libllama*`, `libmtmd`, and `libggml*` dependencies from V61. No runner or engine library was rebuilt for this preparation.
- The exact environment is in `v24-exact-env.json`; closure and hashes are in `runtime-closure-v61.txt`; the plan and safety assumptions are in `v24-plan-manifest.json`.

## Required layout and unchanged quality gates

The planned hierarchy is one model-level Meta2 with two Meta3 islands and six physical V100 leaves. All 512 experts remain present and IQ3_XXS / Q8 KV precision stays unchanged. `LLAMA_PLE_NUMA_GPU_NODES=6` and `LLAMA_PLE_NUMA_GPU_GIB=8` place the existing PLE prefix across the six GPU NUMA nodes, approximately 1.33 GiB per node. The existing four GiB per-card HBM reserve and three GiB CPU DDR reserve remain intact; the manifest makes no fit claim. The V61 loader's actual per-device accounting is authoritative.

The V23 workload remains two banks × two lanes, D3, non-unified per-sequence 64K KV, target ubatch 1024, MTP ubatch 512, standard QSA indexer, one shared model load, and the existing owner-local loop. V23's semantic gate, exact token comparison, full-F32 preclean gate, and diagnostic-to-clean context recreation are unchanged. The previous V23 clean sample reached 132.579 tok/s but failed exact token parity; it is not a quality-certified baseline, so V24 must retain and pass the original parity checks.

## Launch gating

`run-v24-supervisor.py --plan` is read-only. A future launch requires both `V24_TINY_GATE_PASSED=1` and `V24_GPU_WINDOW_AUTHORIZED=1`, then still runs the corrected V23 GPU/BMC watcher and stops on the existing GPU memory, GPU temperature, BMC, or fan-fault protections. It records NUMA node0+8 MemFree/FilePages/Dirty/Shmem/Unevictable and unit RSS without adding a DDR MemFree stop threshold. Fan target transitions to 3000/3500/4000/4300 are observed and recorded without interfering with BMC temperature control.

Do not set either launch gate until the V61 tiny nested-context/graph test passes and root grants the full-model GPU window. V61 requires `model.devices.size() == 1` for the outer Meta2 wrapper, so graph support remains unverified until that tiny gate.
