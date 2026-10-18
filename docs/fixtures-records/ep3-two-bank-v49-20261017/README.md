# V49 ordinary 3+3 EP3 two-bank candidate

V49 derives its runner from the frozen V32 two-bank owner-loop harness, then changes the profile to ordinary `LLAMA_ISLAND_SPLIT=3,3`, `LLAMA_HIERARCHICAL_EP6=0`, and all nested Meta switches off. One loaded model is shared by two target/MTP banks. Each bank has two independent 16K sequences, so the full request pool is four concurrent 16K-capable sequences (65,536 aggregate KV cells).

## Workload

The scenario order is one compact four-flow semantic canary, then one four-way run with distinct 8K real-corpus token arrays and 512 generated tokens per flow. It does not run a separate no-optimization baseline. Target UBatch is 1024; the draft remains 512; MTP is D3 with six fixed-prefix rows per lane and an eight-row target verify budget per bank.

The reused C++ owner loop submits one full `llama_decode()` per bank before joining the owner tasks. It tests grouped concurrent serving through two ordinary EP3 banks. It does not split a single decode graph into alternating MetaA/MetaB stage ranges; an observed throughput gain cannot by itself prove A/B stage interleaving.

## Runtime and safety

The selected profile pins one frozen runtime report and its `runtime/lib` directory. The V49 recipe uses ordinary `3+3` EP3, QSA standard indexer plus the qualified compact causal mask, fixed target UBatch, and `CUDA_DEVICE_MAX_CONNECTIONS=32` for the concurrent target/draft/copy stream set. It sets `LLAMA_PLE_NUMA_GPU_NODES=3` and `LLAMA_PLE_NUMA_GPU_GIB=12`, placing 4 GiB on each GPU NUMA node in island A (nodes 255, 254, and 253); island B receives no PLE stripe. The process remains CPU-bound to node 0 while `numactl --interleave=0,8` distributes ordinary runtime pages across the existing DDR nodes. Loader-owned PLE pages retain strict GPU-node and DDR-node binding. It deliberately leaves `CUDA_DEVICE_MAX_COPY_CONNECTIONS` and `CUDA_SCALE_LAUNCH_QUEUES` unset. `build-host.sh` only links the fixture runner to the selected runtime DSOs; it does not rebuild engine or CUDA objects and does not load a model.

The prepared launch contract requires a fresh hardware/BMC sample: all six GPUs idle with at least 512 MiB free and below 58 C; CPU maximum 2.6 GHz; GPU power caps 300 W; fan targets 3000 RPM with tach readings at least 2650 RPM; fan monitor remains in the installed inactive/masked policy while the watchdog is active. The wrapper is fail-closed, has a 30-minute bound and one model load, and does not change power or fan settings. It verifies all 859 PLE copy chunks with exact byte comparisons, the three-node 4 GiB stripe mapping and actual page placement, plus complete CPU DDR placement on nodes 0 and 8. It requires both a passed-gates token and a root GPU-window authorization before launching.

## Host-only preparation

Run the fixture-local generators and validator from the workspace root:

```sh
python3 monster-engine/standard-opt-20261014/fixtures/ep3-two-bank-v49-20261017/prepare_v49_runner.py
python3 monster-engine/standard-opt-20261014/fixtures/ep3-two-bank-v49-20261017/prepare_v49_scenarios.py
python3 monster-engine/standard-opt-20261014/fixtures/ep3-two-bank-v49-20261017/prepare_v49_profile.py
python3 monster-engine/standard-opt-20261014/fixtures/ep3-two-bank-v49-20261017/validate_v49_plan.py
```

These commands only read local source/manifests and write within this fixture. They do not SSH, compile, start a model, or touch a GPU.
