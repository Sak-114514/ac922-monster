# V255: per-bank MTP ownership + PLE headroom candidate

This fixture is a one-model-load candidate prepared locally from the frozen V244 R2 runner. It tests two families in order: a one-bank 64K prefill at target UBatch 512 / D4, then four independent 16K banks at UBatch 256 / D4. The frozen V244 scenario files and semantic checker are retained byte-for-byte; only the runner and runtime profile are V255-specific.

Candidate settings:

- Per-bank MTP context owner map: CUDA device indices `0,1,2,0`.
- PLE prefix: 14 GiB, with `LLAMA_PLE_NUMA_CPU_BALANCE=1` and `LLAMA_PLE_NUMA_CPU_HEADROOM=1`.
- Normal heap: `numactl --cpunodebind=0 --interleave=0,8`; PLE uses its explicit quota/bind policy.
- Single 64K family CPU budget: 16 decode and 16 prefill threads. Four-bank family: 2 decode and 8 prefill threads per bank.
- D4/PAD6 and draft UBatch 256 remain fixed. Global graph force stays off; the phase-prefill capture hint and grouped-Q8 MoE opt-in are off. Small-J token ceiling remains 512.

The profile is bound to frozen runtime V254 (`20261017T114517.979417Z`, report SHA-256 `7894188621c68f30eb0a3d7c1f43e644fe0bdb8b2bbece5560d89ae4fe951917`). The local runner refuses launch while `safety.launch_eligible` is false; root still controls when the model gate is released. No model, build, remote files, or GPU were touched during local static preparation.

Hostlink command, after this fixture has been copied to the AC922 at the same relative path:

```bash
bash /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/mtp-clones-headroom-v255-20261017/root-host-link-v255.sh
```

The hostlink is CPU-only and checks the frozen V254 report, source digest, DSO resolution, and output-directory nonexistence. Model launch requires the guarded `run-v255-root.py` and a root release; this fixture does not authorize an unguarded launch.

Local verification:

```bash
python3 verify-v255-static.py
python3 test_v255_semantic_gate.py
```

The semantic tests use prior records with synthetic D4 metadata to test the checker plumbing only. They do not establish V255 runtime correctness or performance. V255 remains an unbuilt, unlaunched candidate until hostlink and guarded native canaries pass.
