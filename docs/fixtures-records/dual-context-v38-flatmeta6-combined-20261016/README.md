# V38 flat Meta6 combined candidate

V38 starts from the qualified V37 flat Meta6/EP6 short-context workload and frozen V89 runtime. It keeps the IQ3_XXS weights, Q8_0 KV, standard QSA indexer, one-bank/four-lane D3 geometry, 16K per-sequence capacity, and the existing semantic, full-logit, generated-token, clean-prefill, and V27 comparison gates.

The candidate adds the chunkless flat Meta graph (`GGML_META_CHUNK_NODES=0`), a 16,384-node CUDA structural-signature bound, the GPU-qualified packed dense path, CUDA graph force capture, and the GPU-qualified 32 MiB CPU-to-Meta copy ceiling. The process-wide pinned-copy pool remains 64 MiB. P2P AR uses the existing 64-block grid.

PLE placement requests a 12 GiB prefix spread across six GPU NUMA nodes. This is a capacity candidate, not a forced placement. The existing 4 GiB per-GPU HBM reserve, 3 GiB DDR reserve, page-placement checks, and loader guard stay intact. If the loader rejects the requested prefix, preserve its exact reason and stop; use an 8 GiB profile only as a separately tagged follow-up. Do not lower a reserve or force migration.

## Run contract

`build-v38-runner-v89.sh` links only the small host harness against the frozen V89 whole-library closure; it does not compile engine/CUDA code or execute a model. `run-v38-supervisor.py` refuses to start unless the generated profile, required micro-gate report, V89 report hash, and explicit bounded GPU authorization are valid. It performs the existing read-only BMC/GPU safety gate, runs one model load with no retry, and preserves the fan-monitor mask state while requiring an active watchdog.

The model run must show actual V38 dispatch in logs: merged graphs use one chunk where merge is eligible, structural statistics appear for all six devices, packed and forced-capture paths are selected, the 32 MiB policy is active with at least one real async copy, and the loader reports the 12 GiB PLE prefix. Missing activation is a failed candidate, even when the requested environment variables were present.

After the run, the launcher compares exact generated-token arrays against both the frozen V37 flat-Meta6 same-profile run and V27. V37 comparison is per case and flow; it is not replaced by semantic checks or the older V27 comparison.

All timing remains short-context evidence. `GGML_META_DEBUG=1` is process-static in this runtime, so clean timings remain debug-instrumented. A V27 token-array mismatch is reported as a quality limitation and is never waived by a faster timing.
