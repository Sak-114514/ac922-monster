# V388 cached scheduler plan rebind candidate

This folder stages an opt-in `LLAMA_LAYER_PREFILL_PLAN_REBIND1` path against the live R393 source snapshot. It retains the existing six CUDA plus CPU caller gate and the `LLAMA_LAYER_PREFILL_CACHE_REBIND` restore contract. The scheduler API captures the live split/copy layout on first reuse, rotates four destination slots, retargets root edges and scheduler-owned dependency views, assigns fresh root and split UIDs, and then calls the existing gallocr allocate and reserve retry path.

The staged CPU fixture links the actual `ggml_backend_sched` and gallocr implementations from an isolated candidate build. Three mock CPU devices use distinct host buffer types so graph placement creates cross-backend copies. Each mock backend runs the real CPU graph executor; its graph optimizer adds scheduler allocation-dependency views that reference those copies. `build-cpu-fixture.sh` defaults to the pinned R393 header subset under `source-current/src`.

The fixture source and scripts are prepared only. No build, test, model load, GPU action, or process control was performed. The fixture includes A/B/A input replay with four-slot wrap, full-split output comparison, layout decline before mutation, and an allocator failure after the one root restore. CPU results cannot establish CUDA event safety or full-model performance.

`candidate.patch` applies to the three current source hashes listed in `parent-hashes.txt`. The older R381 files remain in `parent/` for history; `source-current/` contains the exact read-only source fetched from `ac922` and the pinned header subset hashed in `source-current/ggml-headers.sha256`.

`ISOLATED-COMPILE-PROCEDURE.md` records the R393 compile and link commands and a root-only procedure that swaps only `ggml-backend.cpp.o` in a temporary base-library link after verifying the untouched object closure against the frozen R393 library hash.

To run later, first build the candidate source into an isolated ggml library directory, then set `GGML_LIBRARY_DIR` and run:

```sh
./build-cpu-fixture.sh
./run-cpu-fixture.sh
```
