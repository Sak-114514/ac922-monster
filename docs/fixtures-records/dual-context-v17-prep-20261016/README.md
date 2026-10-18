# V17 before/after-MTP prefill-logit diagnostic

V17 is an isolated copy of V16. V16 remains frozen. V17 changes only the prefill fixture: when `ROOT_PREFILL_LOGITS_BEFORE_MTP=1`, it dumps the target's full F32 logits immediately after the existing natural `llama_synchronize(target)` and immediately before `common_speculative_process()`. No additional synchronization or CUDA operation is added. The existing after-MTP dump remains enabled whenever `ROOT_PREFILL_LOGITS_DIR` is enabled.

Before-MTP dump scenario labels append `.before-mtp`; the flow ID, prompt token IDs/positions, and prompt hash stay the same as the ordinary after-MTP dump. `compare_prefill_logits_v16.py` can compare the two files directly. Every file includes the V16 magic/version header, prompt identity, final logit row, F32 payload length, and CRC-32. The V17 CPU fixture proves a before/after pair with matching flow, prompt, position, and valid payloads parses through that comparator.

The flag defaults off. If enabled while `ROOT_PREFILL_LOGITS_DIR` is off, the runner fails before model/CUDA initialization. Logit dumping adds host I/O and marks all result records as diagnostic-only; existing semantic and exact-token gates are unchanged. A before/after logit difference shows that the observed target logits differ across the MTP processing boundary, but does not by itself assign a cause to a kernel, quantization, or context-sharing behavior.

V17 retains V16's optional deferred CUPTI stop failure and SIGSEGV/SIGBUS diagnostic handler. Neither is enabled by default. For the planned clean profile set `ROOT_CUPTI_PHASE=off`; `LD_PRELOAD` sync audit is independent and remains optional outside this runner. Any enabled logit dump or preload audit makes rates diagnostic-only.

## CPU tests and frozen V54 host build

Run the CPU-only regression suite, including V16 fixtures and the new before/after-MTP pair proof:

```bash
./run-cpu-tests-v17.sh
```

Build against the frozen V54 runtime closure without executing a model:

```bash
ROOT_DYNAMIC_LIB_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v54/lib \
ROOT_BUILD_REPORT=/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V54-build.json \
  ./compile-runtime-v17.sh
```

Example enabling both sides of the prefill comparison:

```bash
ROOT_RUNTIME_REVISION=20261015T214046.596738Z \
ROOT_DYNAMIC_LIB_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v54/lib \
ROOT_BUILD_REPORT=/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V54-build.json \
ROOT_CUPTI_PHASE=off ROOT_PREFILL_LOGITS_DIR=1 ROOT_PREFILL_LOGITS_BEFORE_MTP=1 \
ROOT_TURBOPREFILL=0 ROOT_UBATCH=1024 ROOT_MTP_DEPTH=3 ROOT_WARMUP_FRAMES=4 \
  ./run-v17.sh ./build-runtime-v17/dual-context-v17 /path/to/IQ3_XXS-MTP-Q8.gguf /path/to/results/p4-v17
```

The output folder receives pairs named along the lines of `serial-real8k-0_before-mtp__real8k-0.ac922pflogits` and `serial-real8k-0__real8k-0.ac922pflogits`. Compare them with:

```bash
python3 compare_prefill_logits_v16.py before.ac922pflogits after.ac922pflogits
```

This fixture is host-prepared only. CPU tests and linking do not establish model correctness or performance; root owns any model/GPU run.
