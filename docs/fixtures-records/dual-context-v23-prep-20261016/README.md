# V22: diagnostic correctness followed by same-load clean timing

V22 is an isolated harness derived from V21 and links against the frozen V59 shared-library closure. It loads the integrated IQ3_XXS model once, then creates and fully drains/destroys separate Meta context families before changing constructor-sampled settings.

Each copy mode (`1` by default; explicit `1,0` optional control) runs two families in order:

1. **Diagnostic family:** preserves the full original 17-scenario sequence, native 32-case semantic gate, all 20 serial/P4 token pairs, both single-lane prefix checks, full F32 target-logit snapshots before/after MTP, and per-pass GPU spans. V21's known token failures remain visible in the final comparator.
2. **Clean family:** after the diagnostic family is synchronized, joined, and destroyed, creates new target/draft contexts over the same loaded model. It runs the same four real 8K prompts to 256 output tokens with full-logit copies/scans and GPU-span sampling disabled. Target GPU argmax, the real fallback if GPU argmax is invalid, and `embeddings_nextn`/MTP remain enabled. It has its own four exact comparisons against the serial references. Clean timing is reported even if the preserved full quality gate fails, but it cannot certify quality.

Before building the clean family, a preclean gate requires the diagnostic native semantic gate to pass, finite target logits, and exact serial/P4 full-F32 prefill logits. It deliberately does not suppress known token-parity failures. A semantic failure, non-finite logits, or any full-logit mismatch stops the clean family.

V22 keeps the V21 layout: two banks × two lanes, D3 MTP, Q8 KV, non-unified 131072 cells per bank = two independent 65536-cell streams. This is four 64K streams in aggregate, not a 256K per-request context. Target ubatch=1024, MTP ubatch=512, CPU async copy=on with a 1 MiB per-copy limit, legacy async-input scope off, `GGML_META_NCCL_MIN_BYTES=0`, `GGML_CUDA_NCCL_F32=1`, and standard QSA indexer. Both families keep replay guard enabled; diagnostic family uses spans period 1, clean family disables spans.

The launcher clears `LD_PRELOAD` and `AC922_SYNC_AUDIT` before the runner starts. It records that effective state in both manifests and the runtime log. This is required because a preloaded sync-audit library cannot be unloaded from a running model process.

The V22 harness only links against V59; it does not rebuild engine libraries. Host CPU tests and linking do not execute the model. The runner is not launched by the build script.

## Host-only preparation

```bash
./run-cpu-tests-v22.sh
ROOT_DYNAMIC_LIB_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v59/lib \
ROOT_BUILD_REPORT=/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V59-build.json \
./compile-runtime-v22.sh
```
