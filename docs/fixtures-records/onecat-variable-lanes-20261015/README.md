# Onecat variable-query/verify-shape fixture

This bounded GPU fixture tests the actual `runtime-perf-v34` CUDA backend, with no model loaded and no separately patched Onecat object. The executable links to the V34 `libggml-cuda.so.0`; the runner pins `LD_LIBRARY_PATH` to V34, verifies with `ldd`, sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb`, verifies that PCI index 2 maps to that UUID, and rejects a pre-existing or lingering GPU2 compute process.

The input uses QH=24/KVH=2, D=256, Q8_0 K/V, and a gathered every-other-row cache view. T=1 and T=3 cases use NKV=2303, one element short of the 9×256 boundary, exercising the final partial partition. T=6,9,10,12 use NKV=2304 so the stock Volta fallback has an available grouped-MMA specialization; the first odd-NKV probe exposed a stock `NO_DEVICE_CODE` hole at T≥6. Each query row gets a real causal triangular F16 mask; the final three keys are future keys for every query. T=3 with S=2 and S=3 models 6 and 9 total target-verify rows. A separate T=1/max_bias=8 case checks that the Onecat admission gate falls back for key bias.

The fixture directly calls the V34 Onecat admission function to assert expected support, and the Onecat-enabled T=1/no-bias execution must produce V34's `using opt-in Onecat Flash-V100 XQA decode path` marker. T>1 and nonzero-max-bias cases must report unsupported and stay on the stock route even when the environment gate is enabled. The max-bias case uses the production mask contract: visible keys receive `-abs(query_pos-key_pos)` and future keys remain `-inf`; the stock path must show a nonzero bias effect, and the Onecat-enabled bias case must match that stock fallback. Each no-bias case checks an independent CPU softmax oracle on sampled heads/query rows (max absolute tolerance 0.01), an allocator-adjacent 128 KiB canary, and exact output invariance after changing K/V values in the always-masked future tail. T>1 cases also alter a key that is future to query 0 but visible to query 1: query 0 must remain bit-identical while query 1 must change. T=6→T=9→T=6 reuses the original graph/tensor objects and requires bit-exact A replay.

Run on AC922:

```bash
bash fixtures/onecat-variable-lanes-20261015/build.sh
bash fixtures/onecat-variable-lanes-20261015/run-matrix.sh
```

After a complete matrix, only rerun the corrected key-bias pair with `bash fixtures/onecat-variable-lanes-20261015/run-matrix.sh bias-only`; this writes separate `bias-stock`/`bias-onecat` outputs and leaves the original full-matrix FAIL artifact intact.

The runner starts stock and Onecat-enabled cases in separate processes with identical deterministic input and compares every output at absolute tolerance 0.004 (`rtol=0`). It preserves completed stock T=1/T=3 outputs and runs only the remaining stock cases when those outputs exist. GPU2 temperature (abort at 58°C) and free memory (abort below 512 MiB) are polled while each process runs; the runner checks that GPU2 has no compute PID before, between, and after runs. The result does not establish full-model semantics or performance; it validates the Onecat dispatch/fallback contract for these local FA shapes only.
