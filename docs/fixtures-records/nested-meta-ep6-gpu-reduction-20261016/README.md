# Nested Meta2×3 native GPU reduction staging

Status: fixture-only candidate; CPU/syntax and one bounded V59 six-GPU toy comparison completed. Core source was not edited, the engine/model was not built or loaded.

## Staged candidate

`proposal.patch` adds an opt-in bridge under `GGML_META_NESTED_EXPERIMENTAL=1` plus `GGML_META_NESTED_GPU_AR=1`. It supports only an outer Meta2 with two Meta3 children (six unique physical CUDA leaves), preserving depth-first rank order `A0,A1,A2,B0,B1,B2`.

At each explicit outer `PARTIAL` boundary, the staged code descends each island output tensor through `ggml_backend_meta_tensor_get_simple()` to six direct CUDA tensor descriptors and calls the existing six-rank P2P-AR callback in place. It does not call `ggml_backend_tensor_get`, `ggml_backend_tensor_copy_async`, CPU scratch, or per-step D2H. It holds the existing physical-device submit-order locks only around the collective, after both child Meta graph submissions have returned. It leaves the flat Meta communicator state untouched, because the merged-graph path indexes flat collective handles by the outer backend count of two.

P2P6 is preferred. Its current implementation supports up to eight ranks, requires pairwise peer access, and uses a six-rank 3+3 relay that accumulates FP32 ranks 0 through 5 in order. The optional NCCL6 fallback is dynamically available only if the CUDA registry exposes `ggml_backend_comm_is_nccl_f32` and that query confirms the initialized communicator is actual NCCL with forced F32. The staged patch compiles against older plugins because it resolves that proc by name and rejects NCCL fallback when the query is absent. NCCL results remain subject to the established strict tolerance; NCCL does not promise the same addition order as P2P6.

If the opt-in path reaches a partial boundary and neither GPU collective accepts it, the staged code returns `GGML_STATUS_FAILED`; it does not fall through to the unsupported local-partial Meta tensor-get path. Unsupported topology, asymmetric partial state, non-F32 or noncontiguous outputs, shape mismatch, missing leaves, and duplicate device handles also fail closed.

## CPU verification

Run on AC922:

```bash
AC922_MONSTER_SRC=/home/debian/ac922-monster/experiments/standard-opt-20261014/src \
AC922_MONSTER_BUILD=/home/debian/ac922-monster/experiments/standard-opt-20261014/build-reuse \
bash /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nested-meta-ep6-gpu-reduction-20261016/run-cpu-tests.sh
```

The runner compiles only a tiny CPU contract test, verifies patch application/reconstruction, and runs `-fsyntax-only` on the staged Meta TU against the private source headers. It does not build or link the engine and does not initialize CUDA.

The CPU tests prove the explicit 2×3 depth-first rank map, exercise a cancellation witness that yields 0 under flat rank-order sum and 3 under inner-island grouping, and check the disabled-compute-rank zero contract. A second consumer contract checks the V58 metadata transition and an `ADD(global_sum, mirrored_bias) -> RMS_NORM` result that differs if a second island reduction triples the source; it also asserts the A/zero-rank/A route plan and the two 8/16-row size builds. These are contract checks, not runtime correctness results.

## V59 bounded GPU result

Only the small Q8 EP fixture ran; no model was loaded. One six-GPU owner window ran both host-linked V59 binaries once, each with exactly three epochs: distributed route A, all-token-to-expert-0 zero-rank route B (the other five physical ranks contribute zero), then restored route A. Both flat Meta6 and nested Meta2×3 used cold rank-ordered P2P; the nested bridge logged `host_stage=0`.

- Rows/expert 8 (4,104 tokens): all three outputs passed flat/nested bitwise equality, 131,328 values each; foreign-zero and single-owner checks passed with eight route canaries.
- Rows/expert 16 (8,200 tokens): all three outputs passed flat/nested bitwise equality, 262,400 values each; foreign-zero and single-owner checks passed with eight route canaries.
- Each size logged exactly three six-rank reductions in order `A0,A1,A2,B0,B1,B2`, route `p2p6-rank-order`, fixed grid 64, and no host staging. CPU dequantized auxiliary max-absolute error was `7.15255737e-07` for both flat and nested outputs.
- The postflight compute-process list was empty. GPU memory was 1–3 MiB/card; temperatures were 37–38°C (saved samples show 38–39°C).

The readback is validation after the in-place device collective; the bridge itself uses no tensor-get, tensor-copy, or D2H staging. Logs are `results/v59-gpu-bounded/rows8-three-epochs.log`, `rows16-three-epochs.log`, `preflight-gpus.csv`, and `postflight-gpus.csv`.

## V58 split-state and consumer contract

No mirrored alias is needed for the nested allreduce result. V58's `ggml_backend_meta_get_split_state_impl()` transforms a tensor with `tensor_output_partial_flag` and raw `local_partial_axis` to `PARTIAL` when `assume_sync=false`, and to `MIRRORED` when `assume_sync=true`. Graph partitioning queries nodes with `assume_sync=false` (the producer remains the reduction boundary); source split-state queries use `assume_sync=true`. `ggml_backend_meta_buffer_init_tensor()` copies the producer flags into every simple tensor (`t_ij->flags = tensor->flags`). The nested bridge only updates leaf data in place, leaving the producer flags, tensor identity, and graph UID untouched. Therefore the consumer sees the completed six-rank data as mirrored. `RMS_NORM` uses `handle_per_row(src_ss)`, so an `ADD` of that mirrored value and a mirrored bias remains mirrored through RMSNorm; it will not create a second inner AllReduce. An alias or clearing the producer flag would break this existing convention and risk cached-graph/owner identity.

The CPU source-contract test checks those exact V58 source hooks and models the nonlinear consumer. Its mirrored bias makes an accidental 3× second sum observable; RMSNorm alone would hide a uniform scale error.

## Patch base provenance and current verification

`base/ggml-backend-meta.cpp` is the read-only V58 source snapshot, SHA-256 `932200a0ce9662cd851ca4fd9fe4b3c3dce766aca1b9f527979fe26546da153e`. `proposal.patch` is generated against this file and remains fixture-only; it has not been applied to `src/`.

The historical V57 logs `results/cpu-staged-r2.log` and `results/current-source-syntax.log` preserve the in-flight async-input syntax errors seen before V58. The V58 Meta CPP is the frozen patch base. The complete CPU/syntax runner passed against V58; see `results/v58-cpu-staged.log`. No engine build or GPU run is part of this result.

The fixture staging includes `tests/nested-meta-ep6-gpu-consumer-fixture.cpp`, a copy of the existing Q8 EP6 scaffold with a mirrored-bias ADD between the local-partial producer and RMSNorm. `run-cpu-tests.sh` syntax-checks this source but never links or runs it. This provides a real GGML graph shape for the next authorized bounded GPU test, beyond the CPU state-transition model.

## V59 bounded window preparation

`hostlink-v59-two-sizes.sh` links two distinct binaries against `runtime-perf-v59/lib` for 8 and 16 rows per expert (4,104 and 8,200 tokens). It records no engine launch and no GPU work. The last host-link passed against V59; SHA-256s are preserved in `results/v59-hostlink.log`.

`run-gpu-bounded-v59.sh` is the later, explicitly gated runner. It requires `NESTED_META_GPU_WINDOW_AUTHORIZED=1`, a six-entry unique ordered `CUDA_VISIBLE_DEVICES`, and an empty GPU compute-process list. It fixes `GGML_CUDA_P2P_AR_BLOCKS=64`, enables `GGML_CUDA_NCCL_F32=1`, sets `GGML_META_P2P_COLD=1` and disables the planned large-NCCL route so the flat6 oracle and nested bridge both prefer the same rank-ordered P2P6 (or both use strict-F32 NCCL fallback), opts into nested native reduction, and runs exactly three epochs per size: distributed A, all-token-to-rank-0 zero-rank B, then restored A (ABA). It checks exact flat-vs-nested outputs, foreign-zero/owner canaries, three native route logs, fixed-grid logging for P2P6, and `host_stage=0`. This gated runner has not been executed.
