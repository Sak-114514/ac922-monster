# Contiguous expert-axis CUDA consumer fixture

This fixture is an isolated single-V100 CUDA consumer test. It reads packed rows directly from the integrated GGUF, builds a full 512-expert tensor plus exact contiguous byte slices, and compares `MUL_MAT_ID` outputs between the full-bank reference and each local EP slice. It does not load `llama-model`, start a server, or modify weights.

## Cases

- Seven target formats: `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, and `Q2_0`.
- Three `blk.48.nextn` expert tensors from the integrated MTP package: gate, up, and down, all `Q8_0`.
- V100 MMVQ token counts: `1, 8, 12, 16, 32`. The 12/16 cases use `GGML_SM70_DECODE_LINEAR_CHUNKS=8` and require the `ep_mmvq_chunks` dispatch marker.
- Ordinary MMQ: `T=512`, grouped MMQ disabled, requires `ep_mmq` and an actual `mul_mat_q` CUDA kernel.
- Grouped MMQ: `T=512,1024`, grouped path enabled; require `mmq_moe_grouped` selected and an actual `mul_mat_q_moe_grouped` kernel. The Q8_0 expert case must log the known `unsupported_quant_type` grouped decline, then execute ordinary MMQ.
- Three per-case owner slices, boundary IDs `[0,169,170,340,341,511]`, an all-foreign route pattern, and A/B/A replay with exact output checks.

The fixture requires separate target and nextn owner counts at run time. Supply them from each model's own axis-2 split metadata; the fixture does not assume the MTP tensors share the target split.

Optional filters are comma-separated. The first bounded smoke should use one target format plus one Q8_0 nextn tensor, for example:

```sh
EP_RUN_MODES=mmvq \
EP_TEST_TYPES=IQ3_XXS,Q8_0 \
EP_TEST_TENSORS=blk.35.ffn_gate_exps.weight,blk.48.ffn_gate_exps.weight \
bash fixtures/ep-consumer-validation-20261015/run-gpu-bounded.sh GPU-UUID TARGET_COUNTS NEXTN_COUNTS
```

The all-format run omits those filters and runs MMVQ, ordinary MMQ, and grouped MMQ in separate processes.

`tensors.json` records the target packed-bank offsets from the bounded target row manifest and the Q8_0 `blk.48` offsets from a read-only `GGUFReader` tensor-directory audit. Its `tensor_file_offset` is an absolute file offset; each expert stride equals `N * row_bytes`.

## Build only

Compile this fixture without rebuilding the engine. Its CUDA backend is loaded
from an explicit per-run plugin directory (`EP_BACKEND_DIR`), so the selected
runtime is chosen at execution time rather than through a system-wide plugin
search path:

```sh
bash fixtures/ep-consumer-validation-20261015/build-fixture.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v23
```

The script links the selected runtime's existing ggml libraries and does not
link `libggml-cuda.so` directly. The run scripts create CUDA/CPU plugin symlinks
from that exact runtime, set `EP_BACKEND_DIR` and `LD_LIBRARY_PATH`, and unset
`GGML_BACKEND_PATH`. Each run records the resolved plugin paths and hashes;
confirm these before interpreting a result. The CPU `--help` / owner-map path is
separate from the GPU route consumer and must not be counted as consumer
validation.

## GPU run gate

For future runs, wait for the root to release a V100. The runner requires a
single GPU UUID, independent target/nextn split counts, no existing compute
process on the selected GPU, and temperature below 58 C. It runs one process at
a time, records Nsight CUDA kernel summaries, and fails if expected path markers
or kernel names are absent.

## Fused consumer evidence (2026-10-15)

- V23 old-wrapper reproduction: one `IQ3_XXS` fused T=1 case deterministically
  reproduced CUDA illegal memory access. The report and runtime provenance are
  under `/home/debian/ac922-monster/experiments/standard-opt-20261014/results/ep-fused-v23-once-20261015T125017Z/`.
- V24 fixed-wrapper consumer: `IQ3_XXS` target and `Q8_0` nextn gate tensors,
  T=1/4/8, all three owner slices, exact owned-output comparison, foreign
  `+0.0`, and A/B/A all passed. The final GLU was the only graph output. Nsight
  confirmed fused `mul_mat_vec_q_moe` kernels for ggml types 18 and 8 with the
  GLU flag `(bool)1`; the runtime plugin resolved from `runtime-perf-v24`.
  Artifacts are under
  `/home/debian/ac922-monster/experiments/standard-opt-20261014/results/ep-consumer-runtime-perf-v24-20261015T125124Z/`.
- This is a CUDA consumer fixture, not full-model correctness or performance
  evidence. The root's full-model V24 validation remains separate.

## IQ2_S EP three-arm probe (verified 2026-10-15)

The probe selected only the real target tensor
`blk.12.ffn_gate_exps.weight` (`IQ2_S`) and ran unfused MMVQ and fused
MUL_MAT_ID + MUL_MAT_ID + GLU at T=1/4/8 across all three owner slices. It
required actual dispatch markers and native kernel names in Nsight. The full-bank
reference for the native arm has EP range `[0,512)`, so it uses the same native
Volta arithmetic before each owner slice is compared. The standard path is not
used as a bit-exact oracle for native output.

Three arms completed:

1. V24 with `GGML_CUDA_IQ2S_VOLTA_EP=0` and legacy IQ2_S Volta flag off.
2. V26 with both flags off. Its full/local output digests matched V24 exactly
   for all 18 mode/token/rank combinations.
3. V26 with `GGML_CUDA_IQ2S_VOLTA_EP=1`, legacy `GGML_CUDA_IQ2S_VOLTA=1`, and
   `EP_REFERENCE_NATIVE=1`. Both unfused and fused paths passed all three owner
   slices.

Per native mode: 9 cases (T=1/4/8 × owners `[0,170)`, `[170,341)`, `[341,512)`),
27 exact owned-output comparisons, exact foreign `+0.0`, and bitwise A/B/A. The
fused graph marks only the final GLU as output. Nsight confirmed
`mul_mat_vec_iq2_s_moe_volta<(bool)0>` for unfused and
`mul_mat_vec_iq2_s_moe_volta<(bool)1>` for fused, with `ep_iq2s_volta` selected;
the fused run also logged `ep_fused_mmvq`.

V24/V26 flag-off digest baseline:

- V24: `/home/debian/ac922-monster/experiments/standard-opt-20261014/results/ep-consumer-runtime-perf-v24-20261015T133443Z/`
- V26 flag-off: `/home/debian/ac922-monster/experiments/standard-opt-20261014/results/ep-consumer-runtime-perf-v26-20261015T133529Z/`

V26 native artifacts:

- Unfused: `/home/debian/ac922-monster/experiments/standard-opt-20261014/results/ep-consumer-runtime-perf-v26-20261015T133552Z/`
- Fused: `/home/debian/ac922-monster/experiments/standard-opt-20261014/results/ep-consumer-runtime-perf-v26-20261015T133639Z/`

The bounded test read the actual IQ2_S tensor bytes from the target GGUF and did
not load the target model. GPU2 returned to 38 C, about 1 MiB idle allocation,
and no compute process.

Rebuild with `build-iq2s-fixture.sh` for a selected runtime/output path. The V26
native binary can be built with `build-iq2s-native-v26.sh` and run with
`run-iq2s-native-v26.sh`; `run-iq2s-default.sh` runs the flag-off arm.
