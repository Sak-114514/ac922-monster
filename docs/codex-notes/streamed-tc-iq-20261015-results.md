# Streamed Volta Tensor Core IQ fixture results

Date: 2026-10-15

Status: bounded standalone fixture passed its local numerical gates for all ten valid type/K shapes. The streamed-TC path is rejected for engine integration in its current form because its kernel core is about 4.7-5.6x slower than the grouped MMQ core in this fixture.

## Scope and provenance

This fixture is isolated under `fixtures/streamed-tc-iq-20261015/`; it does not edit engine sources or dispatch. It reads the existing `volta-tc-real` packed-row fixture, which contains 16 actual packed rows for each of GGUF experts 0, 17 and 511. The other routed expert rows are repeated from expert 0 so the fixture can exercise the 512-expert grouped schedule without reading or loading a full model. The route is synthetic and deterministic, with 512 input rows and top-k=10 assignments. The skew arm has 258 active experts, mean 19.845 assignments per active expert, and one expert each at 64 and 128 assignments.

The tile shape, swizzle, fragment mapping and independent FP32 accumulator schedule are derived from the MIT-licensed Strata-AC922 source pinned at commit `1f029aadc0441ebc07ab00f411f2537359ceace3`. The exact source reference and license are in `fixtures/streamed-tc-iq-20261015/source-reference/strata-fused-expert.cu` and `fixtures/streamed-tc-iq-20261015/LICENSE-strata`. SHA256 of the source reference is `d0d70e7ff40687d73d9418ea4af570406d3cc6d5fa738030eba73d9d7b7c3caa`; SHA256 of the license file is `949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0`.

The seven packed decoders reuse the production MMQ K-tile loaders, then convert the signed codes and group scales to F16 in shared memory. There is no global or persistent F16 weight cache. Activation F32-to-F16 uses explicit CUDA round-to-nearest-even conversion. MMA is Volta `m8n8k4`, F32 accumulation, F32 output.

## Verification

Build command: `fixtures/streamed-tc-iq-20261015/build-fixture.sh`.

- Standalone fixture build passed with `-arch=sm_70`; generated PTX contains `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`.
- CPU reference preparation passed for all ten valid cases. Five gate formats use K=2560. IQ4_NL down uses K=192, 224 and 640; Q2_0 down uses K=192 and 640. Q2_0 K=224 is excluded because 224 is not divisible by its 64-element quant block.
- On physical GPU2 (`GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb`), GPU-decoded F16 weights for experts 0, 17 and 511 matched the CPU GGML dequantize-then-RNE reference bit-for-bit for every output row and padded K tile in every case.
- GPU F32-to-F16 activation conversion matched the CPU RNE reference bit-for-bit.
- Four routed output rows per case were compared against explicit F16-multiply/F32-accumulate and original-F32 references. Every comparison passed the frozen gates: F16 `atol=1e-4, rtol=1e-4, NRMSE<=1e-4`; original F32 `atol=1e-3, rtol=2e-3, NRMSE<=1e-3`. The maximum observed NRMSE was `1.01260366e-6` for the F16 reference and `3.22877504e-4` for the original-F32 reference. No sampled outputs were nonfinite and no tolerance failures occurred.
- Separate CUDA graphs for streamed TC (A) and original grouped MMQ (B) replayed A/B/A. The second A output was bit-exact with the first A output in every case. Leading and trailing output canaries passed. All full output buffers for both kernels were finite.
- GPU2 watchdog sampled at most 535 MiB used and 41 C. The run ended at 1 MiB and 40 C. The run was pinned to GPU2 only, with a 270-second timeout, 1024-MiB memory cap and 58 C temperature stop. No model or engine server was loaded.

The complete build, CPU preparation, first diagnostic arm and final passing arm are retained under `fixtures/streamed-tc-iq-20261015/`.

## Kernel-core timings

The table compares the fused K-tile decode plus Volta MMA graph against the original grouped MMQ kernel graph. MMQ consumes prequantized Q8_1 activations; its activation preparation is outside the device-core timing. The fixture records a naive host Q8_1 preparation time separately, which is not representative of the engine's GPU quantization path. F16 activation conversion is also reported separately. These are operator-fixture timings, not full-model prefill results.

| Format | K | N | Streamed TC core (ms) | Grouped MMQ core (ms) | TC / MMQ |
|---|---:|---:|---:|---:|---:|
| IQ2_XXS | 2560 | 192 | 2.7288 | 0.5149 | 5.30x |
| IQ2_XS | 2560 | 224 | 2.6259 | 0.5566 | 4.72x |
| IQ2_S | 2560 | 192 | 2.6358 | 0.5648 | 4.67x |
| IQ3_XXS | 2560 | 224 | 2.7847 | 0.5409 | 5.15x |
| IQ3_S | 2560 | 192 | 2.6487 | 0.5507 | 4.81x |
| IQ4_NL down | 192 | 2560 | 2.8095 | 0.5444 | 5.16x |
| IQ4_NL down | 224 | 2560 | 2.8081 | 0.5431 | 5.17x |
| IQ4_NL down | 640 | 2560 | 7.6442 | 1.3947 | 5.48x |
| Q2_0 down | 192 | 2560 | 2.7593 | 0.5140 | 5.37x |
| Q2_0 down | 640 | 2560 | 7.2294 | 1.3012 | 5.56x |

Average ratio across these ten cases is 5.14x slower for the streamed-TC candidate. This rejects the current tile schedule for production use despite passing the local arithmetic gates. It does not prove that every Volta Tensor Core approach is slower; this particular decode-and-stage kernel loses to grouped MMQ.

## Reproduction

From `/home/debian/ac922-monster/experiments/standard-opt-20261014`:

```bash
fixtures/streamed-tc-iq-20261015/build-fixture.sh
numactl --cpunodebind=0 --membind=0 \
  fixtures/streamed-tc-iq-20261015/streamed-tc-iq \
  fixtures/volta-tc-real --cpu-only --skew
fixtures/streamed-tc-iq-20261015/run-gpu2-bounded.sh
```

The GPU runner sets `CUDA_VISIBLE_DEVICES` to GPU2 by UUID and enforces the time, memory and temperature stops. It preserves stdout and watchdog logs under new fixture-local tags.
