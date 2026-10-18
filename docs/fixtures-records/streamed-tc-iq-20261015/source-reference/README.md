# Strata schedule provenance

The standalone fixture adapts only the Volta tile schedule, XOR swizzle, `m8n8k4` fragment mapping and independent FP32 accumulator structure from the MIT-licensed Strata-AC922 kernel.

- Repository: https://github.com/eelgaev/Strata-AC922
- Pinned commit: `1f029aadc0441ebc07ab00f411f2537359ceace3`
- Source path: `src/kernels/cuda/fused_expert.cu`
- Source SHA256: `d0d70e7ff40687d73d9418ea4af570406d3cc6d5fa738030eba73d9d7b7c3caa`
- License: `../LICENSE-strata` (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors)

The fixture's quantized tile loader/dequantization bridge and grouped routing harness are separate local code. The Strata file is retained only as the pinned source reference; it is not compiled into the fixture.
