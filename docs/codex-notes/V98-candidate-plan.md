# V98 BF16 decode panel

Reuse original gemv_sm70_bf16_kernel arithmetic with wide columns; no new quantization or lower-precision activations. Default OFF GGML_SM70_BF16_DECODE_PANEL. Only canonical chunk4/SM70/original BF16 whitelist/contiguous panels; preserve fallback. NC12/16 use ROWS4; NC24/32 use ROWS2. Existing 256-thread FMA/shuffle/warp-fold reduction is unchanged.

Local GPU fixture root-panel-20261016T145956Z: 36 correctness cases,108 ABA,17,033,436 values bitexact; K2560/N8192 and K10240/N2560 + N8195 tail; no canary/fault. NC16 operator latency260.096→190.464us and308.224→165.888us respectively,1.366/1.858x, not model gain. NC24/32 ROWS2 avoid slower high-register ROWS4 cases.

Integrate into one private CUDA TU plus one private direct header; incremental build only, retain V97 immutable runtime. Combine with valid existing paths in next long32K64K and multi2/4lane candidate; original model/FP32 activations/Q8 KV stay. Gate new dispatch before whole-model claim.
