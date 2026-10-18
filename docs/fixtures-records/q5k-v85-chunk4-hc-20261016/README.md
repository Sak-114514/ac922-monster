# V85 Q5_K chunk4 and HC mode-2 fixture

This fixture reuses the exact 2,048-row Q5_K `output.weight` slice and 16-row deterministic F32 activation generation from `q5k-t4-t8-dispatch-20261016`. It requires T16 with `GGML_SM70_DECODE_LINEAR_CHUNKS=4` to be bitwise equal to four T4 calls and compares FP64 dequantized-row samples. It also checks the existing F32 `GGML_OP_HC_FUSED` mode-2 kernel at T16 against four T4 calls, using identical residual, block, and injection tensors with canaries.

Host compilation and fixture construction are preparatory. Run only after V85 is frozen and GPU3 is explicitly assigned. The runner does not load a model or modify core sources.
