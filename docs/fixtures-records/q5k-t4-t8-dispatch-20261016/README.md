# Q5_K T4/T8/T16 dense dispatch fixture

This fixture copies only the first 2,048 complete rows of `output.weight` from the GGUF and feeds the exact same 16 F32 activation rows to three actual GGML CUDA graphs: four T4 calls, two T8 calls, and one T16 graph with the V83 `GGML_SM70_DECODE_LINEAR_CHUNKS=8` route.

It does not load a model, scan tensor payloads beyond the selected rows, alter core sources, or modify weights. The C++ runner uses the production Q5_K `ggml_mul_mat` graph and GGML CUDA backend. It checks host canaries/finite output, reports pairwise RMS/max/bit differences, times three runs per arm, and computes FP64 dot samples from `dequantize_row_q5_K`.

The GGUF extractor uses the GGUF metadata API, seeks directly to `output.weight`, and reads only the selected 2,048 rows. Build and run on AC922 with the exact runtime and one-GPU UUID selected through `CUDA_VISIBLE_DEVICES`.
