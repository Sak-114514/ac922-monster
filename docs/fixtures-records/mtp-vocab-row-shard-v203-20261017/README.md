# V203 MTP output-head row-shard operator fixture

This is a head-only operator test for the ordinary Q5_K `output.weight`; it does not load the model. It reads GGUF metadata, validates `output.weight` as Q5_K `[2560, 248320]` / 437,043,200 bytes, then reads only that tensor payload using the metadata offset. Every shard copies original Q5_K row bytes; no dequantization, requantization, or weight conversion is performed.

The baseline computes the full head on visible GPU5. The candidate retains that full tensor on GPU5, uses a view for its last quarter, and copies the first three contiguous 62,080-row byte ranges to GPUs 0, 1, and 3. Each GPU computes its own Q5_K `ggml_mul_mat` rows for M=1 inputs. The remote F32 slices are copied to GPU5 through `ggml_backend_tensor_copy_async`, concatenated there, and passed through the existing `ggml_argmax`. D2 and D4 denote two and four sequential graph steps with M=1 per step. Activations are deterministic synthetic inputs; this proves the real output-head operator and communication path, not model semantics or full-model throughput.

The harness compares every gathered F32 logit bit-for-bit with the full GPU5 reference and checks every argmax. It prints median wall time for full-GPU5 compute/argmax and the four-GPU row-shard path including gather/argmax, plus buffer sizes and requested peer bytes. Each backend tensor buffer is limited to 1 GiB per card; the preflight also requires 512 MiB free above the head/shard allocation. A P2P capability check is required from GPUs 0/1/3 to GPU5.

The build script is host C++ only and links to `runtime-perf-v196`; it invokes no CUDA compiler. Both build and run scripts require explicit environment gates. The one authorized run is archived in [`results/v203-summary.md`](results/v203-summary.md) and [`results/v203-output-q5k-row-shard-once.log`](results/v203-output-q5k-row-shard-once.log).

```sh
MTP_VOCAB_ROW_SHARD_BUILD_AUTHORIZED=1 ./build-v203-host.sh
```

The fixed visible UUID order `0,1,2,3,4,5` corresponds to physical owners 0/1/2/3/4/5. The fixture uses physical GPU0/1/3/5 and passes the actual GGUF path:

```sh
CUDA_VISIBLE_DEVICES=<six-fixed-UUIDs> \
  LD_LIBRARY_PATH=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v196/lib \
  MTP_VOCAB_ROW_SHARD_GPU_AUTHORIZED=1 ./run-v203-once.sh
```
