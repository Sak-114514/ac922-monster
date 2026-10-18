# V203 Q5_K output-head row-shard result

Run tag: `v203-output-q5k-row-shard-once`  
Runtime: `runtime-perf-v196`; one host-only C++ TU, no NVCC.  
Scope: operator-only head proof; no model load.

## Input and allocation

The fixture read GGUF metadata and only the `output.weight` payload. The runtime printed `type=q5_K`, shape `2560x248320`, `437043200` bytes, row size `1760` bytes, and absolute data offset `11026368`. The head was not converted or requantized. The four row slices contain 62,080 rows each; the three replicas on GPUs 0/1/3 each contain 109,260,800 bytes. GPU5 retains the full head and computes its final quarter through a view.

The hidden inputs are deterministic synthetic F32 vectors; each graph step uses M=1. The D2/D4 cases run two/four such steps. The reference computes the full matrix on GPU5. The row-shard path computes on GPUs 0/1/3/5, copies the three remote F32 shards to GPU5, concatenates them, and runs `ggml_argmax` there.

## Results

All gathered logits matched the full-GPU5 reference bit for bit, and all argmax IDs matched.

| Case | Full GPU5 median | Four-GPU row-shard median | Ratio | Remote peer bytes | Full logits checked |
|---|---:|---:|---:|---:|---:|
| D2, 2 × M=1 | 1,522.408 µs | 548.982 µs | 2.77× faster | 1,489,920 | 496,640 floats |
| D4, 4 × M=1 | 3,032.292 µs | 1,073.460 µs | 2.82× faster | 2,979,840 | 993,280 floats |

Both cases reported `bitdiff=0`, `max_abs=0`, and `argmax_diff=0`. The argmaxes were `[41867, 69365]` for D2 and `[41867, 69365, 69788, 80846]` for D4. Timing is warm-replay wall time for full GPU5 matmul+argmax versus four-device matmul, peer gather, concatenation, and GPU5 argmax.

## Headroom and runtime evidence

Tensor-buffer totals stayed under 1 GiB per participating card. D2 row-shard totals were 109,777,920 bytes on GPUs 0/1/3 and 444,016,896 bytes on GPU5. D4 totals were 110,295,040 bytes on GPUs 0/1/3 and 450,990,592 bytes on GPU5. The fresh `nvidia-smi` preflight showed about 16,123 MiB free on GPUs 0–3 and 16,106 MiB on GPUs 4/5. The fixture's backend free-byte probe was lower (15,815 MiB on GPUs 0/1/3, 15,797.6 MiB on GPU5 in D2 preflight) and still above all guards. Postflight showed 16,123 MiB free on GPUs 0–3 and 16,105 MiB on GPUs 4/5, temperatures 38–39°C, and no compute PIDs.

The host binary SHA-256 is `0cd396c00c1efa6c1ed6b8962a163400292059962a46845896ddd3daf13dc82b`; fixture source SHA-256 is `b29c4bd8f413eb8276ed8947d5108bcb657c55ce9298361b048db4798993a6ab`; raw log SHA-256 is `1013b75efa1e31c7d53137c6648edc0fcf92fee9d49e583e174a4fc6ce701d7a`. `libllama.so.0`, `libggml.so.0`, and `libggml-base.so.0` resolved from `runtime-perf-v196/lib`. `ldd` resolved CUDA/CPU backend DSOs from `build-reuse/bin`; their hashes matched the frozen V196 copies: CUDA `5f663e079037bee6fe0d9cf1246433566e7103d53699ca659574f5efcbd6e53a`, CPU `d51a3e28fafe2fc03584d790512d4dcb88583f9949647602e5cbef1fc0d71781`.

Raw output: [`v203-output-q5k-row-shard-once.log`](v203-output-q5k-row-shard-once.log). Host-link dependency output: [`v203-host-link-ldd.txt`](v203-host-link-ldd.txt). The host-only build recipe is [`build-v203-host.sh`](../build-v203-host.sh); the run gate is [`run-v203-once.sh`](../run-v203-once.sh).

## Evidence boundary

This demonstrates an exact, faster isolated Q5_K output-head operator on synthetic inputs with the production output-head bytes. It does not exercise production MTP hidden states, the live model graph, sampling semantics, or full-model throughput. No engine/CUDA source was changed.
