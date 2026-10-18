# MTP fixed-prefix allocator fixture

This CPU-only fixture uses the real `llama_kv_cache` allocator with the actual model vocabulary loaded in vocab-only mode and a layer filter that prevents KV tensor allocation. It checks a 24-row first batch (four lanes x six rows), seed-only output flags, future-pad masking, rollback of the synthetic row, and three accepted four-lane rows at consecutive positions (D4).

For each of 100 rounds, it verifies that step 1 reuses only the removed pad cell, steps 2 and 3 use distinct cells without overwriting the real prefix or earlier accepted rows, each sequence reaches `seed+3`, and the used-cell span stays compact. The saved step-0 indices and causal mask must remain unchanged after all three accepted steps.

This fixture validates allocator/mask/index behavior only. It does not execute a model, compare hidden states, or validate the unrolled CUDA graph.

Run `build.sh`, then with CUDA hidden:

```sh
CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v16/lib OMP_NUM_THREADS=1 ./mtp-fixed-prefix-pad-fixture /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf
```
