# V267 actual-GPU shared MTP embedding fixture

This fixture validates only the V266 model-owned immutable embedding cache against the real IQ3_S `token_embd.weight` tensor. It is not an inference, quality, throughput, or performance test.

The model loads metadata only (`LLAMA_LOAD_MODE_NONE`, `no_alloc=true`). The fixture then reads only the 273,152,000-byte `model.tok_embd` payload into an externally owned CPU backend buffer. The no_alloc loader may attach a model-owned empty buffer wrapper while leaving `tok_embd.data` null; the fixture preserves that wrapper and rebinds only the source tensor to its separate CPU payload buffer. It uses one 8 MiB file-read staging buffer; even including both GET_ROWS host comparison outputs, statically and dynamically checked scratch stays below 32 MiB. It does not materialize any other weight and keeps the source buffer alive until after `llama_model_free()`. The helper's internal cache is model-owned and is destroyed while the model and source are still valid.

The fixture is designed to check four GPU0 helper requests with real GPU token tensors reuse one pointer, GPU1 gets a separate pointer, and actual IQ3_S `GET_ROWS` outputs match an independent legacy GPU0 copy byte-for-byte for varied IDs and token counts. Each graph/scheduler/context is destroyed before the next comparison.

Resource guards: at most 2 GiB of fixture-owned tensor allocation per GPU; runner requires all six fixed GPU UUIDs, at least 512 MiB free on every card, temperature below 58 C, polls every two seconds, and kills the process group after 180 seconds or a guard violation. It also sets `LLAMA_MTP_BASE_DEVICE=0` to reproduce V245's audited layer-48 placement and requires the runtime log to confirm body CUDA0/output CUDA5. These controls are local fixture safeguards, not a substitute for the root orchestration GPU-window gate.

## Verification sequence

1. Run `./tests/static_check.py` locally (source-only, no compile or model access).
2. After root freezes the V266 runtime and explicitly authorizes one hostlink, set `V267_HOSTLINK_AUTHORIZED=1`, `V267_RUNTIME_DIR` to that frozen runtime's `lib` directory, and a unique `V267_BUILD_TAG`, then run `./hostlink.sh`.
3. Only after a separate root GPU-window grant, set both `CUDA_VISIBLE_DEVICES` and `V267_EXPECTED_GPU_UUIDS` to the same comma-separated six UUIDs in the approved physical order and run:

   ```sh
   V267_GPU_WINDOW_AUTHORIZED=1 \
     python3 ./run_guarded.py ./build-<unique-tag>/shared-mtp-embedding-gpu-v267 \
     /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf \
     ./results
   ```

The hostlink and GPU runner intentionally refuse to start without their explicit authorization environment gates. Keep every build and run in a unique directory/log; do not overwrite earlier attempts.
