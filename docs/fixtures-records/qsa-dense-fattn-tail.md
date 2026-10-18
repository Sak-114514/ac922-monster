# QSA dense FlashAttention tail fixture

This small fixture isolates the CUDA dense masked FlashAttention operation used after QSA top-k selection. It uses deterministic Q data, a Q8_0 KV cache, a causal F16 mask, and compares T=27, 32, and 35 against a scalar FP32 reference over the same dequantized Q8_0 cache.

The Meta split callback in llama-model.cpp keeps attention shards aligned to whole GQA groups. For this model, n_head=24, n_head_kv=2, and GQA=12. A non-empty member shard therefore has QH=12 and KVH=1; empty members rotate across layers. The fixture models that non-empty member shape on one visible CUDA device.

This fixture does not load model weights, construct the indexer, or reproduce the graph-level gather-vs-dense branch in qwen4exp.cpp. It isolates the dense FLASH_ATTN_EXT tail-query computation only. The causal mask exposes 256 KV cells, with positions after the prompt masked, matching a fresh short prompt's padded KV view.

Reference gate: every CUDA output must be finite and satisfy
abs(actual-reference) <= 0.005 + 0.02*abs(reference).
The CPU reference first quantizes K/V to Q8_0 and then dequantizes those same bytes to FP32, so the only remaining difference is the CUDA attention implementation and its internal low-precision staging.

Compile only; this command does not execute the fixture:

    ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014
    RT=$ROOT/runtime-standard-combined
    c++ -std=c++17 -O2 -Wall -Wextra \
      -I$ROOT/src/ggml/include -I$ROOT/src/ggml/src \
      $ROOT/fixtures/qsa-dense-fattn-tail.cpp \
      -L$RT/lib -Wl,-rpath,$RT/lib \
      -lggml -lggml-base -ldl -pthread \
      -o /tmp/qsa-dense-fattn-tail

If the root chooses to run it later, use the same runtime backend plugin directory:

    ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014
    RT=$ROOT/runtime-standard-combined
    LD_LIBRARY_PATH=$RT/lib /usr/local/bin/with-ac922-cuda525 \
      /tmp/qsa-dense-fattn-tail $RT/lib CUDA0

The run command uses only CUDA0 and performs three tiny attention cases. It is provided for the parent agent; it has not been executed as part of this task.
