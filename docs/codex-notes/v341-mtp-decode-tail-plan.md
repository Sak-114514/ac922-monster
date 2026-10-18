# V341 next candidate: ordinary MTP decode graph submission

Status: design, not implemented or built.

The retained four-bank D3 recipe uses ordinary MTP decode, not the slower dedicated D3 result. Extend the existing V329 same-graph bridge to stable ordinary MTP replay only, without touching kernels, weights or the capture policy.

- New independent CUDA registry TLS hook ggml_backend_cuda_set_device_tail_decode_hint, restoring previous value at scope exit. Never borrow the prefill hint: that changes force-capture decisions.
- Context opt-in LLAMA_MTP_GRAPH_DEVICE_TAIL=1; narrow scope Qwen4exp, physical LAYER, MTP context, ordinary1..32-row process_ubatch, no staged Turbo frame. Target decode untouched; dedicated-unroll path untouched initially.
- CUDA submission requires existing original graph instance, !cuda_graph_update_required, !cold, SM70, global bridgeopt-in and the separate hint. First capture/update follows original host graph, only stable replay tries helper. Existing whitelist, boundedcache32, stream/generation lifetime checks and normal host fallback retained.
- Explicit selected/decline logs and all6/backend keyed counts. Capacity is per CUDA backend context, not all banks combined. No LRU or unbounded cache expansion.

Verification: actual backend fixture with separate hook and stable replay (firstcapture expected unselected), exact repeated outputs and stream/lifetime fences; then retained V306 fourbank genericD3/native4+real4x16K512. Report shared/inclusive separately; no production switch based on one noisy sample.
