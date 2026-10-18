# V41 status

Prepared against frozen V94 build `20261016T121300.582979Z` (`V94-build.json`, SHA-256 `95131cd92341a18b828a627236a1d61b598373059509b81e505502d7d9c19258`). The candidate changes target `n_ubatch` to 4096, leaves MTP draft `n_ubatch=512`, uses `GGML_META_CHUNK_NODES=0`, and enables `GGML_META_COLD_PREPARE_COHORT=1` with `GGML_CUDA_GRAPHS_FORCE=0`.

Host harness source and V40-derived launcher are generated locally with exact source SHA and replacement-count checks. Host compile/link against V94 is allowed; GPU/model launch is still blocked until both required local gates pass:

- V94 six-rank cold-cohort F32 oracle/canary gate (root reports this PASS; exact artifact still needs to be copied and hash-verified here).
- 40 MiB ordered Relay6 dispatch/graph-append gate for the 4096 target shape.

No full-model V41 run has been started. After the two gates pass, the supervisor still requires explicit root GPU-window authorization and applies the 58°C and 512 MiB per-GPU free-memory stop limits. There is no retry.
