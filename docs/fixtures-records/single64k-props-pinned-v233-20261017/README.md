# V233: single-request 64K UB256/512 with cached CUDA properties and pinned floor

This fixture is derived from the V219 two-family, one-model runner. It retains
the same V218-era quantization, model math, MTP D2 path, QSA F32 prefill indexer,
virtual TopK, 18 GiB PLE placement, graph binding keys, and other V219 fast
paths. The only added runtime knobs are V228's
`GGML_CUDA_CACHE_DEVICE_PROPERTIES=1` and a 16 MiB minimum for the existing
layer pinned-input allocator; its 1024 MiB cap remains unchanged.

The runner creates one model, then runs two context families in order:

1. UB256, one 24-token-canary scenario with the existing 249-token Paris prompt,
   followed by one uncached 65536-token input with eight unscored output tokens.
2. UB512 with the same canary and long-input sequence.

The 64K request uses `n_batch=32768`, so target and MTP each receive two ordered
logical chunks. Per-request context is 66048. Prompt KV is cleared between
scenarios, and the new generic single-request metrics divide the actual input
token count by the measured whole-prefill time and target-prefill sync sum.

No source under `src/` is changed. No CUPTI or Nsight profile is enabled. The
long outputs are not semantically scored, and their latency is not a full-model
quality or production claim.

## Prepared artifacts

- `dual-context-v233-single64k-props-pinned.cpp`: V219 runner adaptation.
- `prepared-output/copy0/single64k-ub256/scenarios.json` and
  `prepared-output/copy0/single64k-ub512/scenarios.json`: fresh labels for each
  UB family, each with Paris first and one 64K case.
- `v233-root-profile.json`: frozen V228 runtime, inherited V219 environment,
  root guard thresholds, source and scenario identities.
- `root-host-link-v233.sh`: host-only runner link recipe. It was not executed.
- `run-v233-root.py`: root-owned guard/one-model launch wrapper. It was not
  executed and reads any sudo credential from stdin without writing it to disk.
- `verify-v233-static.py` and `static-contract.json`: static inputs, source and
  profile checks only.

The result status remains launch-pending until root confirms the new runner's
actual `ldd` paths and starts it under the in-band SDR/hwmon guard.
