# V31 parent-manifest plus nested-compose candidate

V31 reuses the already qualified V29 two-bank × two-lane D3 runner unchanged. It keeps the 16K-per-sequence capacity, Q8_0 target/draft KV, full expert set, standard QSA indexer, target UB1024, draft UB512, CPU-copy mode 1 with a 1 MiB per-copy cap, V67-qualified fragment cache, and the existing V29/V27 correctness gates.

The only intended runtime flags added to the V29 profile are:

- `GGML_META_FRAGMENT_PARENT_SIG=1`
- `GGML_META_NESTED_COMPOSE=1`

The dynamic libraries come from the frozen V75 closure (`20261016T043346.193648Z`, report SHA-256 `8dac3a5cb624684dfc7583c7bc1ea96804478f5bb326cfc43b41b844cd6a8100`). The runner is the byte-identical V29-qualified binary (SHA-256 `8f25242b2c8bb8fe09eba2002386d5d98f5b8de08886e9e491a6499b76a5c44d`); it is not rebuilt.

The full source profile is archived as `v29-profile-source-manifest.json` (SHA-256 `973eb5016e394fc6116a415af30a6b0e211229aeae191620e5f46bd0cc754f18`). V31 changes only the two candidate flags above and the V75 dynamic-library path/build ID; profile-family diagnostics and V31 launch-gate variables are recorded separately.

The full-model launcher remains fail-closed until both the V75 parent-manifest/compose tiny gate and an explicit root GPU-window grant are supplied. Its 16K capacity is a test limit, not a claim that larger per-sequence contexts fit. The run preserves the native semantic checks, full-F32 prefill-logit comparisons, exact-token and auxiliary-prefix checks, clean performance window, and case-by-case V27 token-array comparison. A performance sample is not accepted as a win if any retained quality gate fails.

No model or GPU run is performed during host preparation. The supervisor applies the existing fresh BMC/fan/GPU safety checks, does not alter BMC state, and launches a single no-retry systemd run with a 30-minute maximum only after the explicit gates pass.
