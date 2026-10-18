# V28 preparation status

**Root authorized one V28 model load; launch still requires the supervisor fresh GPU/BMC/fan gate.**

- V67 runtime build ID `20261016T023533.284373Z`, report SHA-256 `ceb058ccab7513894385d94c80ac0ffaaa4aff6c355d55a667ce0f8184d64e8a`. Runner is reused byte-for-byte; no engine or runner build. V67 ldd resolves all seven application libraries from its whole runtime closure; evidence is in `v67-reused-runner.{file,ldd,sha256,status}.txt`.
- V28 toggles only fragment-cache settings: `GGML_META_FRAGMENT_CACHE=1`, `GGML_META_FRAGMENT_CACHE_CAP=64`, `GGML_META_FRAGMENT_CACHE_MIB=128`. V27 quality/capacity/structural-trace/timing/thermal settings remain unchanged; CUPTI/preload remain off.
- V27 baseline hashes: diagnostic `16cb38688dc2e4581c0f5990dcf432949f431dad842e44ac5560c082cdf6f994`; clean `beee7b03194cea10d8f56225bf2f8bae83c014f461324b8b9603bb746b95ae36`. V27 first-20-token auxiliary prefix evidence is in `V27-AUX-PREFIX-BASELINE.md`; the original two prefix failures remain active.
- Added a case-by-case V28/V27 exact generated-token-array comparison for every matching scenario/flow in diagnostic and clean outputs; it records first mismatch index and first 20 IDs/text.
- V67 cache/parent-UID tiny gate PASS is recorded in `/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/meta-fragment-cache-20261016/results/v67-host-r2-on-20261016T024529Z.log` (SHA-256 `96afe9ca8e48495220bfb3a5fa7c5a343a8d51296dfde9acaab7bb501a91c9bf`): three inner hits/nine misses per context, two parent invalidations, eight outer ARs, twelve merged inner graphs, and A/B/A + new-parent-UID exact/canary PASS.
- Root has granted one full-model load. The supervisor still performs a fresh live safety gate, has a 30-minute limit, and has no automatic retry.
