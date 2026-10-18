# V28 hierarchical EP6 fragment-cache candidate

Static preparation only. V28 reuses the V27 runner and quality gates, uses the frozen V67 closure, and changes only metadata fragment-cache settings (`GGML_META_FRAGMENT_CACHE=1`, `CAP=64`, `MIB=128`). It adds exact V28-vs-frozen-V27 token-array comparison case by case; the original V27 auxiliary prefix failures remain active.

The V67 cache/parent-UID tiny gate and separate root GPU grant are required before launch. The bounded runtime is 30 minutes once. No model is loaded during preparation.
