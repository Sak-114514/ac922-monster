# Selective Meta replay guard CPU fixture

This fixture uses the production Meta backend and scheduler implementations from this experiment tree, compiled as isolated translation units and linked with frozen CPU/base runtime libraries. It creates two independent Meta backend contexts over the same CPU devices and shares one static WEIGHTS Meta buffer between them.

The bounded sequence is cold A, cold B, hot A, then a concurrent hot A replay loop while B allocates a new graph with a shorter view of the same shared weight tensor. It compares exact scalar outputs and checks shared-buffer generation/cache identity around a hot replay. No model, CUDA initialization, GPU work, cache eviction, or broad I/O is used.

The scheduler readiness gate binds the exact root graph pointer and UID to its allocated splits. A scheduler reset or graph switch must make readiness false before the next Meta dispatch. Run build-fixture.sh; then run with GGML_META_REPLAY_GUARD=1 and GGML_META_REPLAY_GUARD_TRACE=1.

This fixture is not a full llama_context/server correctness or concurrency proof. It is a bounded CPU Meta backend ownership fixture only.
