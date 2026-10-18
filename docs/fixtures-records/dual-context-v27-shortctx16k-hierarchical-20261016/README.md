# V27 hierarchical EP6 16K/request candidate

V27 has completed one supervised run; see `STATUS.md` for the overall exit-1 verdict and per-gate results. No autorun or retry is configured. It reuses the exact V26 runner binary and environment, switches the whole runtime closure to V65, and keeps `GGML_CUDA_GRAPH_STRUCTURAL_TRACE=1` plus the existing sparse timing flags. No quality, capacity, or safety setting changes.

V65's intended core change clones merged physical graph descriptors and all internal `view/src` mappings. Temporary AllReduce rewrites stay in owned `ctx_m`; `bcj.nodes` remain unchanged. It retains raw graph UID and rejects opaque extras fail-closed. V26's crash exposed stale `allreduce_1` descriptors, so the old/new Meta merge immutability regression is a required gate.

Launch requires the V65 tiny gate, the Meta merge regression, and root's separate GPU grant. The supervisor is fail-closed and invokes its wrapper through `/bin/bash`; see `STATUS.md` and `v27-shortctx-plan-manifest.json`.
