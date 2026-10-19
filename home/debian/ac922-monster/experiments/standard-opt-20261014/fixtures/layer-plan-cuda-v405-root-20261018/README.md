# V405 CUDA plan-rebind fixture

This folder contains a prepared actual-CUDA regression for the private four-copy plan rebind. It uses one CPU `GET_ROWS` node, six explicit CUDA `ADD` splits, a pinned CUDA-host input/output buffer, 24 asynchronous frames, exact host-oracle output checks, and four-slot destination-copy data checks.

The source and scripts are staged only. At preparation time the active local source tree did not yet contain `ggml-layer-plan-rebind-private.hpp`; the root agent expects that API in the R402 source reload. Exact preparation hashes and this boundary are recorded in [preparation-provenance.txt](preparation-provenance.txt). The build script stops before compiling if the header is missing.

No compile, host link, CUDA initialization, GPU execution, model load, or active engine edit was performed. The future run is gated by `RUN_LAYER_PLAN_CUDA_V405=1` and requires the root-selected six-device visibility and isolated R402 libraries.

Read [PLAN.md](PLAN.md) for the frame protocol and [COVERAGE.md](COVERAGE.md) for test boundaries. `capture-hostlink-manifest.sh` records the exact source/header/library closure after R402 is available; the quick host-link script will also save its expanded argv. `staged-hashes.sha256` covers the prepared fixture, scripts, and notes. The fixture is a primitive gate for later R5 device-tail work; it does not test R398 capture/replay or a model tail.
