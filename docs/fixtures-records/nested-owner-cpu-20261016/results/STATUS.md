# Nested Meta owner CPU fixture status

- Fixture: `nested-owner-cpu-fixture.cpp`; public GGML APIs only.
- Static source check: passed on AC922 against current private public headers; no core build.
- V62 build: binary SHA-256 `5d2531afc86e9c3c98c059a9d1946d58f98e2c2df7e6e98a49b5c5c94f2c679f`; links only `libggml-cpu` and `libggml-base`.
- V62 runtime: expected exit 134 at `init_tensor reached unregistered non-static WEIGHTS tensor A.ple.tap_view.0`; log `v62-expected-failure.log`.
- V63 build: build ID `20261016T013345.175233Z`; binary SHA-256 `5a23acefa380733d0736be6d2465e0cc9334df0a86feb4ae506cc9a5abd942b4`; links only `libggml-cpu` and `libggml-base`.
- V63 runtime: PASS; exact public `ggml_backend_tensor_get` reads for A/B concurrently, same-address/same-offset A reset/rebuild, B after A reset, and B after A owner teardown. Log `v63-cpu-pass.log`.
- Safety: `graph_compute=0`; no model/GGUF load; no CUDA/NCCL library in either binary; no core source edits.
