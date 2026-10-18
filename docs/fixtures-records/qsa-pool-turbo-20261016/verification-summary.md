# Verification summary

- Staged memory lifecycle patch applied in a temporary source copy, followed by the final physical CELLSEL patch. No live source files were edited.
- C++17 syntax-only compile passed for qwen4exp.cpp and llama-memory-hybrid-idx.cpp using GGML_USE_CPU, GGML_USE_CUDA, LLAMA_BUILD, GGML_BACKEND_SHARED, and GGML_SHARED defines. Both TUs used the patched compile-src-v6 tree; no linking, engine build, or GPU run occurred.
- Remote run-cpu-tests.sh passed the private ledger state tests, shared production build_cell_scatter_plan helper test, and Python physical-cell selection oracle.
- The combined patch dry-run succeeded before syntax compilation; final patch is staged only.

The selection oracle uses injected block scores. Compact GEMM score bit parity against the full reference path remains untested and requires GPU qualification.
