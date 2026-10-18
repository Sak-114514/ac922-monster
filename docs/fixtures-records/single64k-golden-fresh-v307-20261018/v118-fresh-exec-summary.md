# V118 fresh graph executable gates

- V118 build report matches SHA-256 `7b0f01069e0ae98fa90e9464d1aa0e4f228b64d0f275ba6f73f20ae8496a6215` (run ID `20261016T195449.960522Z`). Its compile report records one CUDA and one C++ translation unit for this change.
- QSA reused the V110 long-batch binary with `LD_LIBRARY_PATH` selecting V118. Both 4096 and 8192 query cases passed full-vs-1024-chunk bitwise comparison, A/B/A, input immutability and cache canaries. The log reports `cuda_graph_fresh_exec selected=1` and `reason=explicit_fresh_exec` for both cases.
- Meta3 reused the existing `meta3-cold-cohort-v100-meta3-host1` binary and original bounded runner. It completed force0 / structural1 in 4 seconds under the original 30-second cap and 1024 MiB runtime memory guard.
- The Meta3 summary is `PASS`: capture and replay ran on devices 0, 1 and 2; all 96-byte all-reduce phases matched the exact oracle; canary and A/B/A passed; intermediate host syncs were zero; no XID/stall markers were found. The log shows fresh executable selection on all three devices and `explicit_fresh_exec` rebuilds.
- Postflight found no compute owners; cohort GPU usage was 2 MiB per device at 38–40°C. No model was loaded.

These are bounded fresh-executable and synthetic Meta3 gate results, not a model inference or performance claim. No binaries were rebuilt and no V110 artifacts were overwritten.
