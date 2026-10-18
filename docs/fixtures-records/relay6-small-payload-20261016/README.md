# Relay6 small-payload threshold fixture

Fixture-only V97 comparison for the lower Relay6 minimum. It uses one six-rank communicator and the V97 AllReduce source snapshot; it does not load a model.

| Process mode | `GGML_CUDA_P2P_AR_RELAY_MIN_BYTES` | Payloads |
|---|---:|---|
| `default` | unset, historical 204800-byte floor | 64, 80, 128, 160, 192, and 240 KiB |
| `min65536` | 65536 | 64, 80, 128, 160, and 192 KiB |

Each payload runs eager dispatch plus dispatch-captured and explicit graph-append graphs. Each graph gets five replays (`A0,B,A2,C,NEGZERO`); 96-byte one-shot calls are interleaved around the path cases on the same communicator. The validator compares exact FP32 output hashes across modes for common sizes, checks canaries and rank-order results, verifies every rank's fixed-grid epoch after each collective, and preserves both per-rank event times and all-rank completion wall time.

Host link: `build-v97.sh [AC922 experiment root] [tag]`.

The bounded runner requires root's one-window authorization environment variable and checks all six GPU PIDs, free memory, temperature, per-device memory growth, and postflight idle state. It runs only the synthetic small-payload fixture.
