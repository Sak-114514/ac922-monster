# V94 ordered Relay6 40 MiB fixture

This isolated fixture exercises the frozen V94 Relay6 source path at exactly
40 MiB. It compiles a fixture executable only, includes the frozen
`allreduce-p2p.cu` source, and links the V94 runtime closure. It does not edit
engine code or load a model.

## Bounded matrix

The run has one communicator per process and seven cases:

- Ordered kind 4, 40 MiB: stream-dispatch with zero ranks -1, 0, and 3.
- Ordered kind 4, 40 MiB: explicit `ggml_cuda_p2p_ar_graph_append()` with
  zero ranks -1, 0, and 3.
- Existing Relay6 kind 3, 40 MiB: stream-dispatch control with zero rank 3.

Each case checks all six ranks against the exact rank-order FP32 oracle,
including cancellation values, A/B/A replay, per-rank changed B inputs,
all-negative-zero signs, and output canaries. The validator requires the
actual selected kind and rank-order markers on all six ranks. Kind 4 must
report `cross-payload=2V`; the Relay6 control must report `4V`.

## V94 fixture-only build

The included `allreduce-p2p.cu` is unchanged across V92/V93/V94 and has
SHA-256 `b0790248c39350d862e09d0841575a8c7bad1e3c5de19fe50e55c8c9ad311b19`.
The build checks the V94 report identity and completion, compiles only this
fixture for SM70 with `-fno-fast-math -ffp-contract=off`, and records the
binary hash, V94 DSO hashes, and complete resolved dynamic closure.

    ./fixtures/relay6-ordered-40m-20261016/build-v94.sh v94-host1

The host build does not launch a GPU. The GPU runner requires the explicit
root authorization variable, verifies the V94 report and frozen DSO closure,
and enforces one communicator, a 1 GiB per-GPU added-memory cap, at least
512 MiB free per GPU, temperature below 58 C, a 15-second per-case timeout,
and a 60-second total budget. It preserves stdout and stderr separately for
every case.

    RELAY6_40M_GPU_WINDOW_AUTHORIZED=1 \
      ./fixtures/relay6-ordered-40m-20261016/run-bounded.sh v94-40m-$(date -u +%Y%m%dT%H%M%SZ)

The seven authorized cases passed in the V94 window. Ordered kind 4 was
selected at 40 MiB for all six ranks through both entry paths and all three
zero-rank settings; the 40 MiB kind-3 dispatch control also passed. The
summary is `results/v94-ordered-40m-v94-40m-host1-20261016T122500Z/summary.json`
(SHA-256 `6059e7aa06d49c338d92a06495d135f39ada4ecad182639e2a3cb62c4b2eaf83`).
All rank-order, A/B/A, changed-input, negative-zero, canary, and memory gates
passed. The 40 MiB event timing is operator-fixture evidence only, not model
throughput or quality evidence.

No predecessor fixture logs were copied into this directory.
