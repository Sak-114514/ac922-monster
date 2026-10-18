# V269: Meta6 PLE to ordinary CUDA layer graph smoke

This fixture owns only this directory. It checks whether the existing GGML
backend scheduler can run a PLE-only six-member Meta backend and then cross to
an ordinary CUDA0 operation. It deliberately does not touch the llama model
device table, full model weights, engine source, or runtime configuration.

## Contract covered

- PLE uses the V122 ledger's real global tensor format: `IQ4_NL` (GGML type 20),
  width 160, shape `[160, 320001536]`, 28,800,138,240 bytes. The fixture creates
  only a 12-row synthetic payload in that same packed type; it does not read the
  multi-gigabyte PLE tensor or load the full model.
- The Meta6 static weight callback partitions rows uniformly on axis 1; with
  twelve fixture rows, each rank owns two rows. `GET_ROWS` receives global IDs
  that cover all six ranges. The graph follows production's row-id zero,
  repeat, and add join, then requests an ordinary `scale` node on physical
  CUDA0.
- CPU and CUDA graph outputs must match bit-for-bit over twenty distinct input
  frames. Input and output host buffers have canaries. Each frame gets a fresh
  graph UID and the scheduler requests its four normal graph-copy slots.
- The fixture asserts that the joined PLE value is `MIRRORED` before allowing
  the CUDA0 consumer. If that layout cannot safely leave Meta6, the process
  fails at that assertion rather than masking the condition with a manual
  gather.

The public scheduler API does not expose its active copy-slot index. The log
records the slot implied by `ggml_backend_sched_alloc_graph`'s round-robin
contract (`frame % n_copies`); exact output on all twenty distinct frames is
the runtime stale-slot check. A future source change can add direct slot
telemetry if a public accessor becomes available.

The Meta6-to-CUDA0 transition exercises the real scheduler copy path. The
current generic fallback is synchronous and host-staged: Meta's async copy
callback does not implement Meta-to-CUDA, and the CUDA callback accepts CUDA
buffers only. This smoke is therefore a correctness/lifetime gate, not a
direct-P2P performance claim.

## Static check and host link

Run the static ledger/source contract locally:

```sh
./check-v269-contract.py ../layer6-two-bank-v57-20261017/v122-layer-bytes.json
```

After root freezes the runtime closure, host-link on AC922 (this does not run
CUDA):

```sh
./build-v269-host.sh /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-vNN v269
```

## GPU run

Do not run this fixture until root explicitly grants the GPU window. The guarded
runner also refuses if any selected card has a compute PID, more than 512 MiB
already in use, less than 512 MiB free, utilization at least 5%, or temperature
at least 58 C. It stops after 180 seconds:

```sh
AC922_LAYER6_PLE_GPU_WINDOW_AUTHORIZED=1 \
CUDA_VISIBLE_DEVICES=<six validated V100 UUIDs in physical order> \
  ./run-v269-guarded.sh ./ple-meta6-cuda-mix-v269-v269 ./backend-v269 v269
```

## Execution record

On 2026-10-17, the fixture was host-linked once against frozen V264 and the
guarded 20-frame GPU run passed. The result log is retained at
`results/root-v269-meta6-cuda0-20261017t125801.log`; the host-link receipt is
`build-v269.txt`. All six cards passed the preflight at 37–38 C with at least
16 GiB free and no compute PIDs. After the tiny run they were at 39–40 C, with
at most 14 MiB used. All six PLE row shards matched the global IQ4_NL byte
payload, all twenty outputs matched the CPU oracle exactly, all canaries held,
and every frame had a fresh graph UID. This qualifies only the tiny mixed
backend contract; it does not establish model integration or performance.
