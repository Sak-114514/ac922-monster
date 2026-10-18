# CUDA graph force-capture qualification fixture

This fixture is prepared for the opt-in `GGML_CUDA_GRAPHS_FORCE=1` dispatch
change in runtime V88. It uses the public GGML graph API and a single CUDA
backend. It does not load a model and does not change the engine source.

## Coverage

- F32 `MUL_MAT + ADD`, using an identity weight matrix and dyadic inputs so the
  independent host oracle is exact.
- Two graph shapes (`T=4` and `T=8`) with distinct graph UIDs.
- A/B/A execution with changed input values, testing cold capture, shape/UID
  transition, and returning to the original graph.
- Exact host-oracle output and canary checks for each frame.
- Separate force-off and force-on processes because the environment flag is
  sampled once per process. The output artifacts must compare byte-for-byte.
- Logs must show the normal cold/warm/replay route for force-off, and selected
  `compatible_changed_graph_capture` plus `compatible_graph_replay` reasons and
  a `requested=5 selected=5` summary for force-on.

The force path does not waive compatibility, owner, graph-key/signature, or
tensor-property checks. Source review confirms its branch is reached only
after those existing checks, and capture/replay uses the existing CUDA graph
capture/update implementation. This fixture only qualifies the small ordinary
F32 graph path. It does not establish that model-sized MMQ, QSA, MoE, recurrent,
or P2P collectives are capturable or faster; those require separate evidence.

## Build on AC922 after V88 freeze

```bash
fixtures/graph-force-v88-20261016/build-fixture.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/src \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v88 \
  v88
```

The build is host C++ only; it links the frozen runtime and does not compile
CUDA. Do not use an older runtime in place of V88.

## GPU execution

Run only after the root grants a bounded GPU window and sets the explicit
authorization variable. Pass one V100 UUID, not a device index:

```bash
GRAPH_FORCE_V88_GPU_WINDOW_AUTHORIZED=1 \
  fixtures/graph-force-v88-20261016/run-compare.sh \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v88 \
  fixtures/graph-force-v88-20261016/build/graph-force-v88 \
  GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb v88-unique-tag
```

The runner refuses to start without the authorization variable, a clean idle
GPU, less than 512 MiB pre-existing allocation, and temperature below 58 C. It
uses a 45-second timeout and aborts above 1 GiB total allocation or 58 C. The
force-off arm runs first; the force-on arm runs only if that arm passes.

## V88 tiny qualification result

Runtime V88 build ID `20261016T100644.024428Z` was host-linked with this
fixture, then GPU3 (`GPU-fadb0459-4d30-907b-289a-862b3014f165`) ran one bounded
OFF/ON comparison at `2026-10-16T10:17:03Z`.

- Both arms passed all 5 exact-oracle frames and canary checks.
- Force-off logged the expected cold eager calls, A/B capture, and final A
  replay. Force-on logged immediate cold capture for both UIDs and replay for
  changed inputs; its final path summary was `requested=5 selected=5`.
- Force-off and force-on output artifacts compare byte-for-byte; both SHA-256
  values are `6287f48d8051f87d2b78894f7f9e7f49bc9583d9e61dfbe3b8fc638a2c45ab53`.
- GPU3 postflight: 4 MiB allocated, 38 C, 0% utilization, no compute PID.

Saved logs and binary outputs are under `results/v88-gpu3-20261016T1017Z-*`.
This verifies only the small ordinary F32 graph path; it does not validate
model-level MMQ, QSA, MoE, recurrent, or P2P graph capture or performance.
