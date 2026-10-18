# Meta3 P2P64 / NCCL F32 / 4 MiB threshold route fixture

This is an isolated three-arm extension of `nccl-prefill-route-20261016`.
It uses the same deterministic Meta3 graph, inputs, CPU oracle, output hash,
all-rank checks, canaries, payload sequence, and `graph_compute + synchronize`
timing. Each arm runs as a separate process so communicator route and threshold
configuration stay fixed for its full lifetime.

## Arms

- `p2p64`: existing P2P AllReduce route with fixed grid 64 for every payload.
- `nccl-f32`: existing explicit F32 NCCL route for every payload; P2P AR is
  disabled for this process.
- `hybrid4m`: P2P64 remains enabled, with
  `GGML_META_NCCL_MIN_BYTES=4194304`. Payloads below 4 MiB must stay on P2P64;
  payloads at or above 4 MiB must use the existing NCCL F32 communicator.

The hybrid arm should route the 213,044-byte, 409,700-byte, repeated
213,044-byte, and zero-partial 213,044-byte cases through P2P64. Its
4,211,716-byte, 16,797,700-byte, and 33,579,012-byte cases should use NCCL F32.
The boundary is checked against actual output bytes, not nominal labels.
All three processes explicitly set `GGML_META_CONTEXT_VIEWS=1` so the test
retains the physical cohort enqueue ordering used by the production Meta3
context and makes route comparisons configuration-matched.

CUDA graph capture remains disabled in all three arms. The hybrid arm must emit
the V57 initialization marker with `selected=1 reason=initialized_nccl_f32`,
one actual-route marker per large call with the exact byte count and
`graph_merged=0`, and a summary reporting 18 large-route calls. The program
also checks that each small hybrid compute call launched three P2P ranks, each
large call launched zero P2P ranks, and the empty-rank canary remained on P2P.
The runner additionally checks the F32 wire-path reason and 18 selected F32
NCCL operations. No BF16 fallback or silent route fallback passes.

All payload outputs must match the same CPU oracle, each rank must agree, all
canaries must pass, and the output hashes must match across all three arms.
The harness fails closed if a marker is missing, an unexpected size routes to
NCCL, or any route counter differs from the expected count.

## Host-only build and oracle contract

Once the root freezes `runtime-perf-v57`, build/link and run only the CPU
geometry/oracle contract with:

```bash
cd /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/nccl-large-route-20261016
RUNTIME_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v57 ./build-host.sh
```

The `--cpu` path does not initialize CUDA or access a GPU. It is only a fixture
contract check and does not validate the new runtime route.

## GPU comparison gate

The runner refuses GPU execution without `NCCL_ROUTE_GPU_AUTH=ROOT_NCCL_ROUTE`.
It fixes `CUDA_VISIBLE_DEVICES` to the known GPU0–2 UUIDs, refuses existing
compute processes, requires at least 512 MiB free and temperature below 58°C,
uses an 18-second per-arm timeout, stores full per-arm logs, and compares the
three hash files. Do not run until V57 is frozen and root grants the GPU
window.

```bash
NCCL_ROUTE_GPU_AUTH=ROOT_NCCL_ROUTE \
RUNTIME_ROOT=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v57 \
./run-route-comparison.sh
```

The fixture tests a synthetic Meta3 collective and RMSNorm boundary. Even a
passing hybrid route proves only that this route selection works for these
shapes under the tested runtime. It does not establish CUDA Graph capture
safety, full-model correctness, or full-model speedup.
