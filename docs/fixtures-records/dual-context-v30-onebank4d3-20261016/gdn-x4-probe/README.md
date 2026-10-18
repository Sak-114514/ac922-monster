# GDN X4 batched-versus-serial operator probe

This fixture isolates the recurrent GDN operator behind the V30b four-lane
token-parity failure. It is not a full-trunk or model-quality test.

The batch arm executes one F32 `GGML_OP_GATED_DELTA_NET` with width 128, four
heads, four tokens per sequence, and four independent sequences. The serial arm
executes the same op four times with the same per-sequence Q/K/V/gate/beta and
initial state bytes. Initial states are gathered from a shared 20-row pool
using non-monotonic row IDs `18,6,13,1`. The V30b run reports `mem_size=4`
and `n_rs_seq=4`, so this matches the actual `mem_size * (1+n_rs_seq)` state
pool and can exercise deferred state-gather mapping when selected.

Both arms use `K=5` snapshot slots and the production recurrent-cache copy
shape from `llm_build_delta_net_base::build_recurrent_attn`: `[D,n_seqs,4]`,
contiguous sequence rows, and rollback slots separated by `mem_size * row_size`
with the V30b `mem_size=4`. The full cache has 20 rows (`4 * (1+4)`). The
serial arms write each sequence into its same physical cache row. The fixture
compares all attention rows, all four written state
snapshots, accepted-length rollback selection (`accepted=0..3` maps to
`slot=3..0`), and untouched cache canaries. This tests SSM state snapshot
layout/copying; it does not test the separate Qwen convolution-history cache,
the surrounding trunk, target logits, or MTP acceptance logic.

At T=4, the V71 X4 dispatch predicate applies to both the four-sequence batch
and one-sequence serial arms: env opt-in, scalar gate, width 128, Volta/Turing,
and `n_tokens > 1`. There is no `n_seqs == 4` predicate. A CUDA run must report
the actual `volta_gdn_x4` and `gdn_state_gather` selected summaries for all five
GDN executions; if not, it is not an X4/deferred-gather qualification.

## CPU oracle

Host-link and run the CPU backend only:

```sh
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v71 \
  ./run-gdn-x4-probe.sh cpu
```

This links directly to `libggml-cpu` and does not initialize the CUDA backend.
The CPU gate is strict bitwise equality for the batch-versus-four-serial
attention outputs and rollback cache; it verifies the test's input/row/cache
mapping before any GPU window.

V71 CPU run passed: all 8,192 attention floats, 1,048,576 rollback-state
floats, and 1,310,720 complete cache floats (including canary rows) were
bitwise exact. The four accepted-length slot selections per sequence also
passed. The exact production-stride run is
`results/cpu-oracle-prodstride-20261016T044837Z.log` (SHA256
`f7d42ad50206875488ac7c3dcd625f8de3bda0866c1e5baae5b01ccab06408b4`). The
earlier eight-row exploratory run is retained but superseded; it is not a V30
layout proof.
The CPU binary linked directly to `runtime-perf-v71/lib` and loaded only
`libggml-cpu.so.0` plus `libggml-base.so.0`; no CUDA backend was initialized.

## CUDA operator qualification

```sh
GDN_X4_GPU_GRANT=ROOT_GRANTED \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v71 \
  ./run-gdn-x4-probe.sh cuda0
```

The runner requires the explicit grant token, enables `GGML_CUDA_VOLTA_GDN_X4=1`
and `GGML_OPTIMIZATION_PATH_LOG=1`, clears the gather-disable override, then
relies on V71's dispatch summaries. It ran once on GPU0 under a bounded root
grant. Both dispatch paths were selected in all five calls; all attention,
snapshots, cache/canary, and rollback gates passed bitwise. The log is
`results/gdn-x4-v71-gpu0-20261016T045900Z.log` (SHA256
`4bfcaae541986ab7c1eba836190f54ace698526f0e25e2ac7ae71e848b8bead1`). GPU0
was confirmed idle immediately after the run. This is operator evidence only,
not full-trunk or model token parity. Do not repeat it without another root
GPU grant.

`results/cpu-debug-20261016T044505Z.log` preserves the first host attempt. It
linked against the old `/usr/local/lib` GGML instead of V71 and aborted on the
old output layout; the build/run scripts now link to and load the versioned
libraries under `runtime-perf-v71/lib`. It is not evidence of an operator
failure.
