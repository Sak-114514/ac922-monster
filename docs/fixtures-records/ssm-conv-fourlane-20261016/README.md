# Qwen4Exp SSM_CONV four-lane fixture

This fixture isolates the Qwen linear-attention convolution-history assembly
and rollback snapshots, which are outside the standalone GDN-X4 operator
probe. It does not load a model or modify engine code.

Production dimensions come from the Qwen4Exp text config and
`qwen4exp::build_layer_attn_linear()`:

- `d_state=128`, 16 key heads, 48 value heads
- `conv_channels = 128*16*2 + 128*48 = 10240`
- `ssm_conv1d` kernel size 4; history width 3
- four sequences, four rollback state slots plus current (`n_rs_seq=4`),
  `mem_size=4`

The graph assembles the same state path as `build_conv_state_at()`:
`GET_ROWS(state_pool,state_rows) -> reshape [history,channels,sequence] ->
concat(history,transpose(x)) -> ggml_ssm_conv -> SiLU`, while copying all five
rollback tails into row-strided cache views. Initial state IDs are deliberately
nonmonotonic (`18,6,13,1`). Two per-sequence token lengths are covered: `T=1`
and the V35 target-verify `T=4`.

For each shape, the fixture runs one batched four-sequence graph and four
serial one-sequence graphs, then compares raw SSM_CONV (CPU), SiLU output, all
five snapshots per lane, the complete cache, and two untouched canary rows.
It carries slot-0 history into a second step and changes only lane 2's last
input token to verify that outputs/history change only for that lane. CPU raw
SSM_CONV is checked against an independent scalar float64 dot-product oracle
with a `2e-6` max-absolute gate. The CUDA build retains the normal
SSM_CONV→SiLU consumer path (allowing the runtime's fused implementation) and
requires exact batched-versus-serial output/cache equality.

Run the CPU-only check:

```sh
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v83 \
  ./run-ssm-conv-probe.sh cpu
```

CUDA is restricted to GPU0 and guarded by `SSM_CONV_GPU_GRANT=ROOT_GRANTED`;
do not set that token without the root task's explicit GPU0 window. The runner
checks free memory, temperature, and active compute processes immediately
before launching and times out after 60 seconds.
