# Native MTP D3 four-lane head fixture

This is a bounded real-head parity fixture for the root-owned default-off
`LLAMA_MTP_UNROLL_D3_LANES4=1` policy. It changes no engine source. Build and
GPU qualification are separate; do not launch the runner without the root GPU
authorization token.

## Scope and data

The fixture is adapted from the existing D3 two-lane and D2 four-lane head
fixtures. It uses the same model and production split callback. Metadata is
loaded with `no_alloc=true`; the fixture then reads exactly the 34 tensors
selected by `blk.48.*`, `token_embd.weight`, and `output.weight*`. It must keep
the routed block-48 gate/up/down tensors on EP axis 2 with 512 experts split
170/171/171 at offsets 0/170/341. Expert bytes are read through those tensor
descriptors. No trunk or PLE payload is read, and no expert payload is copied
or duplicated to synthesize more lanes.

The previous D3 head-only artifact measured the unchanged selected payload at
3,485,816,832 bytes total: 3,212,664,832 bytes on Meta and 273,152,000 bytes
for the host-resident shared input embedding. Its per-Meta-member GPU weights
were 1,078,665,312 / 1,110,608,992 / 1,110,610,752 bytes. Four-lane D3 does
not change this model subset. The fixture retains the 3 GiB per-member cap and
512 MiB free-memory reserve and rechecks live split sizes before reading any
payload.

The runtime context is one unified-KV Meta3 bank with `n_seq_max=4`,
`n_batch=24`, and `n_ubatch=24`. D3 x 4 reserves 16 planned target verification
rows (`4 * (3 draft + 1 seed)`); fixed-prefix input rows are 24 (`4 * 6`). The
head fixture does not execute target verification. At runtime, fixture input
embedding storage is `24 * n_embd_out * sizeof(float)` bytes per batch; full
logit capture is `3 * lanes * n_vocab * sizeof(float)` bytes per arm. The
actual `n_embd_out`, vocabulary size, device memory, and exact selected weight
sizes are read from the model metadata/runtime and printed in the log. No
full-model load or benchmark is part of this fixture.

## Cases

The four-lane prefix is deliberately ragged while retaining six rows per lane:

| Sequence | First position | Seed position | Real rows | Pad rows |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 4 | 5 | 1 |
| 1 | 3 | 5 | 3 | 3 |
| 2 | 1 | 3 | 3 | 3 |
| 3 | 1 | 4 | 4 | 2 |

Each lane uses its own token and hidden values. The matrix compares ordinary
three-step decoding with the actual unrolled path in one graph context in the
order four lanes -> two lanes -> four lanes. It checks every step's full
logits, token IDs, final hidden, final KV positions, and removal of future
prefix padding. It keeps the existing D3 max-absolute and RMS gates unchanged.

A separate three-lane `{0,2,3}` case asks the context policy to enable D3, and
requires that shape to be declined by the four-lane opt-in. It then aborts the
pending chain and compares the ordinary general path between the two contexts
for all three logits/hidden steps, token IDs, and KV positions. This verifies
the fixture's general-path fallback behavior; it is not a full
`common/speculative.cpp` server-dispatch test.

## Source, host build, and CPU checks

The source of truth is the AC922 tree at
`/home/debian/ac922-monster/experiments/standard-opt-20261014/src`. The build
script requires an explicit frozen runtime, links only the fixture executable,
and does not initialize CUDA or read a model. V69 was frozen as build
`20261016T031949.936341Z`; its `libllama.so.0` SHA-256 is
`4113f9ace2ce998f70bc375f3251fc056f7850ea893d8dad7786acc63f9986b6`.

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v69 \
  ./fixtures/mtp-d3-four-lane-20261016/build-head-d3-four-lane.sh
```

No engine build is part of this fixture task. The V69-linked fixture binary
SHA-256 is
`857b37c6ba675dce84ceb1af38920ae9dbdcac5077f6c2e6656c3f4a95afc332`.

The CPU-only check runs before `llama_backend_init`, reads no model, and uses
`llama_batch_init` for host allocations. It tests the V69 shared lane policy
table (D2/D4 compatibility, D3x2 variable-lane mode, D3x4 exact opt-in, and
three-lane decline) and compares the fixture's ragged prefix batch against the
shared fixed-prefix planner for 4 -> 2 -> 4 and a three-lane general-path
shape.

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v69 \
  ./fixtures/mtp-d3-four-lane-20261016/run-cpu-prefix-self-test.sh
```

The CPU table and prefix checks passed under tag `20261016T032826Z-124118`;
the unique log is `results/cpu-prefix-self-test-20261016T032826Z-124118.log`.
It reports 24 fixed input rows and 16 candidate verify rows for four lanes,
12/8 for two lanes, and three-lane D3 decline. The host build emitted only two
existing deprecation warnings for `llama_new_context_with_model`.

## Bounded GPU command (requires a separate root grant)

The runner refuses to start without the exact authorization token, refuses
existing GPU compute processes, targets only the granted GPU0-GPU2 UUIDs,
requires at least 512 MiB free and temperature below 58 C, and writes a unique
UTC-tagged log. It has a 180-second limit and performs no retry or GPU reset.

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
MTP_D3_FOUR_GPU_AUTH=ROOT_MTP_D3_FOUR \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v69 \
  ./fixtures/mtp-d3-four-lane-20261016/run-head-d3-four-lane.sh \
  /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf
```

The runner uses `LLAMA_EXPERT_PARALLEL=1` exactly and requires both the
explicit D3-lane-4 opt-in and the existing variable-lane opt-in for its
four -> two -> four matrix. Success requires actual engine `lanes=4 depth=3`
and `lanes=2 depth=3` active markers, the three EP axis-2 layout markers, all
three head parity cases, and the three-lane ordinary-path fallback case.

## Evidence boundary

The fixture is host-linked against V69 and its CPU policy/prefix checks passed.
It has not run on a GPU. Its 16-row verify count is a shape contract, not an
observed target verification dispatch. A future head-only pass cannot establish
target/GDN semantics, full-model quality, server fallback selection, or a
speedup.
