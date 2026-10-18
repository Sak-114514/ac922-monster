# D3 non-unified KV head fixture

This is a private, head-only copy of the validated D3 fixture with the context
configuration changed to `kv_unified=false` and `n_seq_max=2`. It loads the same
34 MTP/head tensors only when its GPU runner is explicitly launched. The trunk,
PLE payloads, and full-model scan remain out of scope.

## Static R54 audit

Source of truth inspected on AC922:

`/home/debian/ac922-monster/experiments/standard-opt-20261014/src`

No fixed-prefix or D3 unrolled gate forces a unified KV cache:

- `src/llama-context.cpp::llama_context::enable_mtp_gpu_chain_multi_unrolled()`
  checks depth/lane shape and other MTP prerequisites; it has no `kv_unified`
  condition. The fixed-prefix validator and `process_ubatch_mtp_unrolled()` also
  have no unified-cache guard.
- `llama_mtp_make_graph_step()` and `llama_mtp_snapshot_graph_inputs()` set
  `n_streams = cparams.kv_unified ? 1 : ubatch.n_seqs_unq`. With two unique
  sequence IDs and `kv_unified=false`, each step snapshots two streams.
- `src/llama-graph.cpp` shapes attention masks using the same unified/non-unified
  stream rule. The Qwen4exp unrolled builder iterates the per-step memory context
  and input snapshot and has no `kv_unified` assertion.
- `begin_mtp_gpu_chain_multi()` validates sequence IDs against `n_seq_max`; IDs
  `0` and `1` are valid with `n_seq_max=2`. Fixed-prefix rollback removes future
  positions per sequence ID.
- `src/llama-mtp-lane-policy.h::supports_lanes()` admits four lanes by default,
  and two or three only with the variable-lane opt-in; it never admits one lane.
  D3 has the stricter two-lane shape. The single-sequence D2 unrolled rejection
  is therefore the existing lane policy, unrelated to unified/non-unified KV.

The previous full-matrix log did not record event counts at the failing check,
so its actual `logits_events`/`hidden_events` sizes are not claimed here. Source
does confirm the fixture's count assumption: non-unified `llama_memory_hybrid::init_batch`
calls `split_equal(..., sequential=true)`, which only admits increasing adjacent
sequence IDs. The lane-major `{1,0}` prefix therefore forms two UBs; `llama_context::decode`
processes each UB separately, so the output callback may observe one output pair
per UB. The copied fixture now prints the reference event counts before applying
the unchanged one-event assertion.

The code copy asserts `kv_unified=false`, `n_seq_max=2`, and the per-step stream
count implied by the actual lane IDs. For D3, each of the three steps checks two
unique lanes and logs `n_streams=2`. A separate stepwise D2 reference check uses
only sequence ID `1` and logs one stream. The existing lane policy rejects its
unrolled D2 arm, so it makes no single-sequence unrolled parity claim.
The D3 cases compare the existing stepwise MTP head path against fixed-prefix
unrolled D3 for sequence order `{0,1}`, reordered lanes `{1,0}`, then same-context
reuse `{0,1}`. They check all three full-logit steps, token history, final hidden
state, final KV positions, and future-prefix rollback. The single-sequence D2
check compares both logits steps, history, positions, and rollback. These remain
head-only MTP checks and do not validate target-model GDN/cache semantics or
target verification.

The source hashes recorded when the R54 audit was made:

- `src/llama-context.cpp`: `b5d5577fa12e8c80a272c3e7df6eacfa894bf2a5fdc82742aaf8ce5709d5f8e1`
- `common/speculative.cpp`: `be7bbf090e88160c7d444cc44089dc2fe0e64959311676cd9c0077164a6e4abc`
- `common/mtp-fixed-prefix-plan.h`: `7d993f5e8a805ba5c9118a7d829f8958b228cb1e21a6fcda98eb54f81a91eaa5`
- `src/models/qwen4exp.cpp`: `4ebea3504de3747dd83f1401f282710d3fe1e8cdb1845debf0ef4d07cb6de0b5`
- `src/llama-mtp-unrolled-layout.h`: `455f60d69cd091167a1cd6239606bf0a6d7099be4a2d4e833f87308d9a91747c`
- `src/llama-mtp-lane-policy.h`: `96750fe15058f225e30756ecfc5eec44f1e28dac3644efd51f015feb5bf28e81`

## Host build and CPU regression

Host-link against the frozen R54 runtime on AC922:

```bash
cd /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/mtp-d3-nonunified-head-20261016
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v54 ./build-head-d3-nonunified-fixture.sh
```

The D3 planner/layout CPU regressions are reused from the existing fixture:

```bash
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
./fixtures/mtp-d3-prep-20261016/run-cpu-regressions.sh
```

Neither command launches a GPU or reads model payloads.

For the two selected checks, the runner accepts:

```bash
D3_HEAD_CASES="canonical-reuse seq1-only" RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v54 ./run-head-d3-nonunified.sh
```

The binary also accepts the explicit options `--case canonical-reuse` and
`--case seq1-only`. With no `--case` option, the original `{0,1}->{1,0}->{0,1}`
matrix remains intact and retains its known lane-rebind failure.

## GPU run only after an explicit root window

After root confirms an exclusive GPU window and process availability:

```bash
cd /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/mtp-d3-nonunified-head-20261016
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v54 ./run-head-d3-nonunified.sh
```

The runner pins this authorized window's GPU0-2 UUIDs, checks free memory and
temperature, rejects existing compute processes, uses a 50-second timeout,
loads only the selected 34 tensors, and writes a new UTC result tag.

## First authorized GPU attempt

Against `runtime-perf-v54`, run tag `20261015T221045Z-62166` loaded exactly 34
head tensors, with no trunk or PLE payload reads. The first `{0,1}` D3 case
matched the stepwise reference at all three logits steps, token history, final
hidden state, final KV positions, and future-pad rollback (`max_logit_abs=0`,
`max_final_hidden_abs=4.76837e-7`). Captured attention-mask tensors had a final
dimension of 2, including `{256,1,1,2}` and `{256,12,1,2}`.

The next `{1,0}` lane-rebind case stopped at the reference arm with
`ordinary D3 step 0 did not expose one full logits and hidden event`. The final
reuse case and single-sequence-1 snapshot check were not reached. This run is a
failure, not a non-unified D3 parity qualification; the failure log is retained
at [run.log](results/mtp-d3-nonunified-context-matrix-20261015T221045Z-62166.log)
with SHA-256 `2c86addc36a3251c08a982318068b381bdeb1a0faf650a487ef687bf90e2113e`.
The post-run check found no compute processes on GPU0-2; utilization was 0%,
memory use 0/0/1 MiB, and temperatures were 38-39 C. No retry or reset was run.

## Selected-case run after adding count diagnostics

The selected runner executed `canonical-reuse` and `seq1-only` in one R54 head-only
process. Both canonical `{0,1}` D3 calls passed with exact full-logit/token parity,
final hidden and KV positions, and rollback. The new reference diagnostics logged
one logits and one hidden event for each canonical stepwise call; graph attention
mask metadata had `ne[3]=2`.

The seq1-only stepwise reference logged one logits and one hidden event and the
fixture checked/logged one stream for each D2 step. The D2 unrolled arm was
rejected because `supports_lanes()` does not admit one lane; this is the expected
existing policy and is unrelated to the KV layout. No single-sequence unrolled
numeric parity is claimed. The earlier `{1,0}` count remains unobserved: its preserved log
does not contain event-vector sizes, so the two-UB source explanation is not
reported as an observed callback count.

- Selected result: [run log](results/mtp-d3-nonunified-context-matrix-20261015T221708Z-62885.log), SHA-256 `c608c94a59f3c64c5955b58b5b80671cd33a2cfd2908e3bac8afe74171554959`
- CLI argument mismatch before GPU work: [run log](results/mtp-d3-nonunified-context-matrix-20261015T221549Z-62674.log), SHA-256 `7b43f3a83960b4eb70221d3d27ed22fa5f01bf35394e7708fd844e5fa2cfbc45`

The selected run released GPU0-2 with no compute processes, 1 MiB used per GPU,
0% utilization, and 38-39 C. No retry or GPU reset was run.
