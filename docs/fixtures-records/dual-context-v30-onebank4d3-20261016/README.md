# V30 one-bank four-lane D3 host preparation

V30 is the hierarchical EP6 capacity-qualification profile for comparing one
four-lane bank with the existing two-bank/two-lane layout. It uses one
non-unified target context and one draft context, each with four independent
16K sequence streams. This directory contains an isolated copy of the V28
short-context harness adapted to that topology; no engine or private core
source was changed.

## Frozen configuration

- V71 whole runtime closure for V30b, build ID `20261016T040204.430535Z`, with `GGML_META_FRAGMENT_PARENT_SIG=1`.
- `ROOT_BANK_LANES=4`, `ROOT_BANK_CTX=65536`, `ROOT_KV_UNIFIED=0`,
  `ROOT_KV_VARIANT=per_seq_16k_x4`.
- D3 MTP, fixed prefix 6 rows/lane, `LLAMA_MTP_UNROLL_D3_LANES4=1`, existing
  `LLAMA_MTP_UNROLL_VARIABLE_LANES=1`, target ubatch 1024, draft ubatch 512.
- Hierarchical EP6 / nested GPU all-reduce / NUMA PLE is declared explicitly
  in the runtime log and manifest. This candidate is a 16K-per-request
  capacity qualification, not a full-context serving claim.
- Four P4 flows must appear once each, all assigned to bank 0. The comparator
  rejects duplicate or missing flow IDs and any flow routed to another bank.

The V71 runtime policy declines D3 unroll at three lanes. V30 keeps that
fallback behavior; it only opts in to the four-lane D3 shape. The candidate
does not force a three-lane request through a four-lane graph.

## Preserved quality gates

The V28 diagnostic/clean context-family sequence is retained. The diagnostic
family runs the original native semantic gate, serial/P4 F32 preclean gate,
all exact serial/P4 generation comparisons, and the original cold/warm 32-token
auxiliary prefix checks. The clean four-request generation family is created
only after the native and F32 preclean gates pass. The existing V27 auxiliary
prefix failures stay visible in the original comparator; this harness does
not edit or suppress that check.

The P4 generation scenario uses four real 8K prompts and requires 256 output
tokens per flow. Actual target/draft context totals, per-sequence sizes,
sequence counts, KV stream counts, D3 verify rows (16), and prefix rows (24)
are reported from the initialized contexts. The post-run comparator also
checks the exact flow-to-bank mapping.

## Static capacity ledger

See [CAPACITY-LEDGER.md](CAPACITY-LEDGER.md). It derives a roughly 2.98 GiB/card
tracked-buffer estimate from the saved V28/V23 records and leaves the actual
V71 reserve/context-init gates authoritative. The four-lane target and draft
graph allocations are not measured yet.

## Host-only checks

From the AC922 experiment root:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
./fixtures/dual-context-v30-onebank4d3-20261016/run-cpu-tests-v30.sh
ROOT_DYNAMIC_LIB_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v71/lib \
ROOT_BUILD_REPORT=/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V71-build.json \
  ./fixtures/dual-context-v30-onebank4d3-20261016/compile-runtime-v30.sh
```

The V30b build links only the harness executable against V71. It does not load the
model or initialize CUDA. The V71 build ID is `20261016T040204.430535Z`; the
host runner SHA-256 is
`9f114e7101e909a3d30d1bb7d7c0823160b25e55520294b9dd30b51e3449b24c`. Its
`ldd` output resolves the application libraries from the V71 closure.

`run-cpu-tests-v30.sh` validates D2/D4 compatibility, the exact D3x4 opt-in,
three-lane D3 decline, the shared six-row fixed-prefix planner, the 24/16 row
shape, four unique bank-0 routes, diagnostic/clean plan generation, and
presence of both original 32-token auxiliary prefix checks. Latest saved CPU
log: `results/cpu-tests-v30-20261016T042748Z-140719.log`. These are plan/policy
checks only; they do not run generation or load a model. The V30b CPU regression
also invokes the exact verify-budget helper used by the real fixture runner,
accepting 16 rows for four-lane D3 and rejecting 17 while preserving the
legacy two-lane D3 limit of 8 rows. The first full-model attempt exposed the
fixture's stale 8-row literal; see [STATUS.md](STATUS.md) for the preserved
failure and the corrected runner evidence.

## Future root-supervised launch

The V30b one-shot has completed; see [STATUS.md](STATUS.md) and its retained token/timing reports. Any further model run requires a new root GPU authorization. For reproduction after such authorization, the supervisor can invoke:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
V30_V29_QUALIFICATION_PASSED=1 V30_GPU_WINDOW_AUTHORIZED=1 \
GGML_META_FRAGMENT_PARENT_SIG=1 ROOT_RUNTIME_REVISION=20261016T040204.430535Z \
ROOT_DYNAMIC_LIB_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v71/lib \
ROOT_BUILD_REPORT=/home/debian/ac922-monster/experiments/standard-opt-20261014/notes/V71-build.json \
  ./fixtures/dual-context-v30-onebank4d3-20261016/run-v30-onebank4d3.sh \
  ./fixtures/dual-context-v30-onebank4d3-20261016/build-runtime-v30/dual-context-v30-onebank4d3 \
  /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf \
  /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/dual-context-v30-onebank4d3-20261016/results/root-authorized-run
```

The supervisor must provide the hierarchical EP6 hardware environment from
`v30-onebank4d3-exact-env.json` and a fresh GPU/BMC safety gate. The launcher
requires the reviewed V29 prerequisites and root-authorization tokens. The
V69 first attempt loaded weights once and stopped at the stale 8-row fixture
guard before quality scenarios; it is retained in [STATUS.md](STATUS.md).
V30b loaded the model once and completed; it is not certified because two P4 token arrays differ from the same candidate serial reference and the original auxiliary-prefix gates failed. The full output and timing evidence is recorded in [STATUS.md](STATUS.md).
