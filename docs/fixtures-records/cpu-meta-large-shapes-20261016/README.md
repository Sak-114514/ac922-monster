# Nested Meta large CPU-copy fixture

Status: CPU helper tests and V80 host link/view checks pass. One full-size
bounded GPU run completed; all size, source-lifetime, and canary checks passed.
The cross-owner B copy was accepted asynchronously, so that run did not prove
a budget-full event. No engine source is modified by this fixture.

## Purpose and bounds

This extends the previously validated V77 CPU-to-nested-Meta2 fixture. It
retains the A/B/A sequence, event reuse, source overwrite/free immediately
after the copy API returns, exact F32 readback on all six leaves, and canary
checks. It keeps the process-wide pinned arena budget at 64 MiB and changes
only the copy-size policy to its existing 32 MiB ring ceiling.

The full-size run is preserved at
`results/gpu-v80-large-shapes-20261016T100423Z-201536/run.log`. It recorded A
with 7 async copies / 108 MiB, baseline B with 3 / 30 MiB, and the fresh
B-budget request with 1 / 16 MiB. Thus the B-budget request queued normally;
this was not an adapter error. The 16/20/32 MiB cases were async, 33 MiB used
the policy fallback, all 72 readbacks were exact with canaries intact, and all
12 sources were overwritten/freed after return. The first validator expected a
budget-full decline and marked that run failed; the outcome-aware validator
now accepts and reports the observed async B path.

The new single-run cases are:

- Contiguous mask `VIEW`: 16 MiB, F32 identity data, `[8192, 512, 1, 1]`.
- Contiguous MTP `RESHAPE`: 20 MiB, F32 identity data, `[2560, 16, 128]`.
- A contiguous 32 MiB transfer at the exact per-copy limit.
- A 33 MiB transfer that declines by policy and completes through the existing
  synchronous fallback.
- A fresh second Meta owner attempting 16 MiB after the size cases. The run
  reported whether the request queued or fell back; it was queued, so the
  cross-owner budget-full condition was not exercised on GPU.

The fixture grows owner A's slots only after the initial A/B/A 10 MiB tests;
the baseline B context is destroyed first, returning its arena. The first
full-size run accepted B's 16 MiB request asynchronously, so it did not prove
the aggregate budget was full. It still passed the size, source-lifetime, and
72 leaf-readback gates. The CPU-only helper suite separately covers a true
two-owner 64 MiB pinned-budget fallback.

## View/stride contract

The V80 callback checks the byte limit **before** it computes
`req.source_contiguous = ggml_is_contiguous(source)` and `same_layout`.
Therefore the saved 16 MiB `copy_policy_limit` line for
`attn_inp_kq_mask (view)` does not establish whether that particular view is
contiguous. The actual 16 MiB source and 20 MiB MTP source are only admitted if
the cap is raised and both the contiguity and same-layout checks pass.

The CPU-only `--cpu-view-check` creates metadata for both a contiguous mask
VIEW and a contiguous MTP RESHAPE and asserts their byte size/layout gates. It
also creates a genuinely strided, noncontiguous VIEW and runs it through the
same `cpu_meta_copy_candidate::upload_ring` eligibility policy; this must
decline with `source_not_contiguous`. A view operation is not automatically
rejected: a contiguous VIEW/RESHAPE can qualify. A noncontiguous view cannot
use this snapshot path, regardless of its byte size. Its fallback remains the
existing backend copy path and must pass the fixture's layout assertion and
exact readback; if the real V36 source view is actually noncontiguous, do not
count it as accepted by this fast path.

## CPU-only checks

Run the existing helper regression suite plus the new 16/20/32/33 MiB and
64 MiB cross-owner budget cases:

```sh
python3 fixtures/cpu-meta-large-shapes-20261016/run_cpu_tests.py
```

The host binary also has a metadata-only mode that avoids CUDA backend
registration and device initialization:

```sh
LD_LIBRARY_PATH="$RUNTIME_DIR/lib" \
  fixtures/cpu-meta-large-shapes-20261016/bin/cpu-meta-large-shapes-nested-fixture-v80 \
  --cpu-view-check
```

This is only a source-stride/layout gate; it does not qualify a GPU upload.

## V80 host compile/link

The host builder verifies the frozen V80 runtime manifest hash
`be87339463070c579d226cf13085c5a6b71a3f1eb60591c6178f9476f23a0b5b`, compiles
the nested fixture and event observer, and records hashes plus `ldd`. It does
not run the six-GPU transfer fixture.

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v80 \
  bash fixtures/cpu-meta-large-shapes-20261016/build-host-v80.sh
```

## GPU invocation — requires a later root grant

The wrapper requires `CPU_META_LARGE_GPU_AUTH=ROOT_CPU_META_LARGE`, checks the
six expected UUIDs, free memory and temperature, refuses any existing GPU
compute process, and has a 90-second hard timeout. It does not reset devices.
It ran once; root directed that no additional GPU attempt is needed.

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
CPU_META_LARGE_GPU_AUTH=ROOT_CPU_META_LARGE \
RUNTIME_DIR=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v80 \
  bash fixtures/cpu-meta-large-shapes-20261016/run-gpu-once.sh
```

The full-run validator requires the original A/B/A event reuse, accepted
16/20/32 MiB paths, exact 33 MiB policy fallback, the actual cross-owner
outcome, 72 exact/canary leaf readbacks, and 12 source mutation/free checks.
The event observer reports nested Meta event create/record/query status. The
test is a bounded adapter/lifetime qualification, not a model or speed result.

The budget-only cross-owner fallback remains a CPU-only helper test. Root
closed further GPU work after the bounded size/lifetime run, so this fixture
does not include a second GPU budget-pressure attempt.
