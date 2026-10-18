# V35 one-bank/four-lane host preparation

V35 is a new short-context candidate combining the V77 nested graph reuse path
with the V80 standard QSA sparse/head-loop, narrow GET_ROWS, fused T16, and
four-lane D3 target-padding paths. It removes the second target/draft context
bank used by V32/V34, while retaining the same four independent 16K KV streams.
No model-quality result is implied until the full V35 gates complete.

## Candidate shape

- One target context bank and one draft context; four lanes in bank 0.
- Non-unified KV: `ROOT_BANK_CTX=65536`, four streams of 16,384 cells each.
  The aggregate is 65,536 cells; each request remains limited to 16,384.
- D3 MTP, fixed prefix six rows per lane (24 total), target verify budget 16.
- Target ubatch 1,024; draft ubatch 512; full 8K prefill batch retained.
- Existing hierarchical EP6 and CPU PLE layout retained.
- Standard QSA sparse is explicitly requested. The previous helper GPU fixture
  covered stream counts 1 and 2; it did not cover 4. The full-model log gate
  therefore requires `requested=1 selected=1 streams=4` before this feature can
  be reported active for the new four-lane shape.
- The V80 QSA head-loop must log a selected four-stream shape. Narrow GET_ROWS
  and EP fused T16 must also show selected runtime dispatch markers; static env
  values alone do not count.
- Per-active-lane D3 target padding must report four active lanes and exactly
  16 rows (`real_rows + pad_rows = total_rows = output_rows = row_budget`).
- Meta CPU async-copy cap is 10 MiB; its existing process pinned budget remains
  64 MiB. Nested compose, parent-walk, and parent-manifest cache stay enabled.
- Nsight and CUPTI are off.

## Quality and evidence boundaries

The runner carries forward the native semantic gate, full-F32 preclean gate,
same-candidate serial/P4 token-array comparison, original auxiliary prefix
checks, clean context family, and V27 generated-array comparison. After clean
four-request P4 generation, the same clean family runs one uncached 8K request
with a 32-token output and no F32 dump, spans, Nsight, or CUPTI. Its target,
MTP, full-prefill-wall, and whole-scenario-wall timings are reported separately
from aggregate P4 rates. Its 32 generated IDs must exactly match the same-run
diagnostic single-lane probe using the identical prompt. It does not
remove or weaken the known failures. In particular, the earlier V30b one-bank
four-lane experiment failed serial/P4 token parity for two flows and both
auxiliary prefixes; those results are retained in
[V30B-DIVERGENCE-TRACE.md](V30B-DIVERGENCE-TRACE.md). V35 is a new combination
that must pass the same gates before any quality or speed claim.

The four-lane standard-QSA selector and D3 padding activation are checked from
runtime log markers, not inferred from environment variables. The comparator
requires a selected four-stream QSA marker and a complete four-lane, 16-row
padding marker. Existing per-card VRAM and temperature guards remain the
authority for any later model run; the static estimate in
[CAPACITY-LEDGER.md](CAPACITY-LEDGER.md) is not a fit guarantee.

## Host-only checks

On AC922, after syncing this fixture directory:

```sh
cd /home/debian/ac922-monster/experiments/standard-opt-20261014
./fixtures/dual-context-v35-onebank4-combined-20261016/run-cpu-tests-v35.sh
./fixtures/dual-context-v35-onebank4-combined-20261016/build-v35-onebank4-runner.sh
```

The build script links the harness against the frozen V80 library closure. It
does not load model weights, initialize CUDA, or launch GPU work. A later model
run must be started through the root-owned live safety supervisor and only
after explicit GPU-window authorization; do not invoke the low-level launcher
directly.
