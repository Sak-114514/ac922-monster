# V434 production-rollback onebank D3 preparation

V434 is a corrected copy of the V430 onebank candidate. It keeps one loaded model, one target context, one MTP draft context, four non-unified KV streams of 17,152 cells each, generic D3, and a single D3 4×16K/512 wave. The V430 directory and its results remain untouched.

## Canary protocol

Each real-corpus lane supplies a 1,024-token prefix, then four teacher-forced tokens representing the production seed plus up to three draft tokens, plus one token for the next teacher step. For accepted draft count `k` in 0–3, the production state retains the seed and `k` accepted drafts; the cache removal boundary and next teacher position are both `prefix + 1 + k`. This keeps the seed when `k=0`.

The serial target context uses UBatch 64. The four-sequence target context uses UBatch 256 total (64 rows per sequence). The canary records 56 core serial/multisequence/rollback comparisons, 40 same-input reset/replay F32 comparisons, and 16 post-rollback versus fresh next-token replay comparisons. Same-input replay rows are checked bitwise; raw F32 rows and numeric differences are saved. The replay comparison is separate so reset drift is not mistaken for a sequence-removal error.

By default, any nonfinite value, incomplete comparison set, invalid context shape, numeric error above the existing max-absolute 0.5 / relative-L2 0.05 thresholds, or non-bitwise same-input repeat fails closed before benchmark contexts are allocated. The optional `ROOT_V434_DIAGNOSTIC_PERFORMANCE_QUALITY_UNRESOLVED=1` path only allows an all-finite, complete, geometry-valid report to reach the root-control wait; its status remains `unresolved_diagnostic`, and all later result rows carry `quality_qualified=false`. The profile leaves this mode off. The root must review `canary.json` and its F32 snapshots before writing the fresh `root-control.json`.

The native gate verifies the actual result uses bank 0, sequence IDs 0–3, four target/draft KV streams, and total verify-row budget 16. It adapts a private copy of those verified sequence IDs for the frozen V146 four-bank checker, then keeps the actual onebank topology in the result file.

Prefill uses one owner and synchronized 2,048-token cohorts, with a ten-second cooldown after each cohort. Prefill metrics include those waits; no overlap claim is made. The scenario contains exactly one D3 native gate and one performance wave. No D2/D1 matrix or baseline rerun is included.

## Preparation status

V434 is not built or run. It makes no claim about target-state correctness, native answer quality, or speed. The R409 host-link script is prepared but gated, and the root profile remains launch-disabled.

## Offline checks

```bash
python3 verify-v434-static.py
python3 test-v434-rollback-protocol.py
bash -n root-host-link-v434.sh
```

These checks cover source/profile contracts and CPU-only index arithmetic. They do not compile llama.cpp, load model weights, or touch a GPU.
