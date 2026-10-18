# V29 preparation status

- State: prepared, awaiting V68 nested-async hot-frame qualification and root GPU grant.
- Candidate: V68 closure + `GGML_META_NESTED_ASYNC=1` + inherited V28 fragment cache (CAP64/MIB128).
- Runner: byte-for-byte reuse of the already qualified V28 binary; no runner/core rebuild.
- Scope: 16K per sequence, two banks × two lanes, D3, target UB1024, MTP UB512, all prior quality/clean/exact-V27 gates retained.
- Required output: direct step-compute samples in ms and clean P4 metrics, with no cold-probe throughput claim.
- No GPU/model launch performed by this preparation.
