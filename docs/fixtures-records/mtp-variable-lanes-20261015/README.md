# Variable-lane real KV allocator fixture

This CPU-only fixture reuses the D4 real `llama_kv_cache` allocation, index, and mask checks from `mtp-fixed-prefix-pad-20261015`, with lane count selected at runtime (`2`, `3`, or `4`). It loads the real model vocabulary in vocab-only mode and filters every layer before KV tensor allocation; it does not load model weights or execute model layers.

Each lane gets six step-0 rows (prefix, seed, and future dummy), followed by three accepted rows. Across 100 rounds, the exact used-cell total is `100 * (6 + 2) * lanes`: step 0 reserves six cells per lane, rolls the one dummy back, and then three accepted rows commit, leaving eight cells per lane per round. The cache capacity remains 4096 cells, enough for the four-lane case (3200 cells).

The fixture retains the original D4 assertions: unique in-range step-0 indices; exact seed-only output rows; causal visibility of seed and dummy; dummy rollback; first accepted row reuses only the freed dummy cell; later accepted rows do not overwrite prefix/seed or earlier accepted rows; positions remain contiguous; allocator usage stays contiguous; and step-0 indices, output flags, and mask remain unchanged through accepted steps 1–3. Only lane-dependent row counts and expected totals are parameterized.

Build on AC922 against the frozen V32 runtime, then run the three lane counts once:

```bash
bash fixtures/mtp-variable-lanes-20261015/build.sh
bash fixtures/mtp-variable-lanes-20261015/run-matrix.sh
```

`run-matrix.sh` hides CUDA, sets `OMP_NUM_THREADS=2`, and writes one log per lane count under `logs/`. This checks real KV allocator metadata and CPU-generated masks only. It does not establish model-output parity, MTP graph correctness, or GPU runtime behavior.
