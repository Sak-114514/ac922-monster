# SM70 packed dense MULMAT shape fixture results

## Current status

- Fixture source and bounded runner are prepared under this directory.
- V89 build ID: `20261016T101245.741848Z`; runtime: `runtime-perf-v89`.
- Host-link and source/runtime provenance checks are pending.
- No GPU run has been authorized or performed.
- No model was loaded and no engine source was changed.

## Scope

The fixture will test Q5_K T24/T32/T48/T64 plus the existing lower-shape
regressions, and all seven available original expert-170 raw quantized sample
types at T24/T32/T48/T64 as ordinary 2D dense `ggml_mul_mat` weights. Each
type runs separate OFF and ON processes using the same frozen V89 DSO and
deterministic F32 inputs. The required output gate is byte-exact raw F32 across
arms, with input readback, A/B/A replay, canary, and weight-integrity checks.

The sample bytes are reused without requantization or model scanning. Treating
one expert's raw tensor as an ordinary dense matrix does not reproduce the
original model layer or its expert-routing semantics. A fixture pass cannot
attribute the V38 model divergence to this path.
