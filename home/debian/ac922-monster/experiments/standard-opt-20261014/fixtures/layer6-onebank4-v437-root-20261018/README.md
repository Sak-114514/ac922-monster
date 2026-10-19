# V437 onebank four-stream D3 candidate

V437 keeps the R409 runtime and the V434 canary protocol. Its native prompt
reader records each requested output's **original batch row**, checks the
`batch.logits` mask at that row, and passes that row to `llama_get_argmax_ith` /
`llama_get_logits_ith`. Compact output ordinals are not valid row IDs for these
APIs. The frozen native prompt lengths map to rows `31, 66, 107, 154` in the
155-row batch.

## Canary evidence

By default, the runner computes the 112-comparison target-state canary. The
explicit `ROOT_V437_REUSE_V434_CANARY=1` mode can reuse only the pinned V434
canary JSON. It verifies the JSON SHA-256, exact 112 finite rows and geometry,
quality-unqualified status, same real model/corpus, R409 build report and
runtime directory, and the matching V434 source closure/run manifest/profile/
host-link receipt. Reused evidence is written to the new result with its reference path
and hashes and `canary_computation_performed_this_run=false`; it is not reported
as a new canary run. This mode also requires
`ROOT_V437_DIAGNOSTIC_PERFORMANCE_QUALITY_UNRESOLVED=1`. It never promotes the
result to quality-qualified.

The candidate remains preparation-only until the current source has been
host-linked, the reused canary has been reviewed, and the exact native4 gate
passes. Only then may the root control file authorize the single D3 4x16K/512
wave. No D2/D1 matrix is included. The saved V434 canary is diagnostic evidence
only; its numerical differences remain unresolved.

## CPU-only checks

```bash
python3 verify-v437-static.py
python3 test-v437-output-index.py
bash -n root-host-link-v437.sh
```

These checks validate profile/source pins, closure members, reference evidence
metadata, output-row arithmetic, and the opt-in reuse gate. They do not compile
the runner, load model weights, run a native gate, or touch a GPU.
