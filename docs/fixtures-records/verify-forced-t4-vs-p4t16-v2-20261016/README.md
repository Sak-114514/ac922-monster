# Forced target verify V2: recurrent state and native HC tap evidence

V2 extends the V1 target-only T4/P4 logits capture. It uses one shared model
load and one V35-shaped target context, still with no draft context or MTP
proposal/acceptance/rollback. V1 data remains unchanged in its own fixture.

## Additional observations

1. After each P4 prompt prefill (sequence order 0, 1, 2, 3), export every live
   sequence's recurrent-only state using `llama_state_seq_get_size_ext` and
   `llama_state_seq_get_data_ext` with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`.
   Save each payload as a binary snapshot and record its size/sequence/stage in
   `state-snapshots/index.jsonl`. The postprocessor hashes each snapshot and
   compares earlier live sequences before/after later prompt prefills. It also
   compares each standalone T4 recurrent state with that flow's P4 state after
   its prompt.
2. Before backend/model/context initialization, set
   `LLAMA_DFLASH_NATIVE_HC_TAPS=1`. After all prompts are prefetched and before
   the forced target verify decode, enable only layers 0–3 with
   `llama_set_embeddings_layer_inp`; read each tap through the public
   `llama_get_embeddings_layer_inp`; then disable all four taps before another
   prefill can run. Each tap is the contracted HC attention-block input
   boundary, not the final layer output.

The model's actual hidden width is 2,560. The existing `output_reserve` code
allocates one native tap plane per enabled layer at `n_batch` width, so four
layers reserve `4 × 8,192 × 2,560 × 4 = 335,544,320` bytes (320 MiB). This is
recorded in `run.json`; target `n_batch=8192` is unchanged. Tap payload files
only contain rows used by the forced verify batch, while the reserve is sized
to the full batch as in the public API implementation.

Partial recurrent snapshots omit attention KV and QSA indexer cache. The
per-sequence state writer normalizes its sequence metadata to `n_seq_id=0` for
sequence-specific exports. V2 does not restore these snapshots or mutate
engine state; it only reads and persists them for comparison.

## Build and analysis

Host-only checks and link use the frozen V80 closure:

```sh
python3 fixtures/verify-forced-t4-vs-p4t16-v2-20261016/test_host_only.py
python3 fixtures/verify-forced-t4-vs-p4t16-v2-20261016/test_schedule_source.py
fixtures/verify-forced-t4-vs-p4t16-v2-20261016/build-forced-verify-v2-v80.sh
```

No model/GPU launch is included here. A separate root authorization after the
current kernel fixture gates is required; launch must use the root live safety
monitor and the original V80 environment plus `LLAMA_DFLASH_NATIVE_HC_TAPS=1`.
After a completed run, generate the V1 logits comparison first, then analyze
both the native HC taps and recurrent snapshots:

```sh
python3 fixtures/verify-forced-t4-vs-p4t16-v2-20261016/analyze_forced_v2.py \
  /path/to/v2-run-output --output /path/to/v2-run-output/v2-analysis.json
```

The analysis identifies the first tapped layer with a T4/P4 difference for
each forced row and reports whether recurrent state for an existing P4 sequence
changed when later prompts were prefetched. All results remain diagnostic and
do not replace the original semantic, F32, or generation-parity gates.
