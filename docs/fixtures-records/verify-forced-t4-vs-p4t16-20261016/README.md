# Forced target T4 versus P4/T16 logits capture

This is a bounded target-only diagnostic for the V35 serial/P4 token divergence.
It uses one load of the same integrated IQ3_XXS model and one target context. It
does not create a draft context or call MTP proposal, speculative processing,
acceptance, or rollback. Existing semantic, F32, token-parity, and auxiliary
prefix gates remain unchanged and this fixture cannot waive them.

## Compared arms

- **T4, serial per flow:** clear target memory, prefill one V35 8K prompt into
  its original sequence id, then teacher-force its first four serial-reference
  output tokens in a four-row target decode. Repeat for the four flow ids.
- **P4/T16:** clear the same target memory, prefill those same four prompts into
  sequence ids 0–3, then teacher-force the exact same 16 tokens in one
  lane-major target decode.

Both arms use the V35 target configuration: six-card 3+3 split, non-unified
Q8_0 KV, 65,536 aggregate cells / 16,384 cells per sequence, four sequence
slots, `n_rs_seq=4`, next-token embedding outputs enabled, 8,192-token prompt
batch, target ubatch 1,024, flash attention on, and `LLAMA_QSA_INDEXER_IMPL=standard`.
The schedule is pinned in [target-forced-schedule.json](target-forced-schedule.json)
and generated from V35's serial output prefixes. Its V35 result source SHA256 is
`d2f573ddf68bd080a732513cda4a4d80501cefd4e04f60457f51207429e2798a`; the
schedule SHA256 is `e2b03fb2e5b0fd5ba53b15197bd4b888a66abc2688fd76baeddb172bd61015c5`.

The capture reads all four full-vocabulary F32 target logits rows immediately
after successful target decode and synchronization. Each row stores the input
sequence/position/token, complete prompt prefix, forced token suffix, GPU
argmax, GPU argmax-unique flag, unsuppressed CPU raw argmax, CPU runner-up,
top-two margin, finite/non-finite counts, and payload CRC. Raw logits remain the
authoritative source for max-absolute and RMS comparison.

The GPU argmax-unique field is meaningful only with both `LLAMA_GPU_ARGMAX=1`
and `LLAMA_GPU_ARGMAX_UNIQUE=1`; the runner requires those opt-ins. The raw CPU
argmax deliberately does not apply EOS or suppression filters.

## Host-only verification

On AC922, sync this fixture directory and use only the host build step:

```sh
python3 fixtures/verify-forced-t4-vs-p4t16-20261016/test_host_only.py
python3 fixtures/verify-forced-t4-vs-p4t16-20261016/test_schedule_source.py
fixtures/verify-forced-t4-vs-p4t16-20261016/build-forced-verify-v80.sh
```

The build script verifies the frozen V80 runtime report and links against that
whole library closure. It does not initialize the backend, load model weights,
create a CUDA context, or execute on a GPU. The `forced-verify-capture` binary
must only be launched by the root-owned live safety supervisor after a separate
GPU-window grant; this fixture intentionally contains no direct launch script.

After a supervisor-authorized run, compare the four serial captures with the
corresponding P4/T16 rows:

```sh
python3 fixtures/verify-forced-t4-vs-p4t16-20261016/analyze_forced_arms.py \
  /path/to/run-output
```

The analyzer requires exactly four matched rows per flow and reports full-logit
max-absolute/RMS differences, both argmax routes, top-two margins, and whether
the recorded CPU argmax agrees with the stored F32 payload. The experiment is
diagnostic only; a clean or divergent target-only result does not certify full
MTP generation parity.
