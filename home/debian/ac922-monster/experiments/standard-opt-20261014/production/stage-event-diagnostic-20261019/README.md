# Stage event diagnostic candidate (private, not deployed)

This packet is a private candidate based on the frozen L5 `bank-worker-combo.cpp`.
It does not modify shared source, the running service, the pool, or GPU state.

## Source identity

- Base: `production/safe-pool-reject-20261019/bank-worker-combo.cpp`
- Base SHA-256: `bc490d7edbba51abc224deebbabf6df4b7e4a7479bba60908b29ee006d53dc2e`
- Candidate copy: `bank-worker-combo.cpp`
- Candidate SHA-256: `25edba5f4dbd55e8cd242b27bbf82838bffa8dc5fe1f8b8b1f808dca220c09f9`
- Patch: `stage-event-diagnostic.diff`

## Runtime contract

- Default behavior remains off. `AC922_STAGE_EVENT_DIAGNOSTIC=1` is required.
- Opt-in fails closed unless model devices are exactly six distinct devices whose
  backend registration name is `CUDA`; the existing runner fixes layer split
  mode and has Meta / hierarchical expert-parallel modes disabled.
- Profile setup happens before bank/context creation and sets scheduler spans on,
  the per-call `GGML_SCHED_LAYER_GPU_SPANS_ACTIVE` gate to `0`, and CUDA event
  sampling period to `1`. The one-time trace calibration during scheduler
  initialization is outside the target wave and must be recorded as such.
- Only the exact in-band live-wave IDs `diag-event-prefill32` and
  `diag-event-decode4` can be admitted, at most once each. Admission also
  requires the native semantic gate, drained owners, ready contexts, exact
  request geometry, uncached requests, and one request per bank for decode4.
- `diag-event-prefill32` requires one 32768-token request with `max_tokens=8`.
- `diag-event-decode4` requires four uncached requests with `max_tokens=512`,
  placed on four distinct banks.
- The RAII scope sets `GGML_SCHED_LAYER_GPU_SPANS_ACTIVE=1` only around the
  synchronous live-wave runner and resets it to `0` on return or exception. If
  that reset fails, it tries to disable `GGML_SCHED_LAYER_GPU_SPANS`; the runner
  then fails rather than continuing with uncertain instrumentation state.
- Logs deliberately distinguish `selector_selected` from
  `backend_selected=unverified`. Acceptance requires the backend's actual
  `scheduler layer GPU spans ... selected=1` init log and real span output with
  zero dropped samples. The source inspection establishes the expected plain
  CUDA layer6 route; runtime backend selection remains to be confirmed.
- Measurement scope is the full live wave, including its prefill, warmup,
  generation, and wave-level overhead. It does not claim perfect phase isolation.
- Full logits and prefill-logit dump remain disabled.

## CPU policy test

Run from this directory:

```sh
c++ -std=c++17 -Wall -Wextra -Werror -pedantic \
  test_stage_event_diagnostic.cpp -o /tmp/test-stage-event-diagnostic
/tmp/test-stage-event-diagnostic
```

The worker and engine were not built. No model, service, or GPU test was run.
