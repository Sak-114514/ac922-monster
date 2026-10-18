# TurboPrefill empty recurrent-state view fixture

This isolated fixture reproduces the `n_rs == n_seqs` graph shape where `rs_s_copy_extra` is a zero-byte CPU I32 view, consumed by a MetaB `GET_ROWS` + `CPY` no-op. The view is explicitly expanded into the graph. Core source was not modified by this fixture lane; no full model was loaded.

## Contract checked

- The CPU I32 master `rs_s_copy` has `kTokens` elements; `rs_s_copy_extra` is a zero-element view at the end of that master.
- The graph explicitly expands a zero-byte `GET_ROWS` result and `CPY` into the MetaB side effect.
- With `GGML_SCHED_DEBUG=1`, the test inspects the scheduler’s split-input report and requires `rs_s_copy_extra` to appear in MetaB split #2. It independently asserts the view has zero bytes and CPU backend ownership.
- V41 is expected to reject `seal_last` on this unplanned zero-byte CPU input and drain the frame.
- V42 is expected to skip that zero-byte input and complete the normal eight-frame exact matrix.

## Build and run

Host compile/link only:

```bash
/home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/turbo-empty-rs-view-20261016/build.sh
```

The fixture scripts enforce a separate explicit GPU window, fixed six-GPU UUID order, no pre-existing compute processes, `<58C`, at least 512 MiB free, and runtime library provenance. With an assigned window:

```bash
TURBO_GPU_WINDOW=1 /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/turbo-empty-rs-view-20261016/run.sh v41-reject
TURBO_GPU_WINDOW=1 /home/debian/ac922-monster/experiments/standard-opt-20261014/fixtures/turbo-empty-rs-view-20261016/run.sh full
```

`v41-reject` uses runtime-perf-v41 and runs only one submitted frame after replay warmup. `full` uses runtime-perf-v42 and runs the existing eight-frame two-slot ring, exact serial comparison, owner lifetime, canary, unsealed-drain, and same-sequence ABA checks.

## Results

- V41 expected rejection passed: `ZERO_CPU_VIEW_ASSERT split=2 name=rs_s_copy_extra bytes=0 backend=CPU in_metab_split_inputs=1`; then `seal_last` reported `unplanned MetaB input tensor=rs_s_copy_extra source=CPU`, and the aborted frame drained and released its owner. The first harness attempt is retained as `results/turbo-empty-rs-view-v41-attempt1.log`; it exposed only an incorrect fixture expectation about `finish()` returning true after `abort()` already reset the transaction. The corrected expected-reject run is `results/turbo-empty-rs-view-v41-empty-rs-view-reject.log`.
- V42 full run passed: all eight frame outputs and final MetaA/MetaB recurrent states were bit-exact to the serial reference; canary, owner retirement, unsealed drain, and ABA checks also passed. See `results/turbo-empty-rs-view-v42.log` and `results/safety-v42.log`.

The run used only the synthetic graph fixture; it establishes this scheduler empty-input contract, not full-model serving behavior or throughput.
