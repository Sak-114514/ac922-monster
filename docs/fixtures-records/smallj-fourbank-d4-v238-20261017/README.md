# V238 four-bank D4 candidate

This fixture is a bounded, one-model D4 candidate built on the frozen V235
runtime, with V234's small-J/property-cache/pinned-floor overlay. It keeps four
native semantic canaries, then measures four 16K-input / 512-output flows. The
long outputs are throughput and token-identity data, not semantic scores.

## State and gate

- The V238 C++ runner and hostlink are root-owned; the frozen source hash is
  recorded in `v238-root-profile.json`.
- Root reported hostlink pass against V235, with all seven required DSOs
  resolved. No V238 model has been loaded and no GPU has been used by this
  fixture preparation.
- The V221 KV shape policy is byte-identical and unchanged. For each lane,
  `16,384 + 512 + 4 lookahead + 1 seed = 16,901 <= 17,152` context tokens;
  D4 verify rows plus seed are `5 <= PAD6`.
- `verify_v238_d4_native4.py` has its own strict D4 labels and D4 runtime
  assertions. It delegates the four prompt/output checks to the frozen V146
  native semantic implementation. It does not call the D3-only V221 wrapper.
- `run-v238-root.py` refuses to read sudo input or start anything unless
  `ROOT_V238_WINDOW_RELEASED=1`; root must set this only after V237 releases the
  model window. The actual one-model run remains bounded by the in-band root
  guard and `RuntimeMaxSec=1800s`; there is no automatic retry.

## Root launch interface

After V237 completes and root explicitly releases the window, run from a
terminal with the root-provided sudo credential supplied on stdin without
putting it in command history or a file:

```bash
ROOT_V238_WINDOW_RELEASED=1 python3 run-v238-root.py v238-root-profile.json
```

The wrapper reads one credential line from stdin, pins the V235 report and
runtime libraries, checks the runner's resolved DSOs, starts the in-band
guard, and retrieves the D4 canary/throughput artifacts. Never run this wrapper
before V237 releases the GPU window. Do not reuse its tag or logs for another
candidate.

## Local checks

```bash
python3 verify-v238-static.py
python3 test_v238_native4.py
bash -n root-host-link-v238.sh
```

The checker test uses V234 D3 outputs mapped into a synthetic D4 metadata
fixture solely to exercise the validator, and includes a deliberately
corrupted-answer rejection test. It is not D4 model evidence or a performance
result.
