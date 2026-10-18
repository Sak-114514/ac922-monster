# P4 V8 persistent owner threads

Before code changes, 2026-10-15.

V7 starts a fresh std::async launch thread for every target frame. This discards thread-local graph timings/caches each frame and recreates CPU/CUDA owner state. A two-bank model is long-lived; its owner threads should also be long-lived. The existing coordinator still needs one future per frame for readiness, exceptions, and join-before-KV mutation.

Change only a new runner/helper, no engine source: one persistent worker per bank, at most one task queued/in flight; task completion clears busy state before publishing promise readiness; exceptions propagate through the same future; destructor stops and joins after draining the current task. Keep serial shape warmups and V13 deep Meta guard, target verification, server KV policy and all quality gates. Disable noisy full-step per-getter logs in the new performance run; preserve periodic STEP_TIMING and per-frame clocks.

Regression fixture: persistent worker identity across100 tasks; no second in-flight submit; completion visibility; exception then recovery; destructor drains a pending task; two owners can overlap independently. Host CPU fixture on Mac and POWER9 before model. No claim of speed until real14 semantic and2/4x8K512 tests, same shared-window method.
