# V397 private host-child candidate

Status: **prepared only**. The engine, frozen runtime, GPUs, server, and services were not modified or launched by this preparation task. The profile is intentionally non-launchable until root completes the CPU-only hostlink and explicitly releases it.

## Candidate

One model load will run the exact Paris 8192 semantic gate, one clean uncached 64K prefill, a 90-second cooldown, then one diagnostic-only 32K prefill. The diagnostic CUPTI window covers the sole 32K target chunk before MTP. The shorter diagnostic is intended to avoid repeating V392's second 64K request that reached the 58 C guard.

This combines four requested changes: R400 `GGML_SCHED_LAYER_DEVICE_COPIES=8` with eight pinned host slots, R398's host-child single-exec helper, graph variants cap 8, and cross-UID snapshot lookup. Any full-model result must be reported as a combined candidate; it cannot isolate the gain from any one setting.

## Pinned inputs

- Runtime: `runtime-perf-v400`, build `20261018T033835.695366Z`, report SHA256 `a00b50ddbc2d38aa20b442efa1941d3f0f37c91350b18eba1f2851298072e60a`.
- Helper: R398 R7, `fixtures/host-child-single-exec-v398-root-20261018/build-r7/libac922-tail-bridge.so`, SHA256 `4957bc1841361818ca4245394aa7617efd9a64e1f39d98f9d079fa6f08433bb1`.
- The helper gate passed a six-GPU primitive fixture. That is not full-model activation evidence. The controller requires the runtime marker `launch_mode=host_child_single_exec controller_allocated=0 captured_math_unchanged=1`.

## Verification and release

`verify_v397_static.py` checks the scenario token counts, 90-second cooldown, helper/runtime pins, early credential read, and R400 eight-copy activation requirements. `root-host-link-v397.sh` is prepared for root-owned CPU-only compilation against R400; it has not been run here. `run-v397-root.py` reads the sudo credential from stdin before any SSH/SCP operation, then retains the 58 C, 512 MiB GPU-free, 4 GiB DDR, and in-band guard checks.
