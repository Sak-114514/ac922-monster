# Meta CUDA submit-worker NUMA staging, r1

Date: 2026-10-16. Read-only source investigation plus CPU-only staging. No live/core source,
system affinity, cgroup, driver, power, or GPU state was modified.

## Feasibility and dispatch lifecycle

The candidate is feasible as a default-off Linux-only host-thread scheduling option:

- `ggml_backend_meta_worker` owns a persistent `std::thread`; it waits on a condition variable and
  runs a posted job on that thread (`ggml-backend-meta.cpp:3203-3255`). Its destructor sets `stop`,
  wakes the worker, and joins it (`:3214-3227`).
- The Meta context creates one worker for each simple backend index 1..N-1 only when P2P AR
  initialization succeeds (`:3445-3475`). In merged/chunked graphs those workers submit each child
  graph while the Meta caller submits backend 0 (`:4693-4703`, `:4729-4746`). If the graph is
  unmerged, the Meta caller submits each child backend in a serial loop (`:4904-4910`). The staging
  patch pins persistent workers once at startup and scopes the Meta caller only around direct
  graph-submit, cross-device copy-submit, and synchronization calls. Graph building/replay checks
  and CPU PLE execution remain outside those scopes.
- `ggml_backend_dev_props.device_id` is documented as a PCI BDF in lower-case
  `domain:bus:device.function` form (`ggml-backend.h:161-178`). On this host I read sysfs for the
  six V100 BDFs: GPUs 0-2 are NUMA node 0 with `local_cpulist=0-87`; GPUs 3-5 are node 8 with
  `local_cpulist=88-175`. The sampled topology reports four SMT siblings per physical core.
- The resolver intersects each GPU's sysfs `local_cpulist` with `sched_getaffinity()` for the
  current thread, reads `thread_siblings_list`, and assigns one CPU per physical core in Meta member
  order. Missing BDF/NUMA/topology, malformed masks, no local CPUs in the allowed mask, or exhausted
  independent cores produce an explicit `action=fallback reason=...` log. The default is off unless
  `GGML_META_SUBMIT_AFFINITY=1` is set.
- Every temporary owner binding saves that thread's current mask, narrows it to one selected CPU,
  and restores the saved mask after the single backend call. It never calls `sched_setaffinity` on
  the process, never changes memory policy, and never adds CPUs outside the thread's current
  allowed mask. Worker threads bind only themselves and retain their startup mask until shutdown.
  If a cgroup mask changes while an owner scope is active, restoration checks the resulting mask;
  a truncated restore is logged instead of claiming an exact restoration.

## Current-launch constraint and expected value

The reported launcher uses `numactl --cpunodebind=0`. If that is the effective thread mask, the
resolver can choose independent local cores for GPUs 0-2, but GPUs 3-5 have no allowed local CPU and
must log a fallback. It will not widen that mask to reach NUMA 8. As configured, this is primarily a
host-submit core placement experiment for the first island, and it cannot remove cross-XBUS host
submission locality for the second island. A later run whose existing allowed mask includes NUMA 8
can test that case without changing system policy.

The planning cost is one allowed-mask read and PCI/CPU sysfs discovery per Meta context, only when
opted in. Each persistent worker does one affinity read/set at thread startup and restores at
shutdown. The caller path performs one affinity read, one set, and one restore per direct submit,
copy, or sync call; merged chunks can therefore multiply the syscall cost. This has not been timed
on AC922. It is likely small per call but may offset gains on short chunks, so CUPTI/host-submit
timing must decide whether the option is useful. Default-off mode performs no sysfs lookup or
affinity syscall and leaves existing behavior intact.

Risks to keep visible:

- No local CPU inside the effective cpuset means no locality change; fallback is expected for the
  NUMA 8 island under a node-0-only mask.
- Several concurrent Meta contexts that share physical devices may choose the same owner CPU. The
  existing submit-order locks serialize shared-device enqueue when context-view protection is on,
  but no global CPU-core lease is introduced here.
- Linux affinity APIs and sysfs are platform-specific. Non-Linux builds leave the feature in
  fallback mode; no broader platform affinity change is proposed.
- Any measured benefit must be compared with per-chunk affinity-syscall overhead. This is a staging
  hypothesis, not a performance claim.

## CPU fixture and patch evidence

`meta-numa-submit-affinity-fixture.cpp` tests the exact header helpers using a temporary fake sysfs
tree and a fake affinity API. It covers BDF validation, CPU-list parsing, NUMA/local CPU lookup,
SMT sibling exclusion, cpuset intersection and fallback, default-off/literal-1 opt-in, no-widening,
nested scope behavior, normal/early-return restoration, set failures, and runtime cpuset narrowing.
The fake API confirms a narrowed cpuset is retained and reported instead of widened.

Fresh verification:

- CPU fixture: all PASS, `CPU_ONLY=1 GPU=not_run` (`cpu-fixture-r1.log`).
- V50 `ggml-backend-meta.cpp` translation unit: `-fsyntax-only` PASS (`syntax-only-r1.log`).
- Patch `meta-numa-submit-affinity-r1.patch`: whitespace check and dry-run against a preserved V50
  source copy PASS; SHA-256 `a8157b0136b6de7880df4ce89c4d2040dc31910c44492c0c14b692252cb4c80c`.
- No GPU test, engine build, system affinity change, or CPU frequency/power change was performed.

Artifacts are under `fixtures/meta-numa-submit-20261016/`. Keep the patch staged; the next decision
depends on CUPTI/host-submit attribution and the actual process cpuset.
