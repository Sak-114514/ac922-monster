# V309 single-copy-per-device scheduler fixture

## Purpose

Validate the proposed PR #29963-style user-input copy reduction in the existing
plain six-CUDA-device scheduler, without loading a model or adding a kernel.
The scheduler remains at four graph copies. The candidate may keep one reusable
input destination per physical CUDA device; it must not shrink the scheduler's
four graph slots or weaken their event ordering.

## Scope and source boundary

- Use the small actual-GGML scheduler setup and CPU oracle patterns from V285 / V299.
- Exercise both a caller-owned CPU input and a CPU-produced `GET_ROWS` result
  consumed by six CUDA stages.
- Add one explicit two-level CPU-rooted view sent to a peer CUDA stage. GGML
  canonicalizes view metadata to the base tensor, which the fixture asserts.
  Record its allocation-time route; if it reaches the known-unsafe generic
  fallback, do not execute that graph.
- Add a distinct CPU-rooted `ggml_scale_inplace` view routed to CUDA2; it must
  also decline the one-copy path during allocation; do not execute that negative
  graph.
- Keep the V299 pinned host-source pool at four slots. Do not mix host-source
  slot expansion into this physical-device-copy test.
- Do not modify engine sources or rebuild the engine. Only the bounded fixture
  binaries may run under the guarded six-GPU test gate; no model is loaded.

## Required selection behavior

The default/legacy route remains available and reports four per-device copies.
The opt-in candidate reports `selected=1 physicalDeviceCopies=1` for each
eligible input and each physical CUDA device, while `ggml_backend_sched_get_n_copies`
continues to report four. Unsupported or ambiguous views, allocation-cap
exhaustion, or unavailable event/fence support decline to the old path. A
decline must drain any already queued work before releasing or reusing storage.

## Fixture sequence

1. Preflight all six distinct CUDA backends in the requested physical order;
   record backend/device identity and runtime/source provenance.
2. Construct the plain six-stage graph plus CPU oracle and canary-protected
   caller/producer inputs. Confirm scheduler `n_copies == 4` before allocation.
3. Execute 20 frames in five windows. Submit four fresh-UID graphs before each
   explicit drain; poison each frame's CPU inputs and `GET_ROWS` prefix
   immediately after submission. Queue each of the six stage outputs into that
   frame's own pinned `CUDA_Host` buffers before submitting the next graph. Query
   completion nonblockingly where the runtime exposes CUDA event query; log
   pending status and allocation-time pending transitions without adding
   artificial work or waits.
4. At each window drain, compare all six outputs per frame against the CPU
   oracle bit-for-bit and verify input/output canaries, graph UID, ring slot,
   destination device, and selection markers.
5. Allocate the two CPU-rooted view/in-place cases only long enough to record
   their route markers. Do not execute them if they reach the known-unsafe generic
   fallback; the in-place alias is required to show an explicit decline.
6. Run a separate three-frame 40 MiB input test under the 256 MiB cap. The
   second host slot must hit the real budget decline; verify drain and exact
   legacy fallback output for all six destinations.

## Non-goals

This is a correctness/lifetime fixture only. Pending event observations are
reported as evidence, not inferred from the submit count. It does not establish
prefill throughput, full-model correctness, or a production gain.
