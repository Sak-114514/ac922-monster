# Owner fixture status (2026-10-16)

Latest status: V46 host build and bounded GPU fixture pass. See the V46 follow-up below.

## V45 initial attempts (historical)

- Runtime: runtime-perf-v45
- Build: PASS, host compile/link only; see build-v45.log
- Fixture binary SHA-256: c25f0473f5f964b3ab2efd0e7ed1ca0987123dfbf8a26bc5f9214e80b14cd1aa
- V45 manifest SHA-256: 2a2184af5d50ca9c759a017d47154733a06a2b5aa9aa015e08a14183605d318f
- V45 libggml-base SHA-256: 96618c00b76170a5abac982c96263a1f84e2bd3c71d5f6a2074eb67833c529c3

GPU attempts were bounded to one three-GPU NVLink island and a 30-second timeout.
The first two attempts used generic Meta allocation and failed before graph creation.
The third attempt directly used ggml_backend_meta_alloc_ctx_tensors_from_buft for
both the shared WEIGHTS allocation and axis-0 partial input. It failed identically
at the first ggml_backend_tensor_set(shared_static_f32_weights):

    GGML_META_CONTEXT_VIEWS=1: frozen static split cache miss for shared_static_f32_weights
    ggml-backend-meta.cpp:2161

Logs: run-v45-001.log, run-v45-002.log, run-v45-003.log. No graph dispatched,
so owner-ID separation, same-address ABA, TP AllReduce, exact outputs, and concurrent
scheduler behavior remain untested. All fixture processes exited; no GPU test remains.

## V46 follow-up (2026-10-16)

V46 build passed after making Family the sole owner of each scheduler and clearing
the outer aliases at transfer time; this prevents catch-path scheduler double-free.
The fixture now uses no_alloc=true for GGML contexts as required by backend tensor
allocation. Public split-state inspection uses assume_sync=true, so the check was
corrected to expect the materialized MIRRORED descriptor; runtime graph construction
uses assume_sync=false for the TP partial boundary.

- Build: PASS; see build-v46.log
- Runtime: runtime-perf-v46
- Runtime manifest SHA-256: 5551f7fc0266541ad2f8fe7507e046e664796c988ba975176353150a7f8cb636
- Fixture binary SHA-256: b403905ec88e8bbcfbef30c1020b6082d225af30910851402d0dd4b4bf3aa1be
- GPU run: PASS, results/run-v46-003.log, bounded by a 30-second timeout on GPUs 0/1/2
- MIRRORED sequence: A/B/A exact; same-address B ABA exact; A remains exact after B ABA
- Owner trace: IDs 7, 9, 11 are distinct; all 3 bindings share buffer token 4 and root shared_static_f32_weights
- TP partial/AllReduce concurrent A+B: exact outputs 66496 and 67296; A remains exact after B scheduler/backend teardown
- Host log assertions: 3 creates, 3 binds, common buffer/root, 8/8 exact result rows, PASS
- Resource release: no fixture process remained; GPU 0/1/2 returned to 4 MiB each at 39/38/38 C

Earlier V46 attempts are preserved as run-v46-001.log (fixture no_alloc setup)
and run-v46-002.log (incorrect expectation from the assume_sync=true query).

## V49 lockstep regression status

The V49 frozen build was host-compiled and run once under a 30-second timeout.
That run passed the previous owner, ABA, TP-AR, teardown, and API-overlap gates
(run-v49-001.log, overlap 59,282 us). However, the A partial graph was on its
first shape invocation, which intentionally runs unmerged; the debug log did not
show the merged multi-chunk lockstep branch. Therefore this run does NOT prove
the reported rank-lock/global-suspend path.

The fixture has since been changed to serially warm A's new partial-AllReduce
shape once, then race its second cold merged/chunked graph_compute against B's
cold view alloc/query. It requires GGML_META_CHUNK_NODES=1 and GGML_META_DEBUG,
records actual API-interval overlap, and will need a fresh build/run when the
root grants another short GPU window. The updated source passes a host-only g++ -fsyntax-only check but has not been linked against V49 or run; the P4 full-model run currently owns the GPUs.
