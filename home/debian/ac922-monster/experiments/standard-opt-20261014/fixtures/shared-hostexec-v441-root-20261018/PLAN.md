# V441 shared host-exec helper candidate

## Scope and baseline

Stage a helper-only, default-off candidate under this fixture. Copy the passing
R439 helper and direct/scheduler tests into `helper/source/` and
`tests/baseline-r439/` as immutable rollback evidence. Do not edit engine files,
build anything, load a model, or use GPUs. Root reviews the complete candidate
before deciding whether to build it.

The opt-in is `AC922_HOST_GRAPH_SHARED_EXEC=1`. When absent, existing R439 V1,
V2, and V3 code paths retain their original behavior. In shared mode, use only
the R439 host-child executable instantiated with flags 0; never create a device
launch graph or controller kernel.

## State model

- Keep the existing R439 owner-only map for the default path. In shared mode,
  use `(owner, device)` context identities and its 32-entry per-key cache.
  Give each context a monotonically increasing generation and a close-admission
  gate with active-operation counting. `drop_context` marks the context closing
  in the registry, rejects newly admitted operations (including callers that
  already hold a `shared_ptr`), waits for admitted operations, then drains and
  destroys its slots/fallbacks before removing that exact generation.
- Each shared context is device-pinned and owns at most 32 shared slots keyed by
  the exact `(owner, device, cudaStream_t)`. A slot owns its flags-zero shared exec, last
  committed binding metadata and serial, instantiation/update/fallback
  counters, completion event, and unfenced-launch state. Each per-key entry
  separately owns its captured graph identity, stream, completion event,
  pending state, and optional fallback child exec. Do not store owning pointers
  to one executable in both places.
- Lock in one order: registry lookup, context gate/state, then slot. Serialize
  all updates and launches on a slot; independent streams retain independent
  slot locks. The source graph remains caller-owned and the caller serializes
  its mutation/capture lifetime.

## Shared binding and failure policy

- First use of a stream slot instantiates the current capture with flags 0,
  uploads it using R439's private nonblocking upload stream, and keeps the
  upload-stream completion fence. V3 prepare may do this or update an existing
  slot, but never launches model graph math or records a launch-completion
  event.
- Every V3 prepare call and every actual V2 launch in shared mode binds from
  its current `cudaGraph_t`. Actual V2 calls always attempt an update before
  launch, regardless of per-key cache hits. A shared slot's key hit must not
  suppress binding. Successful updates alone replace the committed key/graph
  token and increment its commit serial.
- V2 readiness is only an eligibility probe: it may report that a shared
  executable exists, but does not assert that the slot still contains this
  key's binding. Every subsequent actual V2 launch must still update from its
  supplied current graph.
- On the exact `cudaErrorGraphExecUpdateFailure` result, leave the shared exec
  and committed metadata untouched, prepare/use that key's separate flags-zero
  fallback child, and mark the fallback route. Every other CUDA status
  propagates unchanged; do not hide a prior asynchronous error behind fallback.
- Preserve the R439 per-key done event and pending-unfenced handling: record it
  after either a shared-slot or per-key-fallback launch, in addition to the
  executable's event. Event-record failure
  retains an unfenced marker; teardown drains the slot stream before retiring
  either kind of exec. A topology-change B fallback followed by A must show
  that the prior shared commit remained A and can still update/launch.

CUDA 12.4 specifies that successful `cudaGraphExecUpdate` updates a
topologically identical executable, while the exact update-failure status
means the update was not performed. It also specifies that updates affect
future launches only, already queued/running launches are unchanged, launches
of one exec are ordered, and `cudaGraphLaunch` permits only one executing
instance of that exec at a time. These semantics support queued A/B/A updates
through one exact-stream slot, serialized under the slot lock. [CUDA Runtime
API 12.4 graph reference](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-runtime-api/group__CUDART__GRAPH.html),
[CUDA Runtime API 12.4 graph thread safety](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-runtime-api/graphs-thread-safety.html).

## Tests

- Keep the R439 tests intact and verify their source hashes. Add a CPU-only
  update-policy test that injects success, exact update failure, and an
  unrelated error; only exact update failure selects per-key fallback.
- Add a CPU-only close-gate test for monotonic context generation, rejection of
  new/late held-context operations after close begins, and close waiting for
  already admitted operations to leave.
- Add direct CUDA graph tests: queue eight same-topology A/B/A payloads with
  eight distinct input/output buffers and no host sync between launches; fence
  once afterward and compare every output exactly. Assert one slot
  instantiation and at least seven successful updates. Use two exact streams
  for one owner with distinct keys and prove they instantiate separate slots.
- Prepare graph A and then graph B on one slot without launching either graph;
  prove both outputs remain sentinel-valued, then launch A and verify it was
  rebound from the current graph. Reuse one opaque owner across devices and
  verify device-paired contexts remain separate.
- Force a topology-changing B update. Assert the exact update-failure fallback
  runs B correctly, then A updates and runs on the unchanged committed shared
  exec. Verify slot counters and metadata show one failure and preserve the
  A commit across B fallback. Verify the next V2 readiness and launch observe
  no residual known update-failure status. Drop the context while A/B/A work is
  still queued; keep graphs and buffers alive until drop returns, then check
  both A and B outputs exactly. Exercise the direct suite on one GPU and then
  six devices; use only small fixture ADD kernels, no production model math.
- No tests intentionally generate an illegal access, Xid, poisoned context,
  reset, or synthetic CUDA API error. The negative unknown-status case is the
  pure CPU decision function.

## Root-owned gates

This plan is the pre-code review artifact. After root approves the implementation
and tests, root can build the candidate helper and run the direct one-device,
six-device, and actual GGML scheduler fixtures. This agent does not build, run
CUDA tests, or claim model behavior or speed.
