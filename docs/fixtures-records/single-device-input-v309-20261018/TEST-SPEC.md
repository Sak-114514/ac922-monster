# V309 acceptance specification

## Environment and immutable controls

- Backends: exactly six distinct ordinary CUDA devices, plus CPU.
- Graph scheduling: `parallel=true`; assert `ggml_backend_sched_get_n_copies() == 4`.
- V299 private pinned host-source ring: exactly 4 slots using
  `GGML_SCHED_LAYER_PINNED_HOST_SLOTS=4` and `GGML_SCHED_LAYER_PINNED_CAP_MIB=256`.
- No model/weights, no CUDA kernel changes, no new framework, no full-model run.
- Use unique run tag and record fixture/source/runtime hashes, CUDA device UUIDs,
  selected/fallback markers, and per-frame graph UID/slot.

## Cases

| Case | Input path | Expected decision |
|---|---|---|
| caller input | CPU-owned, flagged graph input | Candidate selected: one copy per physical destination device |
| computed prefix | CPU `GET_ROWS` output consumed by the CUDA stages | Candidate selected only if completed CPU result is snapshotted/staged safely |
| two-level CPU view through a peer | Separate CPU input → view → view → CUDA1 consumer; both views must resolve to the base tensor | Record allocation-time route/decline; graph is not executed |
| CPU-rooted in-place view | Separate CPU input → view → `ggml_scale_inplace` on CUDA2 | Candidate logs decline during graph allocation; graph is not executed |
| bounded allocation failure | Three 40 MiB frames under the real 256 MiB cap | Drain events, decline to legacy fallback, then exact output |

## Ring and asynchronous lifetime gate

- Exactly 20 executions, 20 distinct nonzero graph UIDs, 20 unique frame IDs.
- Five windows each submit four graph UIDs before an explicit scheduler drain.
- Per-frame readbacks for all six stages use separate pinned `CUDA_Host` buffers
  and events.
- Before each graph allocation, log how many earlier frames still have pending
  readback events; after allocation, log the count again and the allocation
  duration. Do not claim overlap when the measured pending count is zero.
- Four graph-copy slots observed five times each; the scheduler still reports four.
- Candidate marker must identify `physicalDeviceCopies=1` per eligible input/device;
  the physical destinations across the six stages must be exactly the six expected
  CUDA devices. Never label this as one total copy for all GPUs.
- Immediately overwrite every CPU input, row-id tensor, and CPU `GET_ROWS` prefix
  with poison after asynchronous graph submission. The frame must still produce
  its pre-poison outputs.
- Before any staging/copy storage is reused or freed, its completion event must be
  observed complete; cleanup/fallback must drain every queued copy.

## Correctness and memory-safety gate

- All six pinned readbacks per frame are bitwise identical to their independent
  CPU oracle outputs.
- Input and output guard canaries remain intact.
- CPU-computed `GET_ROWS` data is validated before poison and its submitted
  snapshot is distinguished from the next frame's value. The CPU-rooted
  in-place alias must emit a decline marker before kernel execution; the separate
  peer-view allocation result is reported as observed.
- Each log frame binds graph UID, ring slot, input identity, destination device,
  source/base view provenance, copy decision, completion/drain status, and exact
  output verdict.
- Candidate selection without a matching per-frame marker, stale/duplicate graph
  UID, partial device coverage, output mismatch, canary change, missing drain, or
  unexplained fallback is a hard failure.

## Evidence labels

- Static contract: parser/source checks only.
- Fixture correctness: actual scheduler and six CUDA devices passed the full gate.
- Performance: not measured by this fixture and must not be inferred from these
  correctness results.
