# R389: release pinned prefill sources independently of compute backlog

The actual V386 run spent 11.31 cumulative seconds waiting for CUDA5 pinned
source slots across native, clean and diagnostic requests. This is overlapping
host wait, not a promised wall-time saving. Current H2D uses the compute stream.

Add an opt-in private CUDA registry proc and use it only inside the existing
plain-CUDA layer-prefill pinned-input branch. Keep the original destination
consumer wait and peer-reuse fences. The additional nonblocking upload stream
waits for that exact old destination-slot consumer, copies the existing pinned
snapshot, records its existing source-done event, and orders compute after it.
No tensor buffers, compute kernels, formats, attention or quantization change.

The proc may decline only before enqueue. An error after enqueue drains upload
before returning failure; the scheduler must not issue a fallback second copy.
Backend synchronization and teardown drain the upload stream. No CUDA context
header change: one CUDA wrapper object and one CPU scheduler object only.

Gate with actual CUDA pending-event and 4-device-slot/8-host-slot wrap tests,
full integer output oracle, invalid-input declines, and queued teardown. Then
combine with independently qualified tail graph changes for one real 64K run.
Preserve 58C/512MiB/4GiB DDR guards, exact PLE placement and native semantics.
