# V439 host-child prepare-only helper candidate

This candidate starts from the source and API header used to build R398's R7
host-child helper. Its V3 exports reuse that implementation's existing
`cudaGraphInstantiateWithFlags(..., 0)` child, private upload stream, and ordinary
V2 host launch. No device-launch controller graph or executable is allocated.
V1/V2 signatures and their existing host-child launch route remain present.

`ac922_tail_bridge_v3_prepare_captured` validates owner/device/error state,
checks cache capacity and same-key quiescence under the owner lock, then uses
the R398 prepare routine to enumerate supported captured nodes, instantiate
the child with flags 0, upload it, and create its completion event. It does not
launch the captured graph or record the completion event. The existing V2
launch then host-launches that child and records its usual done event. A
same-key event that is still in flight declines before retirement or binding
mutation; a graph launch whose event record failed is tracked as unfenced and
cannot be rebound or launched again until its stream is drained.

The R398 prepare routine uploads on a separate nonblocking stream and
synchronizes that upload stream before returning. This fence is retained: the
child must finish uploading before the later host launch on the compute stream,
and no cross-stream dependency is otherwise established. Preparation does not
synchronize the compute stream. `cudaStreamSynchronize(compute_stream)` is used
only on the exceptional path where a child launch succeeded but recording its
done event failed; that stream fence prevents unsafe retirement or replay.

With `AC922_TAIL_STAGE_TIMING=1`, a host-only diagnostic measures wall time and
the calling thread's CPU time around the child `cudaGraphInstantiateWithFlags`
call. It adds no CUDA API calls or CUDA call-order changes. Thread CPU time
includes runtime waiting/spinning and is not a measure of pure compiler work.
The diagnostic is off by default and emits one line per child instantiation.

## Evidence boundary

This is an unbuilt helper candidate. Its parent identity is R398 source SHA
`6c60f4f21405e18143befac37e6503b3708f9de15bbe503f6e5c79b7783f753b` and
R398 binary SHA `4957bc1841361818ca4245394aa7617efd9a64e1f39d98f9d079fa6f08433bb1`.
Source checks can establish the V3 call and quiescence contracts, but not
CUDA-library behavior. Build, six-GPU direct and GGML scheduler fixtures, and
any model timing remain root-owned follow-up evidence.
