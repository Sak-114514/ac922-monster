# V436 ABI-v3 prepare-only helper

This candidate starts from the frozen R5 helper source and ABI-v2 header.
`ac922_tail_bridge_v3_prepare_captured` reuses R5's owner/key registry and
private upload-stream preparation, but does not launch a graph or record its
completion event. `prepared=1` means the exact captured graph/stream binding is
prepared; `prepared=0` is a safe decline for capacity, unsupported graph shape,
cached decline, or a same-key generation that is still in flight. The caller
must continue with its normal ordered execution path after a zero result.

The same-key in-flight check occurs before retirement or binding changes.
Recorded events are queried: `cudaErrorNotReady` maps to `prepared=0`, while
other CUDA errors propagate. A controller enqueue whose event-record call
failed is marked as unfenced; V3 declines rather than rebinding it. The
allocation-free `ac922_tail_bridge_v3_can_prepare` probe returns 1 for an
absent key when capacity remains, and 0 for an in-flight key or full cache.
It does not create a context or key entry.

The original four R5 exports and their code paths remain. The new header adds
only the two optional V3 symbols. Cache bound remains 32. The build script
builds one SM70 CUDA translation unit with relocatable device code and
`cudadevrt`; it is guarded and was not invoked here.

The CUDA fixture prepares six distinct device-owner contexts concurrently,
checks that output sentinels remain unchanged, repeats an exact prepare,
launches through V2, checks an in-flight decline, safely rebinds to a changed
captured graph after completion, drops contexts, and verifies the 32-key
capacity/unsupported-node behavior without launching the unsupported graph.
The in-flight overlap uses ordinary repeated device memset work rather than a
spin or wait-gate kernel. This staged fixture has not been built or run.
