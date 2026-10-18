# V152 leaf-slot regression and V160 tiny result

## Reproduced bad leaf

The V155 debug-guard run stopped at the second scheduler copy of A's graph, before an output oracle was reached. The leaf was `CPU#A-small.frame1.input.token_ids#0` with shape `[16,1,1,1]`, `flags=3`, `data=null`, `view_src=null`, `buffer_id=-1`, and `n_buffers=3`. The previous frame was queued; the failing allocation was A frame 1 with expected scheduler copy 1. The guard intercepted the invalid leaf before the allocation-size query. Raw details are in [the guard log](v152-galloc-badleaf-20261016T20261016T232954Z.log) and [its summary](v152-galloc-badleaf-summary.json).

Stage OFF, stage ON, and the V143 stage-OFF control all failed in the same scheduler allocation path. Those runs did not verify outputs. The V155 allocator debug stack localized the path to `ggml_gallocr_init_tensor` at `ggml-alloc.c:972`, called from the leaf loop at `ggml_gallocr_alloc_graph:1078`.

## Causal confirmation from the CPU public-API regression

The independent V160 CPU-only public allocator regression is in [galloc-leaf-validation-v160](../../galloc-leaf-validation-v160-20261017/README.md). It reproduces the stale leaf-slot failure when a preallocated CPU input moves from leaf slot 0 to leaf slot 1: V155 child `alloc_graph` exits with signal 11, while V160 returns false and recovers through `reserve` then `alloc`. The 32-to-64 unused-leaf growth case also requires reserve on both runtimes. This supports the leaf-slot diagnosis without using the GPU result as proof of allocator internals.

## V160 V152 tiny gate

With the V160 allocator fix, the same V152 runner and unchanged 32x16 shape completed stage ON:

- Two independent 3-backend schedulers used four copies each. Eight A frames and eight B frames were interleaved; both included ABA inputs and immediate source overwrite. All 16 outputs matched the exact CPU oracle and all canaries remained intact.
- Scheduler A reset and grew from 32x16 to 128x1024 while retaining the input staging ring. All eight large outputs were exact with intact canaries.
- The cap `1` selector fallback was observed and its output was exact. The real per-backend budget overrun remains covered by the CPU planner mock because valid production caps start at 256 MiB.
- The V160 log reported selected staging on both normal schedulers, four copies, zero declines, and high-water marks of 8 MiB and 32 KiB. It recorded private CPU `GET_ROWS` and CPU-host inputs feeding CUDA backends with device-event waits. The output stayed on CUDA1 and was copied to distinct pinned sinks before one final drain.
- Scheduler teardown completed with staged events recorded. The run ended with no compute PIDs, 0% utilization, all cards below 39°C, and more than 16 GiB free per card.

Full per-frame evidence and DSO hashes are in [the V160 tiny summary](v152-v160-stage-on-20261017T001438Z-summary.json) and [raw log](v152-v160-stage-on-20261017T001438Z.log). This validates the bounded fixture route; it is not a model run or throughput claim.
