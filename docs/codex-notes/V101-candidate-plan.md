# V101 preserve allocated Meta tensors on context append

V100 completed native4 and full2x8K/4x8K cases,135.73/162.02 sharedTG, then singleton MTP canary aborted frozen static split cache miss mtp_gpu_chain_tokens. Source: configure_mtp_gpu_chain allocates carry/tokens, appends an embedding tensor and calls alloc_ctx_tensors again before marking the returned embedding buffer WEIGHTS. Meta allocator rebinds every tensor regardless of existing buffer; ordinary allocator preserves existing allocations. This wrongly migrates dynamic token/carry tensors into the frozen embedding buffer.

Fix Meta ctx allocator to leave already backend-bound tensors untouched. Preserve frozen-WEIGHTS assertions; allocate new replica only. Add a small append-allocation regression covering pointer/data retention, dynamic tokens, WEIGHTS replica and free-order behavior before another model load. One Meta CPU TU, no CUDA arithmetic change.

Actual Turbo denial in V100 was pinned-budget: payload201773312×2 +FIFO277717000 =681263624 >536870912, despite topology+frame state active. Next long candidate uses smaller UBatch within same512MiB pin cap, not graph-cache speculation. Longer64K masks may increase budget; record actual selection/fallback instead of assuming all frames overlap.
