# V37 Prefill Frame 与 QSA Pooled Block Cache 正确性审计

日期：2026-10-16  
范围：只读追踪 V37 源码中的 freeze_current、QSA block cache validity、lazy-prefill dispatch shape。未构建、未调用 GPU、未改 core。

## 结论

- 存在单序列 pooled block cache 的失效跟踪缺口：冻结帧明确绕过共享 pooled-block 快路径，改用冻结的 indexer cell metadata 做逐查询 QSA 选择；但 frame freeze 没有把 owner memory 上的全局 blk_valid 置为 false，也没有登记 dirty block 范围。若 frozen frame 实际写入了新的 attention/indexer KV，接下来的普通 context 仍可能通过 blk_ok()，然后小 decode 走单缓存 fast path，读取尚未由 frozen frame 生成的 pooled rows。
- 这不是对 V37 已发生输出错误的断言。由于 frame 候选当前又被 lazy-prefill 的 shape 门挡住（实际 selected_ubatch=256，但 frame gate 比较配置上限 1024），该 stale-cache 路径在该批次不能通过 frame_candidate。若按下述方式让 256-token frame 合法启用，必须先修 block-pool 的状态同步；否则 latent correctness bug 会暴露。
- LLAMA_QSA_POOLED_MULTI 是不同的每序列表/事务机制。它没有启用，frame gate 明确拒绝其环境变量；单序列全局 mem_blk / blk_valid 和 multi 的 dirty_pb / commit 状态不可混为一谈。
- 可以用实际 dispatch_ubatch 的 256-token full chunks 合法启用 frame gate，不需要降低 --ctx-size、缩小 attention/indexer KV backing 或损失 KV 容量。当前 gate 使用 cparams.n_ubatch=1024，和 lazy policy 选出的实际 dispatch 上限脱节。

## 单序列 block cache：代码状态路径

1. 单缓存创建/清空时 blk_valid 初始为 true：llama-memory-hybrid-idx.h:190-192；clear() 在单序列模式也置 true（llama-memory-hybrid-idx.cpp:238-250）。该标记的注释定义为“单序列按位置顺序写、block 成员连续、完成 block 的 pooled key 不会变化，并在 blk_valid 期间增量更新”（hybrid-idx.h:113-119）。
2. 普通 live context 的 apply() 只在当前 ubatch 不连续地写入下一个 cell 时把 blk_valid 清为 false（hybrid-idx.cpp:1555-1570）。ubatch_cells_follow() 检查 cell ids 连续，且 used_max_p1/get_used 恰好到当前 ubatch 末尾（llama-kv-cache.cpp:2886-2905）。顺序追加的 prefill frame 满足此检查，所以这段不触发 invalidation。
3. freeze_current() 先复制 indexer context/cell metadata，再构造只读 frame snapshot；它要求单序列且拒绝 qsa_multi（hybrid-idx.cpp:1675-1728）。它设置 frame-local block_pool_fast_path=false（1720），但不会更改 owner 上的 mem->blk_valid。全源码中 block_pool_fast_path 目前只见于字段声明和这里赋值，没有被读取。
4. frozen context 的 blk_ok() 无条件返回 false（hybrid-idx.cpp:1592-1607）。Qwen4EXP 因此走 build_qsa_top_k()，不走 build_qsa_blk_select()。这个普通路径会把本帧原始 indexer K 写进 mem_idx，再基于 blk_cells 从 mem_idx gather 成 block members 并对每个 query 做选择（qwen4exp.cpp:1995-2031）；它不写共享 mem_blk pooled rows。
5. 共享 mem_blk 的更新只发生在 build_qsa_blk_update()：算均值/归一化/rope 后，以 ggml_set_rows 写入 mem_blk（qwen4exp.cpp:1800-1825）。build_qsa_blk_select() 会先更新当前 ubatch 触及的 block，再用整个 pooled cache 做 scoring（1830-1853）。普通 build_qsa_top_k() 没有这一步。
6. 下一普通 context 上，apply() 对连续追加仍保留 blk_valid=true；blk_ok() 只检查 pooled cache 存在、blk_valid、attention/indexer cell ids 是否连续且对齐（hybrid-idx.cpp:1592-1607），并不检查每个 visible pooled row 是否已由新 cell generation 更新。因此未刷新块可被当作有效行。
7. 长上下文的下一 token 可满足单缓存 fast-path 门。模型 GGUF header-only metadata 为 indexer_top_k=2048；单缓存 fast selector 的条件是 causal+flash、单 stream、上下文完整 block 数超过 top_k/ratio 的预算（qwen4exp.cpp:1294-1320）。32K used-prefix 明显越过该阈值。Sparse kernel 开启时，该路径 token 数上限放宽到 UINT32_MAX；1-token decode 符合小批次条件。
8. 第一条非冻结 decode 的 build_qsa_blk_select() 会刷新它自己触及的 block，但不会全量补齐之前 frozen frames 跳过的 pooled rows。它随后仍会对所有 block 行打分选择。若这些行是空的、旧上下文留下的 key，或其他旧值，选择结果与当前 indexer KV 不一致。

因此，静态代码能确认“可能读 stale pooled rows”的状态缺口。实际 V37 是否已经命中，需要当前 server 日志里的 frame active/freeze 与下一 token 的 qsa_blk fast 命中记录；本审计没有运行请求，也不据此宣称已经观察到语义错误。

## 与 LLAMA_QSA_POOLED_MULTI 的边界

- 单序列 cache：一个全局 mem_blk + 全局 blk_valid；仅在单序列按位置连续的条件下使用。
- Multi cache：构造需 LLAMA_QSA_POOLED_MULTI=1、n_seq_max>1、unified 且所有 indexer ratios 一致；随后使用每个序列自己的 qsa_multi_seq_t（valid/gappy/overflow/dirty_pb）和 pending qsa_multi_pooled_to，而非单序列 cache 的 blk_valid（hybrid-idx.cpp:108-160、194-211）。
- multi 输入会登记新 token、标记对应 dirty_pb，qsa_multi_fill() 建立本 graph 要刷新的 pooled rows，apply() 前提交上一个 graph 的 qsa_multi_commit()；相关路径为 qwen4exp.cpp:1193-1197、1565-1604 与 hybrid-idx.cpp:826-885、888-955、1097-1109。
- freeze_current() 直接拒绝 qsa_multi（hybrid-idx.cpp:1676-1678）。frame gate 也要求 !blk_multi() 且 LLAMA_QSA_POOLED_MULTI 环境未设（llama-context.cpp:2986-2989）。V37 当前交接/profile 标注此 env 未设置；V37 runtime manifest 本身只含二进制和库 provenance，不含启动环境。
- 因此这次发现针对单序列 mem_blk 的 global validity；不要用 multi 表的 dirty_pb/commit 论证单缓存已经安全，也不要把单缓存缺口推断成 multi 同样有问题。

## Lazy-prefill UBatch 与 frame gate

llama_prefill_workspace_ubatch() 只选择 scheduler 的实际分批大小；注释明确说明它不改 KV backing、可见性、token 顺序或精度，256 是 scheduling tile，不是 context 限制（llama-prefill-workspace-policy.h:5-16）。process_batch() 用当前 used_max、即将追加的 prompt tokens、cache capacity 算 dispatch_ubatch，再把它传给 memory->init_batch()（llama-context.cpp:3781-3797）。

以 requested=1024、used_max=32768、upcoming=8192、cache_capacity=132096 为例：

- projected view = 32768 + 8192 = 40960；round-up 8K 后仍为 40960。
- affordable = floor(1024 × 8192 / 40960) = 204 tokens。
- policy 按 256-token 对齐并设置 256 最小 tile，故 dispatch_ubatch=256。
- KV backing 仍是原来的 132096；变化的是每次送入图执行器的 prefill ubatch size。

与此相对，prefill frame gate 在 process_ubatch() 调 llama_context_prefill_frame_batch_ready(ubatch, cparams.n_ubatch)，即拿 actual ubatch.n_tokens 与 1024 配置上限比较（llama-context.cpp:116-135、2992-3000）。256-token 实际批次因此返回 false，候选原因是 batch-or-context-shape；不调用 freeze_current()。该调用链未把 process_batch() 的 dispatch_ubatch 传给 process_ubatch()。

修正 shape gate 可把有效目标大小（dispatch_ubatch）传到 process_ubatch，要求 ubatch.n_tokens == effective_dispatch_ubatch，而不是配置最大值。这样实际的完整 256-token、单序列、连续、无输出 chunk 可通过此 shape 门；最终不足 256 的尾块仍自然不通过。它不会修改 cparams.n_ctx、132096 KV backing 或 cache capacity，仍须满足拓扑、scheduler/FIFO、PLe-host、no-output、single-sequence 等其余 frame 条件。

## Root 修复建议（仅源码审计意见）

- 正确性首要门：只要 frame graph 不更新共享 mem_blk，就应在 frame 写入 indexer KV 前使 owner 的单序列 block pool 进入 dirty/invalid 状态，或跟踪它跳过的 pooled block ranges。此状态必须跨 frame drain 保留，直到 mem_blk 对所有需要访问的可见 block 重新计算完毕。
- 最小 fail-safe 是使下一普通 graph 退回 build_qsa_top_k() 的 per-cell 选择，避免旧 pooled rows 被使用；但当前 blk_valid 在 invalidation 后通常只在 cache 清空时重新置真（seq_rm 清空 / clear），会让固定成本 decode fast path 长期关闭。生产修复若要恢复该性能，应加明确的 pending dirty range/全量 refresh transaction，并仅在 refresh graph 已执行后恢复 valid。
- 不能只保留 frame-local block_pool_fast_path=false：它阻止冻结帧自己读/更新共享 cache，但不传播 invalidation 给下一 mutable memory context。
- shape 修正应和 pool state 修正一起评审；单独把 frame gate 从 1024 改成 256 会暴露上述 stale cache 问题。KV 容量无需降低。

## 证据路径

- freeze/frame lifecycle：src/src/llama-context.cpp:116-135, 2986-3013, 3087-3107；src/src/llama-memory-hybrid-idx.cpp:1555-1607, 1675-1728；src/src/llama-memory-frame-snapshot.h:9-18。
- 单 cache validity：src/src/llama-memory-hybrid-idx.h:113-123, 190-192；src/src/llama-memory-hybrid-idx.cpp:238-274, 308-377, 410-423；src/src/llama-kv-cache.cpp:2886-2905。
- QSA graph update/select/full rebuild：src/src/models/qwen4exp.cpp:1294-1320, 1800-1825, 1830-1853, 1924-2031, 2811-2837。
- Multi pooled state: src/src/llama-memory-hybrid-idx.cpp:108-160, 653-715, 826-885, 888-955, 1097-1109; src/src/models/qwen4exp.cpp:1187-1197, 1565-1604。
- Lazy UBatch sizing: src/src/llama-prefill-workspace-policy.h:5-16; src/src/llama-context.cpp:3781-3797; GGUF header key qwen4exp.attention.indexer.top_k=2048 (header-only parse).

