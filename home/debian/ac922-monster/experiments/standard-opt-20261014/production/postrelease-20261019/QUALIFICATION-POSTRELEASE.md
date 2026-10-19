# AC922 L2 继续叠加优化验收

2026-10-19 CST。L1基础上增量；最终状态以promotion.json为准。原模型/全部专家/权重和KV精度/固定D3/standard indexer保持不变；没有BMC、风扇、驱动、内核、频率、功耗或整机开关变更。

## 本次新增

1. 初始2–4请求使用完整chat模板后的精确token前缀，枚举最多24种request→bank映射，与原capacity/UB/retention plan联合评分。只计仍保留原bank/epoch/profile的命中；无有效命中回原序，singleton原bank0/profile。状态留在原bank，不搬KV；refill与外部队列顺序保持原逻辑。
2. 保留每路4个真实生成准备帧和全部token，使用已存在的per-device init-ready协议后，显式允许准备阶段跨bank并行。bootstrap native4依旧串行。没有重编CUDA或任何引擎DSO，仅重编worker57.34秒；七库和R5helper与L1逐字节相同。新初始化门验证了Onecat直接GGML路径；当前模型本身沿用已有QSA路径，不把这项称为新attention kernel吞吐收益。

## 实测

| 工作负载/窗口 | 本轮 |
|---|---:|
| 32K首次uncached prefill | 4244.77 tok/s |
| 同profile重复32K，完整重算 | 4753.75 tok/s |
| 64K实际输入，预留最大262K | 3914.59 tok/s |
| 四路8K，含温控等待 | 3330.77 tok/s |
| 四路512，IPC到达至最后完成 | 204.79 tok/s |
| 四路512，完整worker执行窗口 | 205.77 tok/s |
| 四路512，全输出首个至最后窗口 | 215.22 tok/s |
| 真实HTTP四路512，实际2048token/9.75224秒 | 210.00 tok/s |

L1同HTTP203.07，本轮210.00约+3.4%；IPC200.77→204.79约+2.0%。这是单样本小幅变化，不宣称大幅raw提速。32K两轮基本持平；64K本轮3914.59，未单独归因。四路8K包含4.431709秒的各bank温控等待总和，不能直接视作纯计算回退。raw300/6000目标仍未达到。

准备阶段每flow仍为4帧，等待3–4微秒；host阶段时间窗最多3路重叠、至少2路重叠393.691ms；不把host窗口冒充GPU kernel重叠证明。

## 前缀路由

四份不同文档把DOCUMENT_ID与不同NOVA值放在开头；不存在所有请求共享同一个测试前缀的假命中。

| 波次 | 整批时间 | 每路命中 |
|---|---:|---:|
| 首次四路 | 15.107秒 | 0 |
| 同序四路重放 | 1.329秒 | 10496 |
| 打乱顺序3/1/0/2 | 1.637秒 | 10496 |
| 两路子集2/0 | 0.888秒 | 10496 |
| 三路子集1/3/2 | 1.313秒 | 10496 |
| 修改早段正文的两路 | 7.455秒 | 0 |

四份文档答案3179/6842/5301/9274均正确；编辑后7301/4159正确。worker实际bank事件与原owner一致。上述是缓存请求延迟，不是raw prefill吞吐。

## 混合请求及语义

额外一次late-arrival probe：长请求开始0.5秒后才到达3个短问。长请求10.513秒完成；短问在全局0.956/1.088/1.513秒已输出，1.033/1.160/1.795秒完成，Paris/11/严格JSON全部正确。这证明目前已有idle bank能在长prefill期间接入兼容短请求，本轮不需要另写chunk-yield执行器。

原native4、4x8K、64K检索、33K/60K级HTTP检索、取消复用/refill/前缀A-B-A与earlyedit通过。4段IPC512输出的全部2048token IDs与已全文审阅的L1完全一致；另8段HTTP长输出逐字匹配上述IPC，原语义审阅适用。没有宣称全logits一致或全面事实性评测。

30API测试、affinity纯CPU、4-owner setup guard、5项真实coordinator、harness正负mock与post-cleanup静态检查通过。六个实际设备各自2context同时首次capture、CPU oracle/bitexact/replay全部通过，精确运行库哈希已绑定。

## 容量、安全与后续

总池524288cells、单请求预算262140不变。满池分配与最长实际65536输入通过；不是52万全文测试。最终只读NVML采样最低余量755.812MiB、GPU最高61°C，无OOM/fault。GPU65°C、其余原阈值和256MiB底线未改变。

另外在同次加载测了UB256 bank0容量17152/24576/32768/33024/40960，native4全对。每8K增长额外约28MiB/卡，GPU5约56–58MiB；这只是测得的稳态delta，不能直接当作任意shape峰值保证。后续有界growth slack仍未启用，将先做分配前预算与CPU策略证明，避免每几百token增长就重建context丢缓存。

worker SHA：`52d7ab82a656c794e4ffd01be15d13f59c8b3e8225d3f642df7c569d696b284f`。
原始证据evidence/、完整run/与onecat-fixture/保留；匹配哈希在lastwave-acceptance.json。
