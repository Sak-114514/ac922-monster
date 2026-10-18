# ac922-monster

**AC922（2×POWER9 + 6×V100-SXM2-16GB，双 3-GPU NVLink 岛）上为 Qwen3.8-Flash-Next（179B MoE，512 专家 top-10，MTP-4）深度定制的推理引擎**——llama.cpp fork 单架构特化（monster 树）。

**当前战绩（2026-10-18，全部语义门验证）**：

| 指标 | 数字 | 配方 |
|---|---|---|
| decode 4 流聚合 | **282.6 tok/s** | 四 bank × 拍平 Meta6 × smallj 内核 × MTP 词表分片（runtime-perf-v232） |
| prefill 单流 64K | **4084 tok/s** | UB 矩阵最优工作点 + GDN/QSA 算子分块（runtime-perf-v364） |
| MTP 接受率 | 96.5 / 90.8 / 86.5%（步长 3.74/4） | 真实衰减结构（v13 假接受案修复后） |
| 单流 decode | 115-131 tok/s | 8K 真实语料 |

Champion 二进制：[Release: inference-opt-champions-202610](https://github.com/Sak-114514/ac922-monster/releases/tag/inference-opt-champions-202610)

## 战役速览（2026-10-16 → 10-18，两天）

```
decode:  50.5 ─140.3─ 133 ─248.4─ 261.6 ─282.6   图组合→拍平→四bank→分片→定型
prefill: 1265 ────1926────── 2965 ────4022─4084   MTP分片→UB矩阵→算子分块
```

70+ 个运行时版本（V75→V310+）、9+ 个全模型验证轮、约 10 条内核探针线全部探针级验证。
完整原理、计划、结案报告见 [docs/CAMPAIGN-OVERVIEW.md](docs/CAMPAIGN-OVERVIEW.md)。

## 目录

- `src/` — 引擎源码（llama.cpp fork，monster-v0 裁剪谱系：852k→427k 行，仅保留 qwen4exp 架构，
  含 meta 双岛后端、图组合、四 bank 调度、MTP GPU 链、PLE 行缓存全套改动）
- `docs/CAMPAIGN-OVERVIEW.md` — 战役总览（成绩时间线 + 每项优化的原理）
- `docs/` — 设计文档（DESIGN-BLUEPRINT-v2 三路审核版、ENGINE-MASTER v1.1 含乱码案终审、
  ALL-EXPERIMENTS 实验台账、P3 蓝图、2→4 流计划、vLLM 路线评估）
- `docs/codex-notes/` — 优化会话全部滚动计划/结案文档（4500+ 文件）
- `docs/fixtures-records/` — 全部实验轮次的运行日志、判词、语义门结果、构建清单（4300+ 文件，
  含 runtime.log / unit-journal / results.json / supervisor-manifest / build manifest；
  二进制与大型遥测不在库，构建 sha256 清单可溯）

## 硬件与部署

- IBM Power System AC922（8335-GTW 水冷）：2× POWER9，6× V100-SXM2-16GB（NVLink 岛内 3 卡全互联，
  岛间激活交接），128GB DDR4-2666，CUDA 12.4 ppc64le
- 层流水双岛拓扑：前 24 层专家驻岛 0、后 24 层+输出+MTP 驻岛 1；岛内 TP3；PLE 大表主机 NUMA 近端
- 部署/安全：BMC night-3000 风扇守护（boot_guard，3000 转分层温控，勿动 phosphor-fan-monitor）
  见 [Sak-114514/super-ac922](https://github.com/Sak-114514/super-ac922) `ac922/operations/`

## 已知未决

- serial-vs-P4 生成 token 分叉案（五审未结，第 1 层入口差异已定位、内核已排除）
- 图组合接四 bank（等 fresh-exec 验收）、TurboPrefill 跨岛（形态漂移待适配）、relay 10MiB 全模型接入
- monster 树上游 #29381（MoE 融合核 FMA 收缩破坏投机精确性）补丁回植

## 许可

引擎源码继承 llama.cpp（MIT）；文档 CC-BY-4.0。
