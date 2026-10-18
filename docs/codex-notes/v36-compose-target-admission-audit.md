# V36 clean P4：target 与 draft 的 nested-compose 准入审计

日期：2026-10-16  
范围：只读审计；未 build、未启动 GPU workload、未改 engine。

## 结论

- **V36 四路主窗口里 target 确实用了 composed graph，不是只有 draft 在 replay。** Target outer Meta context 0x7ffdc82bed30 的 49-fragment/48-AR6 graph 在 P4 中成功录制并反复 replay。Frame 58 附近日志已到 replays=64，frame 63 target step timing 有 rebuilt=0。
- **P4 计数边界值**：target 在 P4 结束时累计 84 replay、4 次 recording、32 declines；draft 在 P4 结束时 128 replay、6 次 recording、34 declines。计数来自各上下文独立的累计器，first single-8K-PP event 在 P4 complete 后读取 target 的 replays=84，draft 的最后一条 P4 event 是 replays=128。它们不是只覆盖稳定 64 帧的窗口计数。
- 整个 clean runtime 之后还运行了一次单路 8K PP probe；进程退出前 summary 是 target 96 / draft 155 replay（合计 251），不能把这组全程 summary 当成 P4 主窗口数字。
- **稳定四路、T16（frame 4–67，共 64 帧）的 host-wall 分布**：draft 阶段 median/mean 9.324/9.319 ms，target full-call median/mean 93.473/93.615 ms，whole-step median/mean 102.747/102.935 ms。均值约 9.1% draft、90.9% target；guard_wait=0。这些是 runner wall timers，不是 GPU kernel busy 时间。
- **单路 8K prefill 仍约 11.271 s**：target 9.932 s、MTP 1.338 s。这次 target prefill 期间 target Meta 多次 selected=0 phase=record；第一条 target 49/48 compose-ready 和 replay 出现在 prefill-lane-timing 之后，因此本次测得的 target prefill 没有用上该 composed graph。
- V36 的 selected=0 **没有逐条件拒绝原因**。declines 同时可由 compose replay 失败以及随后 rebuild 失败累计；不能把所有 decline 归为某个 CUDA graph node 类型。CUDA compose allowlist 对 host callback / alloc/free 及非 D2D memcpy 的拒绝，是待验证候选，不是本次运行已证实的 prefill 根因。

## 工件与配置

- Tag：root-p4-v36-v85-onebank4-ep6-d3-16k-20261016t091525z
- Runtime：runtime-perf-v85，build ID 20261016T085939.361444Z
- V36 runtime log SHA-256：573187014c4787f5ee45f09d412b4ce8c526623c857c016de0fa4431fc1ec405
- Results JSON SHA-256：5df73e5ec92987253bb70668d64196ce28b5861e9ab09f176bdd86de510d5a71
- Single-prefill measurement SHA-256：2e709a8065da6704930d6bb1e3674e8adb506d5b11dbe0a1840ec4f35cae3d82
- Runtime log：fixtures/dual-context-v36-combo-20261016/results/root-p4-v36-v85-onebank4-ep6-d3-16k-20261016t091525z/copy1/clean/runtime.log
- Results：同目录 results.json、single8k-prefill-measurement.json
- Results 标为 clean_performance_eligible=true；scenario semantic="unscored"，此报告不把该性能日志当语义门证据。
- 配置：one bank × 4 lanes、mtp_depth=3、target verify rows 16、islands 3,3、GGML_META_NESTED_ASYNC=1、GGML_META_NESTED_COMPOSE=1；GPU span/CUPTI 为 off。P4 scenario shared window 6.909493 s，shared 865 tokens，reported TG 125.1901 tok/s。
- 本报告不根据 relay6 kind/debug 行数推断缺 rank；V36 的 relay 日志有静态打印限额，root 已指出 rank0–3 的缺行属于该限额。

## Context 映射与图形状

bank-config（runtime.log:2094）将 llama target/draft 分别映射为：

| 路径 | llama context | outer Meta context | P4 composed graph |
|---|---|---|---|
| Target | 0x7ffedf0fb800 | 0x7ffdc82bed30 | 49 child fragments / 48 AR6 boundaries |
| Draft / MTP | 0x7ffc69ecc250 | 0x7ff8a5417f60 | 常见 1 child / 0 AR；另有 4 children / 3 AR6 |

Target 的 compose-ready 记录在 runtime.log:2560、4067、4526、5494；形状均是 ranks=6 children_per_rank=49 ar6_per_rank=48 expected_ar6=48。稳定四路阶段 frame 58 附近 runtime.log:4926 记录 target phase=replay ... replays=64。Frame 63 附近 runtime.log:4955 的 target step timer 为 n=96 rebuilt=0 tokens=16..16, inputs=1327 us, compute=21748 us, total=23122 us。

Draft 的 1-child/0-AR compose-ready 出现在 runtime.log:2762、3259、3749、4213；4-child/3-AR 形状出现在 4519、5502。其 P4 结束前累计 replays=128（最后一条 P4 draft compose 记录 runtime.log:5620）。Draft replay 数较高，但不能据此忽略 target：target 在同一 P4 steady window 有直接 replay 记录及其独立累计计数。

## P4 steady wall 时间统计

从 runtime.log 的 frame-submit/frame-complete 提取 frame 4–67，并只取 lanes=4 rows=16 的 64 个完整帧：

| Host wall timer | Min | Median | Mean | P90 | Max |
|---|---:|---:|---:|---:|---:|
| Draft draft_us | 9.103 ms | 9.324 ms | 9.319 ms | 9.429 ms | 9.533 ms |
| Target full_target_us | 92.872 ms | 93.473 ms | 93.615 ms | 94.216 ms | 97.929 ms |
| Full step full_step_us | 102.127 ms | 102.747 ms | 102.935 ms | 103.535 ms | 107.156 ms |

均值 draft + target 与 full-step 相符。该分解表示 runner 测得的调用阶段 wall time；本 run 没有 GPU-span/CUPTI trace，不能解释成 GPU 忙时或计算/通信占比。

## Compose decline 的证据边界

- 日志首个 phase=prepare 仅显示 private_cuda_api_available，证明私有 API resolve 成功，不代表某图可录制。
- phase=record selected=0 只写 source=validated_prepared_fragments 和累计计数，不写命中的拒绝 guard。
- Meta builder ggml_meta_nested_compose_build()（私有源 ggml/src/ggml-backend-meta.cpp:4558-4645）把多种检查折成 false：父/子 uid 与 generation、saved child fragment 完整性、binding signature、comm generation、每 rank chunk 存在、boundary 为六叶 contiguous F32 PARTIAL、AR reserve，以及 append child/AR/finish 成功等。
- outer dispatch 在 ggml/src/ggml-backend-meta.cpp:6196-6254 中，replay 失败和 build 不成功均增加同一个 declines；失败后继续普通 dispatch。
- CUDA child graph allowlist 位于 ggml/src/ggml-cuda/graph-compose-private.cuh:93-124：支持 kernel/empty/memset/event wait/event record、递归 child graph；memcpy 只接收 D2D；其他类型（包括 host callback 与 alloc/free）拒绝。Child 的 graph 查找、warmup、idle 超时和 signature 比较也可拒绝（同文件:157-184）。
- 因此，**当前 log 不足以区分**：target single-8K prefill 的多次拒绝是否由 graph node allowlist、fragment/signature/generation、boundary、reserve/append 还是 replay 失配导致。需要下一次非生产/有界诊断增加每个失败出口的枚举原因；不能只凭 selected=0 补推原因。

## Single-8K prefill

single8k-prefill-measurement.json 的范围是单请求、未命中 prompt/KV cache 的 8K prefill，另生成 32 token；总 prefill wall 11,270,597 us：

- Target prefill：9,931,859 us（target-only 口径 824.82 tok/s）
- MTP prefill：1,338,038 us
- Whole prefill：726.85 tok/s

runtime.log:5643 起 target context 在 prefill 期间对 generation 17–19 多次记录 selected=0，累计 replay 仍为 84；测量边界在 runtime.log:6112 prefill-lane-timing。之后 runtime.log:6293–6295 才出现 target 49/48 compose-ready、phase=record selected=1、phase=replay replays=85。故可确认的是：**这条 measured target prefill 没有 target composed replay**。它为何被拒绝，现有日志没有 guard-level reason。

Draft context 在 MTP 这部分确有 replay；但 target 占整次 prefill 时间约 88.1%。因为没有 GPU activity trace，不应把约 9.93 秒全部归因为 host launch 或 graph allowlist。

## 建议的下一步

先给 ggml_meta_nested_compose_build() / CUDA append_child 增加低开销、计数式拒绝原因（按 context、fragment 数和 compose attempt 聚合；不要逐节点刷日志），再用一个单路 8K prefill probe 定位 target 为什么直到 measured prefill 后才选中 49/48 compose。保持当前 selected=0 的普通路径 fallback；本报告不建议扩大模型测试。
