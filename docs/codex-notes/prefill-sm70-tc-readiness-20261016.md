# V100 SM70 MoE Tensor Core 路径就绪度审计

日期：2026-10-16  
对象：AC922 的 Qwen3.8-Flash-Next IQ3_XXS GGUF、V100 SM70、六卡 3+3／EP 路径  
结论类型：源码与现存工件审计；未改 core、未构建、未启动模型、未调用 GPU。

## 结论

1. V35 源码中的七种路由专家格式并未自动走 FP16 Tensor Core。最近可读的完整配置 proof 是 V31（其运行库实际为 V28）：在 EP=1、GGML_CUDA_MOE_GROUPED=1 的该环境中，量化专家 GEMM 走 grouped MMQ；V35 源码的 dispatch 仍是同一组条件，但 V35 manifest 本身没有 launch 环境，不能单凭 manifest 断言 V35 的实际 env。该 MMQ 路径为：F32 activation 先转为 Q8_1，量化权重在 tile loader 中解包，再以 DP4A 向量点积计算。V100 的 Compute Capability 为 7.0；本树的 MMQ MMA-layout gate 要求 Turing 7.5 或相应 AMD 指令路径。因此名字叫 “Ampere config” 的配置表不代表 V100 用 Tensor Core。
2. grouped MMQ 已经去掉大部分“空专家 block”发射，但每个 CTA 仍只计算单个专家的一个 live tile。Top-10 路由把 batch 分散到 512 个专家；ubatch 512/1024 时平均约 10/20 个路由 token/专家，和 J=32/64 的 tile 相比仍是小矩阵、填充较多。IQ 解包、Q8_1 转换、tile 元数据和每层多个 MoE 算子都保留在路径中。
3. 已有的 “先 dequant 到 F16、再跑 grouped MMF/Volta MMA” 局部实验已否定了最直接的方案：IQ3_XXS T=512 的 conversion 为 4.155 ms，MMF core 为 0.969 ms；完整转换+MMF 约 5.18 ms，而 grouped MMQ prep+core 约 1.11 ms，慢约 4.7 倍。不能把“有 Tensor Core”直接当成 prefill 提速证据。
4. 另一种保持量化码、以 Volta MMA 处理权重码的 streamed-TC fixture 覆盖七种格式和 1024–4096 token 路由，正确性门通过，但端到端仍比 grouped MMQ 慢 2.32–4.74 倍。现有两类 TC 路线都不应重复实现或并入生产。
5. V11 的旧 profile 显示，低 prefill 并非只由 MoE 算力解释：52.45 秒窗口内任一 GPU 活跃的 wall-union 为 20.17 秒，61.5% 窗口没有记录到 GPU 活动；旧 3+3 运行两岛 kernel 同时活跃为 0 秒。这个 trace 是 V10、旧 PLE 配置，不能直接代表 V35，但它说明单写 MoE TC 内核不够支撑 1,300 → 6,000 tok/s 的目标。下一优先级应以当前 runtime 的真实 8K trace 核实岛间重叠与 host/scheduler 空隙。

## 证据边界与当前版本

- 源码树：/home/debian/ac922-monster/experiments/standard-opt-20261014/src
- V35 运行包清单：/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v35/runtime-manifest.json；该清单给出二进制与库哈希，但不记录启动环境。
- 最近可读的完整 launch proof 是 results/v31-native-trace-20261015T140556Z/runtime-proof.json：其中 EP=1、岛为 3,3、GGML_CUDA_MOE_GROUPED=1、GGML_CUDA_MOE_ADAPTIVE_J=1、GGML_SM70_MOE_REWORK=1；但它指向 runtime-perf-v28，不是 V35。下文对“当前路径”的判断来自任务指定的 V35 源码；旧 trace/结果只用作历史证据。
- 现有 V35 event-query fixture 只验证 backend event query/re-record，不加载模型，不是 prefill 或 MoE 性能数据。
- GGUF 仅读 header/metadata 的 tensor descriptors；未访问 tensor payload。模型文件大小为 78,615,622,080 字节，GGUF v3，1,256 tensors，49 个模型 block，embedding=2,560，expert FFN=640，512 experts，top-k=10。

## V35 源码中的量化专家 dispatch

- src/ggml/src/ggml-cuda/mmq.cu:185-190 在 MUL_MAT_ID 量化路径中，若 GGML_CUDA_MOE_GROUPED=1 且 shape/type 支持，就先 dispatch grouped MMQ。
- src/ggml/src/ggml-cuda/mmq-moe-grouped.cu:71-86 显式支持 Q2_0、IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL，以及其他 GGUF 类型；当前模型实际使用的七种全部受支持。
- src/ggml/src/ggml-cuda/mmq-moe-grouped.cu:249-320 将路由 compaction、tile offsets、自适应 J 和 grouped launch 放在同一 CUDA 路径内。src/ggml/src/ggml-cuda/mmq-moe-grouped.cuh:3-18 说明它把普通 MMQ 中按专家预留、随后早退的大网格压缩成 live tiles；它没有把多个专家的工作融合成同一个矩阵 tile。
- src/ggml/src/ggml-cuda/mmq.cuh:188-201 只在 Turing MMA 或特定 AMD MMA/Wmma 时选择 MMA data layout；src/ggml/src/ggml-cuda/common.cuh:51-55、361-367 定义 Volta=700、Turing=750，并使 SM70 的 turing_mma_available=false。其余 quant IQ/Q 路径选 mmq_load_tiles_* + *_dp4a 向量点积和 DP4A writeback（mmq.cuh:544-669）。activation 由 mmq.cu:157-159 转为 Q8_1。
- src/ggml/src/ggml-cuda/mmq.cu:360-366 在 Volta 上对所有 MoE 批量偏向 MMQ，因为旧的 per-expert fallback 要 host sync 并对很多小专家矩阵逐个 GEMM。
- V31 环境里 GGML_CUDA_MOE_ADAPTIVE_J=1 已启用；自适应路径只在 CC=700、token≥512 时覆盖七种格式，候选 J=32/64（mmq-moe-grouped.cu:32-51）。已有配置已包含该快路径。
- GGML_SM70_MOE_REWORK 不是 prefill 的全专家 TC 解法：mmvq.cu:1318-1353 只为 Q2_0 down、K=192/256、非 EP MMVQ 形状生效；当前 EP offset 存在时源码打印“EP fallback”。LLAMA_MONOMOE_SM70 也不是 prefill 路径：qwen4exp.cpp:3045-3060 限制 n_tokens≤32，且现有配置为 0。

## GGUF 专家格式与实际字节

GGUF 描述中主模型有 48 层 routed expert；每层 gate、up、down 三个张量，共 144 个主专家张量。额外 blk.48 的三个 Q8_0 张量是 MTP block，不计入下面七种主专家格式。

gate/up 的 shape 为 [K=2560,N=640,E=512]，down 为 [K=640,N=2560,E=512]；每张含 838,860,800 个权重。下表的“次数”是 48 层中使用该格式的主专家张量数，“每张大小”基于 GGML_QUANT_SIZES 算出，未读 payload。

| 格式 | block | 使用矩阵 / 次数 | 每张量化权重大小 | 48 层该格式累计 |
|---|---:|---|---:|---:|
| IQ2_XXS | 256 / 66 B | gate 9 + up 9 | 206.25 MiB | 3.6255 GiB |
| IQ2_XS | 256 / 74 B | gate 10 + up 10 | 231.25 MiB | 4.5166 GiB |
| IQ2_S | 256 / 82 B | gate 10 + up 10 | 256.25 MiB | 5.0049 GiB |
| IQ3_XXS | 256 / 98 B | gate 6 + up 6 | 306.25 MiB | 3.5889 GiB |
| IQ3_S | 256 / 110 B | gate 13 + up 13 | 343.75 MiB | 8.7280 GiB |
| IQ4_NL | 32 / 18 B | down 18 | 450.00 MiB | 7.9102 GiB |
| Q2_0 | 64 / 18 B | down 30 | 225.00 MiB | 6.5918 GiB |

48 层主 routed expert 张量共约 39.966 GiB 量化权重。每层总量为 637.5–1,137.5 MiB，平均 852.6 MiB。示例 blk.0 为 IQ3_XXS gate/up 各 306.25 MiB、IQ4_NL down 450 MiB，合计 1,062.5 MiB。

## TP3 / EP3 本地 shape 和 F16 staging 容量

源码 llama-model.cpp:341-353 使用常规 TP 时将 gate/up 按 axis 1（N）切分、down 按 axis 0（K）切分。分段粒度从 down weight block size 推导（586-604）：

- down=Q2_0 时，粒度 64，TP3 的 N/K 三片为 192/192/256。
- down=IQ4_NL 时，粒度 32，TP3 的 N/K 三片为 192/224/224。
- 因而 gate/up 本地 shape 是 [2560, N_local, 512]；down 本地 shape 是 [K_local, 2560, 512]。另一矩阵维度保持完整；层间的 remainder 会轮转。

当 LLAMA_EXPERT_PARALLEL=1 时，routed expert weights 走 axis2（llama-model.cpp:134-135、330-339），每个 TP3 rank 拿完整 K/N、E_local=170/171/171；shared/dense expert 保留 TP placement。故V31 proof 所示 EP=1 环境下，routed experts 实际不是 gate/up 的 N 切片或 down 的 K 切片。两套 shape 不要混为一谈。

FP16 staging 体积（一个元素 2 B）：

| 布局 | 每 rank 单个 gate/up/down F16 矩阵 | 若一层三矩阵同时驻留 |
|---|---:|---:|
| EP3，本地 E=170/171 | 0.519 / 0.522 GiB | 1.556 / 1.566 GiB |
| TP3，片长 192 | 0.469 GiB | 1.406 GiB |
| TP3，片长 224 | 0.547 GiB | 1.641 GiB |
| TP3，片长 256 | 0.625 GiB | 1.875 GiB |

如果按 gate → up → down 的串行执行顺序复用同一个 scratch，FP16 workspace 的理论下限是上表“单个矩阵”一列，而非三矩阵总和；这要求 graph 的依赖/lifetime 真正允许复用，并且不能同时让多个 frame/stream 共用未完成的 scratch。若双缓冲或多个 CUDA stream 同时做 dequant+MMF，应按同时活跃矩阵数乘算。没有 V35 模型运行时的每卡峰值 free-space 数据，不能判定哪一项能安全分配。

## 现存 MMF / Tensor Core 路线：已测，不重做

现有函数可以构成试验原型，但当前 dispatch 不满足 V35 EP prefill：

- convert.cu:629-675 的 ggml_get_to_fp16_cuda(type) 包括这七种格式；mmf.cu:112-118 有 F16 MUL_MAT_ID 分支；mmf.cuh:221-226 使用 mma tile。V100 F16 TC 是可调用的硬件能力。
- 但 mmf.cu:133-137 拒绝 quantized source；mmf.cu:162-167 对 Qwen expert N=640 的 MUL_MAT_ID batch>512 返回 false，对 down N=2560 的 batch>128 返回 false。V31 已存档 launch 的 ubatch=1024 不会自然进入 MMF。
- mmf.cu:64-88 的设备 ID compaction 调用 mm_ids_helper 时 expert_offset 固定为 0；EP rank 拿的是全局 ID 对应的局部 expert slice，因此当前 MMF 还缺 EP-aware offset/foreign-zeroing。仅转 F16 不足以保持 EP 正确。
- cuBLAS 常规 F16 MUL_MAT 是另一条 dense 路径；当前没有直接接受 global expert IDs 并完成 EP local mapping 的 grouped cuBLAS MUL_MAT_ID 路径。已有可复用的 grouped F16 TC 实现是 MMF，而不是 cuBLAS grouped GEMM。

此前已经完成的实验，不要重跑：

1. notes/prefill-mmf-trial-results.md：真实 IQ3_XXS gate T=512，F16 dequant 4.155 ms，grouped MMF 0.969 ms；加上转换约 5.18 ms，MMQ prep+core 约 1.11 ms，完整转换+MMF 慢约 4.7×。IQ4_NL down T=32 grouped MMF core 在 15 s 诊断看门狗内未完成；此形状结果不确定，不能说成 correctness bug。
2. notes/streamed-tc-iq-large-route-20261015-results.md：保持真实量化权重、streamed decode + Volta MMA；七格式 fixture 路由/数值门通过，但 1024/2048/4096 token 下，TC 相比 grouped MMQ 端到端慢 2.32–4.74×。不应将现有 streamed-TC 代码合入。
3. notes/volta-tc-mmq-perf.log、volta-tc-mmq-reg-perf.log、volta-tc-mmq-magic-perf.log：多个单专家 J=16 小矩阵的 Volta integer-TC 原型仍慢于 DP4A；不同原型版本实测慢约 1.4–4×，没有全模型或 PP 结果。
4. notes/VOLTA-INTEGER-TC-DESIGN.md 是设计草案，不是生产实现，也无模型收益证明。

所以“复用已有 converter+MMF 避免新写 CUDA kernel”在功能上可行作 prototype，但在速度、当前 EP 支持、1024-token 形状、workspace 和质量门上均未通过。不能据硬件理论峰值判断会提高模型 PP。

## 当前真正的 PP 利用率证据

- results/v11-full-trace-20261015T052734Z 为旧 runtime-perf-v10、四路 8K、ubatch=1024、旧 lazy PLE 的 profile；不是 V35/V24/V31 结论。
- notes/prefill-critical-path-audit-20261015.md 的 interval union：52.45s 窗口中 20.168s 有任意 GPU kernel/copy 活动，32.281s（61.5%）没有 GPU 活动；两座 3-GPU 岛的 kernel 同时活跃为 0 ms，所有 GPU activity 的岛间重叠仅 146 ms。 grouped MoE kernel union 5.328s；P2P AR union 5.889s。各 category 不能简单相加。
- 同一旧 trace 的 graph compute wall span 约 890.951 ms/call ×32，接近无 GPU 活动预算；但该计时不能单独证明具体是 PLE GET_ROWS。旧 V11 PLE 是 lazy GGUF 路径，因此不能重复宣称“当前 V35 只是缺少 PLE prefault”。
- 现有 V35 event-query fixture 没有模型/PP，不能证明当前 frame pipeline 已让两个岛并发工作。应基于当前 V35 实际 8K timeline 再判断；本审计没有新 profile。

## 建议给主线的决定

- 不要再实现/测试同一类 F16 dequant+MMF 或 streamed-TC kernel；已有测试已否定它们对整层或模型 PP 的性能预期。
- 当前 PP 可继续调查的是 scheduler/frame pipeline 的真实岛间 overlap、CPU 侧非阻塞输入/GET_ROWS 与各类 host wait；MoE 代码上的小块余量可视化为 grouped tile fill/active experts/J 分布，而不是直接把 Tensor Core 当答案。
- 如果提出全新 TC 方向，必须在动 core 前先证明它不是以上旧原型的重复；profile 需把 dequant、路由准备、MMQ/MMF kernel 时间分开，使用 V35 实际二进制和启动环境。所有质量门需对 full logits/token/语义做验证；现有单矩阵 fixture 不替代模型质量验证。

## 主要源码与工件

- GGUF metadata：model IQ3_XXS-MTP-Q8.gguf；只解析 GGUF v3 header/tensor infos。
- 结构/shape：src/src/models/qwen4exp.cpp:217-218,272-279；src/src/llama-model.cpp:130-135,330-353,386-390,586-604。
- MMQ：src/ggml/src/ggml-cuda/mmq.cu:185-200,360-366；mmq-moe-grouped.cu:32-86,249-320；mmq-moe-grouped.cuh:3-18,91-122；mmq.cuh:188-201,398-424,544-669；common.cuh:51-55,361-367。
- F16 conversion/MMF：convert.cu:629-675；mmf.cu:64-88,133-189；mmf.cuh:221-226。
- Existing no-repeat results: notes/prefill-mmf-trial-results.md; notes/streamed-tc-iq-large-route-20261015-results.md; notes/prefill-critical-path-audit-20261015.md.
- Runtime manifests/proofs: runtime-perf-v35/runtime-manifest.json; results/v31-native-trace-20261015T140556Z/runtime-proof.json.

