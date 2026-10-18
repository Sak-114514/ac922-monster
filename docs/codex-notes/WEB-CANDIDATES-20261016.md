# AC922 批量联网候选筛选 2026-10-16

本机约束：POWER9，6×SM70 V100 16GiB，3+3 EP/TP；CPU2.3GHz/GPU180W；原始IQ3_XXS GGUF七格式专家保留，q8/q8 KV。目标300+真实多路decode /6000+单请求uncached PP，未达。

| 原始项目/资料 | 可借机制 | 本机判定/下一动作 |
|---|---|---|
| [TurboPrefill](https://github.com/sergey-automation/TurboPrefill) v6.8/v6.9.1 | 单请求UB跨stage流水、较小UB、backpressure | 现有V41已接实际CPU→Meta3→Meta3，检查真实selected+token exact；UB512在途，128/256只在适配真实workspace后排队。README示例不能当本机倍率。 |
| [anyei llama V100](https://github.com/anyei/llamacpp-v100) | prefill stream ring、small-width FA、SM70小批量GEMM | 代理逐项读代码对照已有kernel。MoE doorbell为CPU miss专家，本机主专家全GPU不照搬；AVX2不适用POWER9。 |
| [1Cat #966](https://github.com/1CatAI/1Cat-vLLM/pull/966) | 原始IQ3_XXS/IQ3_S gate/up共享激活、FP32 dot | 已合并；实测shape M8/N4352/K5120 与本机EP全K2560/N640不符，不能直接套论文倍率或替换成FP4。仅机制候选，需要shape级fixture。 |
| [1Cat #754](https://github.com/1CatAI/1Cat-vLLM/pull/754) | 只在真正保留的Mamba checkpoint处分chunk | 32K零hit改进是其自身vLLM调度对照；本机checkpoint0、真正KV+GDN snapshot语义不同，先读chunk/trim代码确认是否还有无用边界；warm reused-token表不算uncached目标。 |
| [1Cat #957](https://github.com/1CatAI/1Cat-vLLM/pull/957) | fuse allreduce+RMSNorm减少同步 | 其TP4环拓扑不直接套3卡岛；需本机AR具体tensor/bytes和图边界证据后移植。 |
| [Strata V100 DETAILS](https://github.com/jmnargi/Strata-V100/blob/main/docs/DETAILS.md) | compact专家输入、两kernel HC window、lifetime复用 | 现有HC/EP/grouped先对照；新fused prompt expert标明sm80+、cluster decode sm90+，不入SM70队列；control vector改残差会改模型质量，排除。 |

持续证据：V39原失败4请求在target/fenced与adaptive8 exact，但12 async首波失败；不能定罪peer fanout，因为policy2既同步又关闭它，且先后顺序混杂。V41现用rows6、UB512交叉冷/热验证中。rows4局部mixedcatchup full-logit失败，明确不启用。

## 已执行筛选增量
- 小UB(128/256/512)实际流水selected、输出exact但未超过普通UB1024；warm各档数字已在V43-ubatch-summary.json，不继续盲调UB。
- 六卡dense完整复制按header+V42实测得17.18GiB/card，拒绝该布局；单Meta6+TP6 dense+EP6 weights另一路沿现有loader可装，当前已启动短语义/512候选测试（非全dense复制）。
- 两版已装Nsight2023.2/2024.2 tiny graph+跨岛copy都产生可读SQL，不能把官方PPC停支持直接定成collector EOF根因；fullmodel warm后interactive start掉agent原日志保留。下一次有意义的采集差异应在CUDA graph构造前启动capture，不再仅更换版本反复整模型加载。
- GPU矩阵无额外量化/专家裁剪。Pooled CELLSEL静态审计发现漏partial-tail +1e9且tie物理ID不一致，旧开关不直接全开；新physical scatter+失效ledger仅staging待GPU门。

## 新批量筛选（03:25）
- https://github.com/lukolszewski/llama.cpp-multigpu : 同qwen4exp六卡47patch，layer流水+decode groups；README高倍率主要3090/4090长ctx相对stock，且不含MTP，不能外推SM70/EP3；已分配只读源码对照，挑缺项增量移植，不换掉已有专用kernel。
- https://github.com/sgl-project/sglang/issues/29960 : target verify拆两CUDA graph以遮host launch。是proposal/issue不是已验证本机收益；当前Meta3+3已有图段/P4 owner pipeline，先用真实event span看host是否被遮住。
- Turbo pin预算：UB1024普通当前更快，但256MiB总pin cap拒流水；准备受控显式预算候选，保留实际combined检查/硬上限/DDR余量，不因读取全局含HBMfree误放开。

## 第二轮批量筛选（UTC 21:05）
- vLLM DBO：官方 docs/design/dbo.md 明确 DP+EP、microbatch token pad一致和完整双批CUDA graph；可借调度与图捕获，不把DeepEP作为V100即装即用依赖。https://github.com/vllm-project/vllm/blob/main/docs/design/dbo.md
- SGLang DPA+EP：普通none backend以AllGather/AllReduce做dispatch；可作为SM70原语路径参照，先核算按请求持有KV/状态而非每GPU复制4流。https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx
- PPLX/新DeepGEMM/SM90专核不入SM70执行队列；不要借Hopper宣传数值预测本机。
- Monpack真实现存算子：7format×T1/4/8，19/21 bitwise；IQ2_S T4/T8 maxabs1.1920929e-7，严格门exit1留存。尚不能启生产EP，axis2局部ID/foreignmask/TC归属需适配，且成功dispatch标记需补。
- V50设备端ordinary QSA bias仅prefill>=128 optin；单CUDA/Meta3各5case exact+masked+topK过，V14大模型32native过但四路prefill139；V15 ordinary UB1024先prefill1 trace再并发定位，不定罪kernel。

## 第三轮批量筛选（UTC22:30后；执行中）
- NVIDIA graph launch机制复核：https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/ 。其CUDA12.6近似常数launch实测针对Ampere；不能外推本机SM70，也不为此改驱动/工具链。继续现有图重用/CPU签名优化。
- 上游 async CPU→CUDA拷贝/减少同步：https://github.com/ggml-org/llama.cpp/commit/2cd20b72ed3565ac6935911ca0d9b5d73ae70d0d 。当前私有树已有batch_input_copies和FIFO pinned snapshot，不能重复移植后声称新增。代理逐条审计Meta fallback及同步public契约，只有闭合生命周期的新缺项入场。
- mx fork最新调度说明：https://github.com/mxxm-t/mx-llama.cpp/blob/master/FEATURES.md 。借鉴 pinned输入容量几何增长、allocation布局缓存、长prefill归约交NCCL；其gfx906倍率不作SM70预测。当前snapshot_input确实按size精确扩容，需先查实际HostAlloc频率和容量预算；当前AR64保留decode图路径，NCCL需强制F32，同图局部fixture后才可选择prefill路由。
- Strata V100近期多卡说明：https://github.com/jmnargi/Strata-V100/blob/main/docs/MULTI_GPU.md 。CPU AVX2核不适用POWER9；MTP speculative window改变接受率，不能由短样本TG直接判质量/速度。当前原GGUF/IQ3核不改精度。
- 已启动下一组合V19准备：V56 wholeclosure、独立KV每路64K、MTPub512、AR64、LRU64entries；保留全F32logit/semantic/exact-token gates。V18 perseq128K触发显存观察线，不定罪精度或温控。

## 第四轮批量筛选（UTC00:10；执行）
- SGLang DPA+EP官方说明 https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/dp_dpa_smg_guide.mdx ：按request拥有attention KV，MoE dispatch/combine；此处采用DPA2×TP3 dense+EP6而非6份dense，不照搬DeepEP/SM90。V59 nested6归约小门已过，loader mixedplacement审计进行。
- Hybrid P/D https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg ：SSM/R/S/conv和KV必须各自完整转移，不能仅拷贝attentionKV。现fork同布局prefill/decode context可先利用完整seq-state，但还需logits/nextn hidden sidecar和n_stream匹配；不能把两阶段吞吐相加。
- state API官方当前memory-only接口核实，非unified n_seq1→2不兼容；候选prefill n_seq2只active1匹配decode2，先查私有fork/history，不重复大UB死路。
