# V352候选：精确图快照跨分配UID候选检索（尚未实施）

## 目的
在V350提高draft prefill后，进一步降低目标图结构重复canonicalization成本。不修改专家数学、数据精度、target KV或图执行顺序。

## 现有证据
V338每卡flat_snapshot hits约2/misses134，signature_build约0.24–0.29秒/卡；六卡CPU总账不能直接相加为墙钟收益。缓存键包含调度每次分配产生的新UID，图的地址和结构若实际完全相同仍会miss。

## 优先最小切片
默认关闭的新开关仅允许同CUDA context+raw_key跨UID候选检索，候选必须经过完整当前节点/完整descriptor/当前src与view闭包及extra_state比较。通过后仅复用canonical signature；原本CUDAgraph lookup/readiness/update及同UID快速路径均不动。

## 风险与验收
- 不得解引用缓存指针。只从当前cgraph根及已经验证的当前父边访问后代。
- type/ne/nb/op/op_params/flags/name/padding/viewoffset/buffer/data/extra/extra_state及节点顺序均需一致。
- raw指针相同并不充分；不同CUDAcontext不能共享；不支持的external拒绝保留。
- 纯CPU fixture需要跨UID多轮、实际copy slot绑定轮转，逐项突变、外部边变更、释放后地址复用、容量和drop/context重建，输出signature与完整builder逐项一致。
- 若同raw地址无法复用导致hit仍低，再研究stable结构token+reserve epoch；不要先大改整个调度器。
- 真实GPU门包括原生语义、稳定graph replay/绑定路径日志，最终64K或四路实际吞吐；fixture通过不等于性能收益。

状态：只读设计，待当前V351结束与审查结果，不与模型运行同时重编。
