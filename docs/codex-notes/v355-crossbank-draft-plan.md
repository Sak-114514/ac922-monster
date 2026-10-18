# V355 跨 bank 合批 draft：准备阶段，未接入/构建

保留最快4bank独立target LAYER6拓扑，集中一个GPU5四seq MTP context，复用fold_multi的per-seq carry/history/accept处理。禁止把未通过的target多seq路径带入本实验。

最小core适配器仅在一次process调用期间从指定target读取原顺序hidden行：输入batch只把local seq0重映射到该bank独占global0..3，pos与行顺序不变，RAII恢复target来源。仅sole MTP /4seq/fold_multi/nonshared memory/window0允许。原public API及默认执行路径不动。

容量必须自行建立draft总ctx68608、每seq17152、独立KV/GDN/SSM状态；不能使用把ctx_total覆盖成target单路17152的旧init helper。Qwen ctx_other不会共享target0 memory；仍需实际断言。

协调器必须是唯一draft owner；接收process/accept/rollback/next-draft消息，保持每lane request epoch和frame顺序。目标bank线程不得直接碰共享draft。按ready lane合批1..4，不能为凑四路永久等待已结束lane；deadline受整体守卫约束。每次privateprocess的source target已经同步且所有h_nextn保持有效。

局部门：路由/epoch/容量CPU fixtures；随后真实4路hidden对4×single参考的D3、accept/rollback/SSMconv、错位和reset检查。token差异要报告来源与质量，不把浮点顺序差异自动判坏；任何跨lane污染必须修。局部通过再接原native4和4×16K512整模型。

当前只有候选适配器和CPU协议fixture准备；没有模型收益，不能算完成多路合批。
