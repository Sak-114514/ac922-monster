# AC922 post-release 生产运维

发布候选基于 L1/R444，固定 D3。生产版本目录为
`/home/debian/ac922-monster/production/releases/20261019-r444-banked-l2`。
只有 `lastwave-acceptance.json` 为 PASS 且 worker SHA 对应本次真实验收后才允许推广。

## 运行方式

- `ac922-inference.service`：四个单 lane target/MTP bank；后台 `127.0.0.1:8003`，由既有鉴权网关代理。
- `ac922-inference-safety.service`：沿用原传感器/风扇监测和故障锁存。
- `ac922-apigateway.service`：沿用既有 Bearer/X-API-Key，模型 `flash-next-iq3-xxs`。
- 总池 524288 cells，单请求 prompt+output 上限 262140 tokens，四路非对称共享；对齐和 idle bank 也占池空间。
- 文本 greedy temperature=0，max_tokens 上限8192，支持 chat、completions 与 SSE。
- 同一 bank/同一几何可复用完整前缀检查点；target recurrent state、draft 有效窗口及 pending hidden 同步保存/恢复。
- 每 bank 一个检查点，合计预算512MiB；不承诺跨 bank 命中。早段正文变化、context 重建或预算不足均冷算。
- 非流式响应通过 `usage.prompt_tokens_details.cached_tokens` 报告真实缓存命中。
- 空闲 bank 可在其他请求仍运行时接收适配当前容量的新请求；当前批次客户端全部结束后形成下一批次。

## 检查

```bash
ssh ac922 'systemctl is-active ac922-inference ac922-inference-safety ac922-apigateway'
ssh ac922 'readlink -f /home/debian/ac922-monster/production/current'
ssh ac922 'cat /run/ac922-inference/safety.json'
ssh ac922 'curl --noproxy "*" -sS http://127.0.0.1:8003/ready'
```

既有入口：`https://server.wuxuexi.top:8443/v1` 与
`https://ac922.tail58d5eb.ts.net/v1`。API key 在服务器
`/home/debian/apigateway/api_key`，此文档不保存密钥。

## 回到 L1

```bash
ssh -t ac922 'sudo python3 /home/debian/ac922-monster/production/backups/postrelease-20261018T172325/stage_lastwave.py --rollback /home/debian/ac922-monster/production/backups/postrelease-20261018T172325 --force'
```

回滚恢复 L1 current、原 systemd drop-in 和 8001 网关上游，停止新模型后重新加载 L1。
原温控/显存保护保持运行；若持久故障仍在，脚本保持推理停止，须先调查故障。

## 源码与编译边界

- `bank-worker-combo.cpp`、`lastwave-runner.h`、`bank-retention-policy.h`：bank 生命周期、独立推进、容量保留/回收与兼容补位。
- `prefix-checkpoint*.h`、`prefix-runtime.h`：有界、同 bank 的前缀检查点。
- `production-ipc.h`、`api.py`：补位握手、关闭边界、取消与队列、缓存 usage。
- `host-src/ggml/src/ggml-backend.cpp`、`host-src/src/llama-context.cpp`：每 context 固定调度几何，避免 global env 相互污染。
- `host-src/common/speculative.cpp`：固定 D3 的前缀 MTP sidecar。
- `build-postrelease.py`：复用L1全部引擎DSO，只重编worker单C++ TU；零CUDA/引擎TU重编。CPU回归与fixture主机编译并行。
- `stage_lastwave.py`、`verify_candidate.py`：备份、自动回滚窗口、哈希验证、冻结推广。

安全阈值延续 C12：GPU65°C；长prefill59°C暂停/54°C恢复；其余传感器原58°C；GPU余量最低256MiB。
大型多 bank context 使用 UB128 控制计算缓冲；一般四路 UB256，中等单流 UB1024、长单流 UB512。MTP 始终 D3。
本次不改变模型/专家数/量化格式、BMC、风扇、驱动、内核，不做整机重启。

## L2新增配置

`AC922_PREFIX_AFFINITY=1`：初始2–4请求精确token匹配及联合retention规划；singleton/refill不变。`AC922_CONCURRENT_SETUP=1`与`GGML_CUDA_ONECAT_INIT_READY=1`：通过六设备同时firstcapture门后并发执行每路保留的4个真实准备帧。实际host阶段有重叠，不宣称GPU kernel重叠。

L2构建入口为build-postrelease.py：全部7个引擎DSO和R5helper复用，只编译worker单C++TU。旧build-lastwave.py已从新候选移除。新源码bank-affinity-policy.h和bank-setup-guard.h及相应CPU测试保留。

资格验收进程无重载推广；冻结配置将AC922_LASTWAVE_RECORD_OUTPUTS置0，下一次正常启动使用冻结配置和独立可写run/postrelease-model目录。当前已加载进程沿用私有验收路径，须保留候选目录到正常服务重启后。不得直接清理当前映射库。
