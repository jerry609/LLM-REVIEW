# 分布式系统

> 这一组笔记把分布式推理拆成两条主线：一条是 TP / PP / EP 的部署权衡，另一条是 MoE 路由、容量限制和 All-to-All 通信。目录已经按 GitBook 连续阅读方式调整成“先 MoE 主线，再回到并行策略”。

## 最推荐的阅读顺序

1. [moe-formula-to-code-walkthrough.md](moe-formula-to-code-walkthrough.md)：先把 router、top-k、capacity、drop rate、All-to-All 公式和源码对上。
2. [../../math_dictionary/moe-routing-math.md](../../math_dictionary/moe-routing-math.md)：再看更完整的路由数学、负载均衡损失和通信分析。
3. [moe-ep.md](moe-ep.md)：把 Expert Parallel 和 All-to-All 放到系统视角里理解。
4. [moe-inference-deep.md](moe-inference-deep.md)：再看 Mixtral / DeepSeek-MoE 一类模型的真实设计取舍。
5. [tp-pp-tradeoff.md](tp-pp-tradeoff.md)：最后回到 TP / PP / EP 的组合部署。

## 这一组专题覆盖什么

- MoE 路由：softmax、top-k、归一化权重、expert load。
- 负载均衡：辅助损失、capacity factor、drop rate。
- 通信：dispatch / combine 两次 All-to-All 的数据量和尾延迟来源。
- 部署：EP 与 TP / PP 的组合，以及 shared expert 一类工程折中。

## 对应源码

- [../../src/simulators/moe_routing.py](../../src/simulators/moe_routing.py)：softmax、top-k router、辅助损失、capacity、dispatch、drop rate、All-to-All 字节量。

## 如果你只剩 20 分钟

- 先读 [moe-formula-to-code-walkthrough.md](moe-formula-to-code-walkthrough.md)
- 再看 [../../math_dictionary/moe-routing-math.md](../../math_dictionary/moe-routing-math.md) 里的辅助损失和通信量公式
- 最后对照 [../../src/simulators/moe_routing.py](../../src/simulators/moe_routing.py) 的 `topk_route()`、`load_balancing_loss()`、`dispatch_to_experts()`
