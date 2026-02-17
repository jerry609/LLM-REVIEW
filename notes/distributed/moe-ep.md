# MoE Expert Parallel (EP)

## 什么是 Expert Parallel？
- 将不同 expert 放到不同 GPU
- 每个 token 根据路由结果 All-to-All 到对应 GPU
- 对应 GPU 运行 expert → All-to-All 回来

## 通信模式
```
Input tokens → Router → All-to-All (dispatch) → Expert computation → All-to-All (combine) → Output
```
- 每层 2 次 All-to-All 通信
- All-to-All 不同于 AllReduce：每个 GPU 向每个 GPU 发不同数据

## 通信量分析
- 每层 All-to-All 量 ≈ 2 × batch × seq × d_model × bytes / ep_degree
- 关键：All-to-All 延迟对小 batch 不友好（启动开销）

## 负载均衡
- 路由不均 → 部分 GPU 处理更多 token → 尾延迟
- 训练时加 aux loss 促进均衡
- 推理时：① capacity factor 限制每个 expert 最大 token 数；② 溢出 token drop 或 fallback 到 shared expert

## EP + TP 组合
- EP 处理 expert 间并行
- TP 处理 expert 内权重切分（当单个 expert 太大时）
- 例：Mixtral 8x7B，TP=4 + EP=2 → 每 GPU 放 4 个 expert 的 1/4

## DeepSeek-V3 方案
- 256 个小 expert + 1 个 shared expert
- EP=8 或 16，每个 GPU 放 16-32 个 expert
- 更细粒度 → 路由更灵活、负载更均衡

## 面试一句话
- "EP 用 All-to-All 通信实现 MoE 的 expert 级并行，核心挑战是负载均衡和 All-to-All 的延迟开销。"
