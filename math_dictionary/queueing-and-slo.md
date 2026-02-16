# 排队论与 SLO 速查

## 1) Little 定律
- `L = lambda * W`
- 含义：系统内平均请求数 = 到达率 × 平均逗留时间。
- 普适性：对任何稳态排队系统成立，无需假设到达分布。

## 2) M/M/1 基础
- 假设：泊松到达、指数服务时间、单服务器
- 利用率：`rho = lambda / mu`（要求 `rho < 1`）
- 平均响应时间：`W = 1 / (mu - lambda)`
- 平均排队时间：`W_q = rho / (mu - lambda)`
- 系统中平均请求数：`L = rho / (1 - rho)`

## 3) M/M/c（多服务器）
- `c` 台服务器并行处理
- 利用率：`rho = lambda / (c * mu)`
- Erlang C 公式给出排队概率（公式较复杂，面试给结论即可）：
  - 当 `rho` 接近 1 时，即使多服务器也会排队暴增
- 工程含义：多 GPU/多实例 ≈ 多服务器，但负载均衡质量决定实际效果

## 4) M/G/1（通用服务时间）
- Pollaczek-Khinchine 公式：
  `W_q = (rho * (1 + C_s^2)) / (2 * (1 - rho) * mu)`
- `C_s = sigma_s / E[s]`：服务时间的变异系数
- 关键结论：**服务时间方差越大，排队越长**
- LLM 场景：输出长度差异大 → `C_s` 高 → 排队恶化
  - 这是为什么输出长度预测和请求调度很重要

## 5) 尾延迟分析
- P99 ≠ 均值的简单倍数
- 对 M/M/1：`P(W > t) = rho * exp(-(mu-lambda)*t)`
  - P99：`t_99 = -ln(0.01*rho) / (mu - lambda)`（当 `rho` 可忽略时近似）
- 当 `rho` 从 0.5 → 0.9，P99 可膨胀 5-10×
- 实际系统的尾延迟通常比理论模型更差（因为有 GC、通信抖动等）

## 6) 工程解释
- 当 `rho` 接近 1，延迟会非线性爆炸。
- 面试常见结论：宁可轻微降吞吐，也要远离高利用率危险区。
- 经验法则：`rho < 0.7` 延迟可控，`rho > 0.85` 进入危险区。

## 7) SLO 预算分解
- 例：端到端 P99 预算 2s，可拆成
  - 排队 300ms
  - prefill 700ms
  - decode+网络 1000ms
- 任何子项超预算都应触发降级或限流。

## 8) 自适应限流策略
- Token bucket：控制平均速率和突发
  - `rate`（token 补充速率）、`burst`（桶容量）
- 基于队列深度：
  - `if queue_depth > threshold: reject/delay new requests`
- 基于 KV 利用率：
  - `if kv_utilization > 90%: reduce admission rate`
- 基于 SLO 余量：
  - 动态调整准入阈值，使 P99 保持在预算内

## 9) 优先级调度
- 多级优先级队列：VIP 请求优先调度
- Shortest-Job-First（SJF）近似：预测输出长度，短请求先服务
  - 可降低平均延迟，但可能饿死长请求
  - 实践中用 aging 机制防饥饿：`effective_priority = base_priority + wait_time * aging_factor`

## 面试一句话
- "LLM 推理的排队抖动常被低估：输出长度方差高 → 服务时间方差大 → 尾延迟爆炸。"
