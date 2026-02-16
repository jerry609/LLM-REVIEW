# KV 驱逐策略的数学建模

## 1) LRU 基线
- 按最近访问时间驱逐最旧块。
- 优点：开销小（O(1) 操作）；缺点：不考虑"未来价值"。

## 2) LFU（最低频率）
- 按历史访问频次排序，驱逐频次最低的块。
- 优点：考虑热度；缺点：冷启动慢，对突发模式响应差。

## 3) 注意力感知驱逐（Attention-aware）
- 利用注意力权重作为重要性信号：
  `Importance_i = sum_{recent_steps} attention_weight(i)`
- 变体：H2O（Heavy Hitter Oracle）保留累计注意力最高的 token
- 窗口注意力（Sliding Window）+ 重要 token 保留的混合策略

## 4) 价值打分模型（通用框架）
- 统一分数：
  `Value_i = a*P_reuse_i + b*Cost_recompute_i + c*TenantWeight_i - d*Age_i`
- 驱逐规则：优先驱逐 `Value_i` 最小的块。
- 各系数可通过离线数据或在线学习调整。

## 5) 重算成本近似
- `Cost_recompute_i ≈ tokens_i * L * H_kv * d_head * (compute_cost_per_flop)`
- 简化口径：`Cost_recompute_i ∝ tokens_i`（层数和维度固定时）
- 也可加权"是否在关键前缀路径上"（prefix 共享场景）。

## 6) 命中率分析
- 理论最优（Belady's algorithm）：驱逐未来最晚被访问的块
  - 无法在线实现，但可作为离线上界
- 实际命中率：`hit_rate = 1 - miss_rate`
- 有效容量：`effective_capacity = physical_capacity * hit_rate`
  - 命中率低则等效于显存缩水

## 7) 公平性约束（多租户）
- 每租户配额：`quota_j`
- 软硬结合：`used_j <= quota_j + burst_j`
- 超额租户先做本租户内驱逐，再进入全局池。
- 加权公平队列：`priority_j = used_j / quota_j`，比值越高越先被驱逐。

## 8) 驱逐粒度选择
- Token 级驱逐：细粒度，但管理开销大
- Block 级驱逐（PagedAttention）：以 block 为单位（如 16 token），管理简单
- Layer 级驱逐：某些层对质量影响小，可选择性驱逐特定层的 KV
  - 研究发现：浅层和深层的 KV 可能比中间层更"可牺牲"

## 9) 评估目标
- 最大化：命中率 + 吞吐稳定性
- 最小化：重算率 + P99 + OOM
- 监控：`Refill Rate` 是误驱逐的重要信号。
- 驱逐开销：`eviction_overhead = eviction_freq * per_eviction_cost`

## 面试一句话
- "驱逐不是纯缓存问题，而是命中率、重算成本和多租户公平性的三目标优化。"
