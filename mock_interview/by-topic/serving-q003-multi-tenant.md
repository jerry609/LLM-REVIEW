# 面试题：多租户 LLM Serving 的公平性保证

## 题目
"你的 LLM 推理平台服务多个客户，大客户流量是小客户的 100 倍。如何保证小客户的 SLO？"

## 答题思路

### 问题定义
- 大客户流量大 → 占满 GPU / KV Cache → 小客户排队 / 被驱逐
- 公平性指标：Jain fairness index = (Σx)² / (n·Σx²)

### 多层解决方案

#### 1. 请求层：Rate Limiting
- 每个租户独立 token bucket
- 超限请求排队或 reject
- 允许短暂 burst（burst bucket）

#### 2. 调度层：Weighted Fair Queuing
- 调度器按租户权重分配 GPU 时间片
- 小租户的请求优先级提升
- 最大等待时间兜底（防止饿死）

#### 3. 缓存层：配额驱逐
```
每租户 KV 配额 = tenant_weight / total_weight × total_blocks
超配租户优先被驱逐
```
- Adaptive 策略：fairness < 阈值 → 自动切到 Fair 驱逐

#### 4. 资源层：物理隔离
- 关键客户独占 GPU pool（最强保证，最贵）
- 或虚拟隔离：NVIDIA MIG / time-sharing

### 监控
- 每租户独立的 TTFT/TPOT/命中率指标
- Jain fairness index 实时监控
- 告警：任何租户 SLO 违规率 > 5%

## 面试一句话
- "多租户公平性需要请求层限流 + 调度层加权 + 缓存层配额 + 资源层隔离，四层递进。"
