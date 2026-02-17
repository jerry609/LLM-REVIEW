# KV Cache 面试题 002：驱逐策略设计

## 题目
> 设计一个 KV Cache 驱逐策略，需要同时考虑命中率和多租户公平性。
> 说明你的设计思路、关键数据结构和评估指标。

## 参考答案（限时 10 分钟）

### 1. 问题分析
- **单租户**：LRU/LFU 各有优劣
  - LRU：对 temporal locality 好，实现简单
  - LFU：对 frequency-based workload 好，但有冷启动问题
- **多租户**：需要防止一个租户独占缓存

### 2. 方案设计

#### Adaptive + Quota-aware 策略

```python
class AdaptiveEviction:
    # 维护滑动窗口命中率和 Jain Fairness Index
    # 命中率低 -> 尝试切换策略（LRU <-> LFU）
    # 公平度低 -> 切换到 Fair 策略
    # cooldown 防止抖动
    
    def select_policy(self):
        if jains_fairness < threshold:
            return "fair"  # 某租户被饿死
        if hit_rate < target:
            return opposite_of(current)  # 试试另一个
        return current  # 保持
    
    def fair_evict(self):
        # 驱逐超配比例最高的租户的最冷 entry
        over_quota = {t: usage[t]/quota[t] for t in tenants}
        victim_tenant = max(over_quota, key=over_quota.get)
        evict_lru_from(victim_tenant)
```

### 3. 数据结构
- **Per-entry 元数据**：last_access_time, frequency, tenant_id
- **全局索引**：OrderedDict (LRU) + Counter-based min-heap (LFU)
- **Per-tenant 统计**：usage_blocks, hit_count, miss_count

### 4. 评估指标
- 总命中率 + Per-tenant 命中率
- Jain Fairness Index: `(sum hi)^2 / (N * sum hi^2)`
- 策略切换频率（应 < 5%）

### 5. 面试加分项
- 实际部署中的 **warm-up** 问题：新策略需要建立数据
- **hysteresis（滞后）**：切换有 cooldown，避免频繁切换开销
- **observability**：日志记录策略切换时间点和原因
