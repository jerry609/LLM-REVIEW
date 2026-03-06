# 从公式到源码：KV Eviction / LRU / LFU / Fair Quota 对照手册

> 这一页把 KV 驱逐问题写成一个明确的“排序问题”：预算不够时，系统必须给每条序列一个保留优先级，然后淘汰价值最低的候选项。不同策略的区别，本质上就是评分函数不同。

## 这页覆盖哪些源码

- [../../src/kv_cache/eviction/policies.py](../../src/kv_cache/eviction/policies.py)：`LRUPolicy`、`LFUPolicy`、`FairPolicy`。
- [../../src/kv_cache/core.py](../../src/kv_cache/core.py)：`last_access_step`、`use_count`、`num_blocks()` 等元数据来源。

## 1. 驱逐策略的最小抽象：先定义受害者函数

当 KV 预算不足时，系统需要从候选集合 $\mathcal{S}$ 里选出一个受害者：

$$
\text{victim} = \arg\min_{s \in \mathcal{S}} \text{Value}(s)
$$

或者等价地写成某种“驱逐优先分数”的最大者：

$$
\text{victim} = \arg\max_{s \in \mathcal{S}} \text{EvictScore}(s)
$$

代码里不同 policy 的差别，就体现在 `select_victim()` 的 key 如何定义。

## 2. LRU：最近最少使用为什么只看最后访问时间

### 2.1 公式

LRU 的目标函数最简单：谁最久没被访问，谁最先淘汰。

$$
\text{victim}_{\text{LRU}} = \arg\min_s \; t_{\text{last\_access}}(s)
$$

### 2.2 对应源码

[../../src/kv_cache/eviction/policies.py](../../src/kv_cache/eviction/policies.py) 的 `LRUPolicy.select_victim()`：

```python
def select_victim(self, sequences: Dict[str, SequenceKVCache]) -> Optional[str]:
    if not sequences:
        return None
    return min(sequences.keys(), key=lambda k: (sequences[k].last_access_step, k))
```

这里的 `last_access_step` 由 [../../src/kv_cache/core.py](../../src/kv_cache/core.py) 在 `allocate_for_sequence()`、`append_tokens()`、`fork()` 等路径中更新。

也就是说，LRU 依赖的不是 token 内容，而是“最近有没有被碰过”。

## 3. LFU：为什么要同时看 use_count 和 last_access_step

### 3.1 公式

LFU 的想法是“历史上被访问更少的序列更该淘汰”：

$$
\text{victim}_{\text{LFU}} = \arg\min_s \; \text{use\_count}(s)
$$

但如果两个候选者访问次数一样，就需要一个 tie-break，于是常见实现会再拼上 LRU：

$$
\text{victim}_{\text{LFU}} = \arg\min_s \; \big(\text{use\_count}(s), t_{\text{last\_access}}(s)\big)
$$

### 3.2 对应源码

```python
def select_victim(self, sequences: Dict[str, SequenceKVCache]) -> Optional[str]:
    if not sequences:
        return None
    return min(
        sequences.keys(),
        key=lambda k: (sequences[k].use_count, sequences[k].last_access_step, k),
    )
```

这说明 `LFUPolicy` 并不是“纯频率论”，而是“先看频率，再看最近性”。

## 4. Fair quota：多租户场景为什么要先公平再最旧

### 4.1 配额公式

若租户 $t$ 的权重为 $w_t$，总 block 预算为 $B_{\text{total}}$，则它的理论配额是

$$
\text{quota}_t = \frac{w_t}{\sum_j w_j} \times B_{\text{total}}
$$

这在源码里对应 `FairPolicy._quotas()`：

```python
def _quotas(self, sequences: Dict[str, SequenceKVCache]) -> Dict[str, float]:
    tenants = set(self.tenant_weights.keys())
    for sid in sequences:
        tenants.add(self._tenant_of(sid))
    weights = {t: float(self.tenant_weights.get(t, 1.0)) for t in tenants}
    total_w = sum(weights.values())
    return {t: (w / total_w) * self.total_blocks for t, w in weights.items()}
```

### 4.2 超配租户的定义

若某租户当前占用为 $u_t$，则它的超配量是

$$
\text{overuse}_t = u_t - \text{quota}_t
$$

只有当

$$
\text{overuse}_t > 0
$$

时，这个租户才会进入“优先回收”集合。

源码对应：

```python
usage: Dict[str, int] = {}
for sid, seq in sequences.items():
    t = self._tenant_of(sid)
    usage[t] = usage.get(t, 0) + seq.num_blocks()

quotas = self._quotas(sequences)
over = [t for t in usage if usage[t] > quotas.get(t, 0.0)]
```

### 4.3 为什么最终还是回到租户内 LRU

Fair policy 的逻辑不是“直接全局最旧”，而是两层选择：

1. 先在超配租户中找超配最多者。
2. 再在该租户内选最旧序列。

数学上可以理解为

$$
\text{victim\_tenant} = \arg\max_t \; \max(0, u_t - \text{quota}_t)
$$

$$
\text{victim} = \arg\min_{s \in \mathcal{S}_{\text{victim\_tenant}}} \; t_{\text{last\_access}}(s)
$$

源码正对应这两步：

```python
if over:
    victim_t = max(over, key=lambda t: (usage[t] - quotas.get(t, 0.0), t))
    candidates = [sid for sid in sequences if self._tenant_of(sid) == victim_t]
    return min(candidates, key=lambda k: (sequences[k].last_access_step, k))

return min(sequences.keys(), key=lambda k: (sequences[k].last_access_step, k))
```

所以 Fair policy 的核心语义是：先保证跨租户公平，再在租户内部维持简单稳定的 LRU。

## 5. 驱逐策略为什么依赖 `core.py` 的元数据维护

驱逐策略本身不负责更新访问统计，它只消费这些统计。真正更新 `last_access_step` 和 `use_count` 的位置在 [../../src/kv_cache/core.py](../../src/kv_cache/core.py)：

```python
self._step += 1
seq.last_access_step = self._step
seq.use_count += 1
```

这意味着：

- 如果访问统计更新不及时，LRU / LFU 都会失真。
- 驱逐层和缓存管理层是分工关系，不是彼此独立的两套系统。
- `num_blocks()` 让 Fair policy 能按 block 占用而不是按请求条数计费，更符合真实显存压力。

## 6. 建议的源码阅读顺序

1. 先读 [../../math_dictionary/kv-eviction-math.md](../../math_dictionary/kv-eviction-math.md)，把驱逐目标、重算代价和公平约束的数学抽象建立起来。
2. 再读 [../../src/kv_cache/core.py](../../src/kv_cache/core.py)，先搞清楚策略依赖的元数据是怎么维护的。
3. 接着读 [../../src/kv_cache/eviction/policies.py](../../src/kv_cache/eviction/policies.py)，对照三种 `select_victim()` 的 key。
4. 最后跑 [../../tests/test_kv_cache.py](../../tests/test_kv_cache.py)，把生命周期和元数据更新逻辑验证一遍。

## 这一页记住一句话

> KV Eviction 的本质是给“保留价值”排序：LRU 用时间排序，LFU 用频率排序，Fair quota 先按租户预算排序再按时间排序。你能不能解释清楚排序函数，基本就决定了这题答得深不深。
