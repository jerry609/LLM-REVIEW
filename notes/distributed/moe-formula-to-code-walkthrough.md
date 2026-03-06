# 从公式到源码：MoE Router / Capacity / All-to-All 对照手册

> 这一页只做一件事：把 MoE 推理里最核心的四个对象连起来——router 概率、top-k 选择、capacity 限制、All-to-All 通信。读完之后，`moe_routing.py` 里的每个函数都能对应到一条明确的公式。

## 这页覆盖哪些源码

- [../../src/simulators/moe_routing.py](../../src/simulators/moe_routing.py)：softmax、top-k route、辅助损失、capacity、dispatch、drop rate、通信量。

## 1. Router 是怎么把 token 送进 expert 的

### 1.1 先把 logits 变成概率

对某个 token，router 给出 $E$ 个 expert 的 logits，记为 $z \in \mathbb{R}^{E}$。softmax 概率为

$$
p_i = \frac{\exp(z_i)}{\sum_{j=1}^{E} \exp(z_j)}
$$

为了数值稳定，工程实现通常先减去最大值：

$$
p_i = \frac{\exp(z_i - \max(z))}{\sum_{j=1}^{E} \exp(z_j - \max(z))}
$$

源码对应 `softmax()`：

```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)
```

也就是说，router 的第一步和普通分类器没有本质区别，都是“logits -> 概率”。

### 1.2 top-k 路由为什么还要重新归一化

MoE 通常不会让每个 token 经过所有 expert，而是只保留概率最大的 $K$ 个 expert：

$$
\operatorname{TopK}(p, K)
$$

但是被选中的这 $K$ 个 expert 的原始概率和通常小于 1，所以还要在被选集合内重新归一化：

$$
\tilde{p}_i = \frac{p_i}{\sum_{j \in \operatorname{TopK}(p, K)} p_j}, \qquad i \in \operatorname{TopK}(p, K)
$$

这一步很关键，因为最终 expert 输出的加权和必须保持在一个归一化尺度上。

源码里的 `topk_route()` 分三步完成这件事：先 softmax，再找 top-k，再对 top-k 概率做归一化。

```python
router_probs = softmax(router_logits.astype(np.float32), axis=-1)
topk_indices = np.argpartition(router_probs, -top_k, axis=-1)[:, -top_k:]

row_ids = np.arange(num_tokens)[:, None]
topk_scores = router_probs[row_ids, topk_indices]
order = np.argsort(-topk_scores, axis=-1)
topk_indices = np.take_along_axis(topk_indices, order, axis=-1)
topk_scores = np.take_along_axis(topk_scores, order, axis=-1)
topk_weights = topk_scores / np.sum(topk_scores, axis=-1, keepdims=True)
expert_load = np.bincount(topk_indices.reshape(-1), minlength=num_experts).astype(np.int32)
```

这段代码里最容易忽略的点是：`np.argpartition()` 只保证“最大的 k 个元素被挑出来”，不保证它们有序，所以后面还要用 `argsort()` 再按概率从大到小排一次。

### 1.3 expert load 是怎么来的

若第 $t$ 个 token 的第 $k$ 个路由槽位选中了 expert $e$，则该 expert 的路由负载可以写成

$$
\text{load}_e = \sum_{t=1}^{N} \sum_{k=1}^{K} \mathbb{1}[\text{route}(t, k) = e]
$$

源码里直接用 `np.bincount()` 把 `topk_indices` 展平后统计频次，这正对应上面的计数公式。

## 2. 负载均衡损失为什么能约束路由塌缩

### 2.1 辅助损失的定义

MoE 最怕的情况是：router 总是把 token 往少数几个 expert 上堆。为避免这种塌缩，常见的辅助损失写作

$$
\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} f_i P_i
$$

其中

$$
f_i = \frac{1}{N} \sum_{t=1}^{N} \mathbb{1}[t \text{ 选择了 expert } i]
$$

$$
P_i = \frac{1}{N} \sum_{t=1}^{N} p_i(t)
$$

这里的 $f_i$ 是“实际被选频率”，$P_i$ 是“router 赋给该 expert 的平均概率质量”。

### 2.2 为什么均匀分配时损失等于 1

如果路由完全均匀，则对所有 expert 都有

$$
f_i = \frac{1}{E}, \qquad P_i = \frac{1}{E}
$$

代回去得到

$$
\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} \frac{1}{E} \cdot \frac{1}{E} = 1
$$

这也是很多实现里把“越接近 1 越平衡”当作直觉参考的原因。

### 2.3 对应源码

[../../src/simulators/moe_routing.py](../../src/simulators/moe_routing.py) 里的 `load_balancing_loss()` 完整对应上述定义：

```python
def load_balancing_loss(router_probs: np.ndarray, topk_indices: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    num_tokens, num_experts = router_probs.shape
    selection = np.zeros((num_tokens, num_experts), dtype=np.float32)
    selection[np.arange(num_tokens)[:, None], topk_indices] = 1.0

    actual_freq = np.sum(selection, axis=0) / num_tokens
    mean_prob = np.mean(router_probs, axis=0)
    loss = num_experts * np.sum(actual_freq * mean_prob)
    return float(loss), actual_freq, mean_prob
```

其中：

- `actual_freq` 对应 $f_i$。
- `mean_prob` 对应 $P_i$。
- `loss` 对应 $E \sum_i f_i P_i$。

## 3. Capacity factor 怎么从公式变成 drop rate

### 3.1 每个 expert 最多能接多少 token

若一个 batch 中有 $N$ 个 token，每个 token 选 $K$ 个 expert，共有 $E$ 个 expert，capacity factor 为 $\text{CF}$，则平均每个 expert 的理论负载是 $N K / E$，实际容量通常写成

$$
\text{Capacity} = \left\lceil \text{CF} \cdot \frac{N K}{E} \right\rceil
$$

对应源码：

```python
def expert_capacity(num_tokens: int, num_experts: int, capacity_factor: float = 1.25, top_k: int = 1) -> int:
    if num_tokens < 0 or num_experts <= 0 or capacity_factor <= 0 or top_k <= 0:
        raise ValueError("invalid capacity inputs")
    return int(np.ceil(capacity_factor * num_tokens * top_k / num_experts))
```

所以 capacity factor 本质上是在“理想平均负载”之上再乘一个冗余系数。

### 3.2 dispatch 为什么会产生 dropped mask

当某个 expert 接收的 token 数超过 `capacity` 时，后来的 token 就必须被拒绝、丢弃或者 fallback。对于第 $e$ 个 expert，可接受 token 数满足

$$
|\mathcal{A}_e| \le \text{Capacity}
$$

超过容量的路由位置会落入 dropped 集合。源码里的 `dispatch_to_experts()` 是逐 token、逐路由槽位做这个检查：

```python
for token_id in range(hidden_states.shape[0]):
    for slot in range(top_k):
        expert_id = int(routing.topk_indices[token_id, slot])
        if len(accepted_indices[expert_id]) < capacity:
            accepted_indices[expert_id].append(token_id)
            accepted_weights[expert_id].append(float(routing.topk_weights[token_id, slot]))
        else:
            dropped[token_id, slot] = True
```

这段代码的工程含义很直接：capacity 不是“建议值”，而是 dispatch 阶段的硬门槛。

### 3.3 drop rate 的公式

如果一共存在 $N K$ 个路由槽位，其中被丢弃的个数是 $N_{\text{drop}}$，则

$$
\text{Drop Rate} = \frac{N_{\text{drop}}}{N K}
$$

源码实现就是取布尔掩码的均值：

```python
def drop_rate(dropped_mask: np.ndarray) -> float:
    if dropped_mask.size == 0:
        return 0.0
    return float(np.mean(dropped_mask))
```

也就是说，`drop_rate()` 默认统计的是“路由槽位丢弃率”，而不是“token 级丢弃率”。如果做 top-2 routing，一个 token 丢一个槽位和丢两个槽位，对这个指标的影响不同。

## 4. All-to-All 的通信量为什么正比于 token 数和 hidden size

### 4.1 dispatch + combine 的最小模型

在 Expert Parallel 中，token 的 hidden states 要先发到目标 expert 所在设备，计算完成后再发回原设备。因此通信至少包含两次：

$$
\text{dispatch} + \text{combine}
$$

若一共有 $N$ 个 token，每个 token 选择 $K$ 个 expert，模型维度为 $d_{\text{model}}$，每个元素字节数为 $s$，则总通信量近似为

$$
\text{Bytes}_{\text{A2A}} = 2 \times N \times K \times d_{\text{model}} \times s
$$

这里前面的 2 就对应发出去再收回来两次。

### 4.2 对应源码

`all_to_all_bytes()` 正是这个公式：

```python
def all_to_all_bytes(num_tokens: int, model_dim: int, top_k: int = 1, bytes_per_elem: int = 2) -> int:
    if min(num_tokens, model_dim, top_k, bytes_per_elem) < 0:
        raise ValueError("all_to_all inputs must be non-negative")
    return 2 * num_tokens * top_k * model_dim * bytes_per_elem
```

这也是为什么 MoE 的系统瓶颈经常不在专家计算本身，而在 All-to-All 的尾延迟和负载倾斜。

## 5. 把整条 MoE 推理链串起来

从单个 batch 的视角，MoE 推理的最小链路可以写成：

$$
\text{router logits} \rightarrow \text{softmax} \rightarrow \text{top-k} \rightarrow \text{capacity check} \rightarrow \text{dispatch} \rightarrow \text{expert compute} \rightarrow \text{combine}
$$

把这条链映射到 [../../src/simulators/moe_routing.py](../../src/simulators/moe_routing.py) 就是：

1. `softmax()`：把 logits 变成概率。
2. `topk_route()`：保留 top-k，并计算归一化权重和 expert load。
3. `load_balancing_loss()`：分析是否出现路由塌缩。
4. `expert_capacity()`：给出每个 expert 的最大接收量。
5. `dispatch_to_experts()`：真正把 token 分给 expert，并记录 dropped mask。
6. `drop_rate()`：衡量容量限制对质量的影响。
7. `all_to_all_bytes()`：估算系统级通信量。

## 6. 建议的源码阅读顺序

1. 先读 [../../math_dictionary/moe-routing-math.md](../../math_dictionary/moe-routing-math.md)，把路由、辅助损失、通信量的大图建立起来。
2. 再读 [../../src/simulators/moe_routing.py](../../src/simulators/moe_routing.py) 的 `softmax()` 和 `topk_route()`，先看路由结果怎么生成。
3. 接着读 `load_balancing_loss()` 和 `expert_capacity()`，把“均衡”和“容量”这两个约束接进来。
4. 最后读 `dispatch_to_experts()`、`drop_rate()`、`all_to_all_bytes()`，从算法视角走到系统视角。

## 这一页记住一句话

> MoE 的核心不是“多几个 expert”这么简单，而是把每个 token 的去向、每个 expert 的容量、每次 All-to-All 的通信量都写成可计算的账本；router 决定算哪里，capacity 决定能不能接住，通信决定系统会不会被拖慢。
