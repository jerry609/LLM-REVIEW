# 从公式到源码：KV Compression / Quantization / Sparsification 对照手册

> 这一页把 KV Compression 拆成两半来讲：一半是“每个 token 还占多少字节”的量化，一半是“哪些 token 还值得保留”的稀疏化。前者改数据表示，后者改缓存集合。

## 这页覆盖哪些源码

- [../../src/kv_cache/compression/quantizer.py](../../src/kv_cache/compression/quantizer.py)：对称 / 非对称 per-channel 量化、反量化、误差指标。
- [../../src/kv_cache/compression/sparsifier.py](../../src/kv_cache/compression/sparsifier.py)：H2O 风格的重分配、SnapKV 风格的一次性选择、压缩比。

## 1. 量化先回答一个问题：每个元素还能用多少 bit 表示

### 1.1 对称 per-channel 量化

若第 $c$ 个通道的浮点值为 $x_c$，量化上界为 $q_{\max}$，则对称量化写成

$$
\text{scale}_c = \frac{\max |x_c|}{q_{\max}}
$$

$$
q_c = \operatorname{clip}\left(\operatorname{round}\left(\frac{x_c}{\text{scale}_c}\right), -q_{\max}, q_{\max}\right)
$$

这里“per-channel”的意思是：不同通道各有各的 `scale_c`，而不是整张量共用一个尺度。

源码对应 [../../src/kv_cache/compression/quantizer.py](../../src/kv_cache/compression/quantizer.py) 的 `quantize_per_channel_symmetric()`：

```python
qmax = (1 << (bits - 1)) - 1
abs_max = np.max(np.abs(flat), axis=0)
abs_max = np.clip(abs_max, a_min=1e-8, a_max=None)
scale = abs_max / qmax

quantized = np.round(flat / scale).astype(np.int8)
quantized = np.clip(quantized, -qmax, qmax).astype(np.int8)
```

这里 `flat` 的最后一维就是“通道维”，所以 `axis=0` 上的 `max` 正对应公式里的按通道取极值。

### 1.2 非对称 per-channel 量化

若通道分布并不以 0 为中心，则更常见的写法是

$$
\text{scale}_c = \frac{\max(x_c) - \min(x_c)}{q_{\max} - q_{\min}}
$$

$$
\text{zero\_point}_c = \operatorname{round}\left(q_{\min} - \frac{\min(x_c)}{\text{scale}_c}\right)
$$

$$
q_c = \operatorname{clip}\left(\operatorname{round}\left(\frac{x_c}{\text{scale}_c}\right) + \text{zero\_point}_c, q_{\min}, q_{\max}\right)
$$

源码对应 `quantize_per_channel_asymmetric()`：

```python
c_min = np.min(flat, axis=0)
c_max = np.max(flat, axis=0)
c_range = np.clip(c_max - c_min, a_min=1e-8, a_max=None)
scale = c_range / (qmax - qmin)
zero_point = np.round(qmin - c_min / scale).astype(np.int8)

quantized = np.round(flat / scale).astype(np.int32) + zero_point.astype(np.int32)
quantized = np.clip(quantized, qmin, qmax).astype(np.int8)
```

### 1.3 反量化和误差指标为什么一起看

反量化公式是

$$
\hat{x}_c = (q_c - \text{zero\_point}_c) \times \text{scale}_c
$$

源码里的 `dequantize()` 正是逐公式实现：

```python
zp = qt.zero_point.astype(np.float32)
result = (flat - zp) * qt.scale
```

量化在工程上从来不是只看“压了多少”，还要看“坏了多少”。因此 `quantization_error()` 计算了

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_i (x_i - \hat{x}_i)^2}
$$

并同时输出 `max_abs_error`、`mean_abs_error` 和 `compression_ratio`。

## 2. 压缩比为什么不能只看 bit 数，还要看元数据

如果原始张量有 $N$ 个元素，每个元素原来占 $s_{\text{old}}$ 字节，压缩后数据本体占 $N \cdot s_{\text{new}}$ 字节，再加上 `scale` 和 `zero_point` 的元数据开销，则真实压缩比应写成

$$
\text{Compression Ratio} = \frac{M_{\text{original}}}{M_{\text{data}} + M_{\text{scale}} + M_{\text{zero\_point}}}
$$

这也是源码里 `quantization_error()` 的写法：

```python
"compression_ratio": original.nbytes / (qt.data.nbytes + qt.scale.nbytes + qt.zero_point.nbytes)
```

所以如果你只说“INT8 比 BF16 节省 2 倍”，这个说法在工程上并不完整，因为你忽略了标定参数开销。

## 3. 稀疏化回答的是另一个问题：哪些 token 值得留下

### 3.1 H2O 风格：累积注意力分数

H2O 类方法的核心是把某个 token 在整个观测历史中的“被关注程度”累积起来：

$$
\text{Score}_{\text{H2O}}(i) = \sum_{t=1}^{T_{\text{obs}}} a_t(i)
$$

其中 $a_t(i)$ 表示第 $t$ 个观测步分配给 token $i$ 的注意力权重。

[../../src/kv_cache/compression/sparsifier.py](../../src/kv_cache/compression/sparsifier.py) 中 `cumulative_attention_scores()` 正是这一步：

```python
def cumulative_attention_scores(attention_history: np.ndarray) -> np.ndarray:
    if attention_history.ndim != 2:
        raise ValueError("attention_history must be rank-2 [num_steps, num_tokens]")
    return np.sum(attention_history.astype(np.float32), axis=0)
```

### 3.2 为什么“最近窗口 + heavy hitter”是一个并集

很多 KV 稀疏策略不会只保留高分 token，还会无条件保留最近的 $W$ 个 token，以降低短期依赖被误删的风险。于是最终保留集合可以写成

$$
\mathcal{K}_{\text{keep}} = \operatorname{TopK}(\text{Score}_{\text{H2O}}, B - W) \cup \mathcal{R}_W
$$

其中 $B$ 是总预算，$\mathcal{R}_W$ 是最近窗口集合。

源码对应 `keep_recent_and_heavy_hitters()`：

```python
recent_start = max(0, total_tokens - recent_window)
recent = np.arange(recent_start, total_tokens, dtype=np.int32)

prefix_scores = cumulative_attention_scores(attention_history)
prefix_scores[recent_start:] = -np.inf

heavy_budget = max(0, budget - recent.shape[0])
heavy = _topk_indices(prefix_scores, heavy_budget)
keep_indices = np.unique(np.concatenate([recent, heavy])).astype(np.int32)
```

这段代码最关键的工程点是：最近窗口先保底，再把剩余预算交给历史高价值 token。

### 3.3 SnapKV 风格：一次性观测窗口决策

如果不想在 decode 过程中持续更新分数，也可以在 prefill 结束后基于观测窗口做一次性选择：

$$
\mathcal{K}_{\text{SnapKV}} = \operatorname{TopK}(s, B - W) \cup \mathcal{R}_W
$$

这里 $s_i$ 是观测窗口估计得到的 token 重要性。

源码对应 `snapkv_select()`：

```python
recent_start = max(0, total_tokens - recent_window)
recent = np.arange(recent_start, total_tokens, dtype=np.int32)

prefix_scores = observation_scores.astype(np.float32).copy()
prefix_scores[recent_start:] = -np.inf

score_budget = max(0, budget - recent.shape[0])
selected = _topk_indices(prefix_scores, score_budget)
keep_indices = np.unique(np.concatenate([recent, selected])).astype(np.int32)
```

和 H2O 的区别是：H2O 用“累积 attention 历史”打分，SnapKV 用“一次性 observation score”打分。

## 4. 稀疏化压缩比和保留预算怎么对应

若总 token 数为 $T$，实际保留 token 数为 $K$，则 token 维度上的压缩比是

$$
\text{Compression Ratio}_{\text{token}} = \frac{T}{K}
$$

源码中的 `compression_ratio()` 就是这个最小公式：

```python
def compression_ratio(total_tokens: int, kept_tokens: int) -> float:
    if total_tokens <= 0 or kept_tokens <= 0 or kept_tokens > total_tokens:
        raise ValueError("require 0 < kept_tokens <= total_tokens")
    return total_tokens / kept_tokens
```

如果你把量化和稀疏化叠加起来，则总体收益近似是

$$
\text{Total Gain} \approx \frac{s_{\text{old}}}{s_{\text{new}}} \times \frac{T}{K}
$$

这也是为什么“量化 + 稀疏化”通常是正交可叠加的两类手段。

## 5. 建议的源码阅读顺序

1. 先读 [../../math_dictionary/kv-compression-math.md](../../math_dictionary/kv-compression-math.md)，把量化误差和 token 保留预算的账本建立起来。
2. 再读 [../../src/kv_cache/compression/quantizer.py](../../src/kv_cache/compression/quantizer.py)，把 `scale`、`zero_point`、`compression_ratio` 对到公式。
3. 接着读 [../../src/kv_cache/compression/sparsifier.py](../../src/kv_cache/compression/sparsifier.py)，理解 H2O 和 SnapKV 在“打分时机”上的区别。
4. 最后跑 [../../tests/test_kv_compression.py](../../tests/test_kv_compression.py)，确认量化和 token 选择逻辑的最小实现都成立。

## 这一页记住一句话

> KV Compression 的两条主线是“缩小每个 token 的表示成本”和“缩小需要保留的 token 集合”。量化解决前者，稀疏化解决后者，真正的工程收益来自两者叠加。
