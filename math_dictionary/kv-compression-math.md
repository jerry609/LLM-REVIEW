# KV Compression：从量化误差到保留预算

> 这页把 KV Compression 拆成两条更适合工程决策的主线：第一条是“每个 token 还能不能更便宜”，也就是量化；第二条是“是不是所有 token 都必须保留”，也就是选择与稀疏化。真正的线上方案，往往是这两条线一起用，而不是只做其中一边。

## 1. 先把总账写对

压缩后的 KV 总开销，不应该只写成“原来多少倍”，而应该直接写成预算式：

$$
M_{\mathrm{KV, final}} = T_{\mathrm{kept}} \times \text{bytes}_{\text{token}}^{\mathrm{compressed}} + M_{\mathrm{meta}} + M_{\mathrm{index}}
$$

这条式子把两个核心旋钮都暴露出来了：

- `T_kept` 控制你最后到底保留多少历史 token。
- `bytes_token_compressed` 控制每个保留下来的 token 有多贵。

如果只盯着其中一个，很容易得出看起来漂亮、但上线不稳的结论。

## 2. 路线一：先压低每个 token 的字节数

### 2.1 对称 `per-channel` 量化的主公式

对于某个通道上的浮点向量 `x_j`，`b` bit 对称量化可以写成：

$$
\text{scale}_j = \frac{\max |x_j|}{2^{b-1} - 1}
$$

$$
q_j = \operatorname{round}\!\left(\frac{x_j}{\text{scale}_j}\right)
$$

$$
\hat{x}_j = \text{scale}_j \cdot q_j
$$

这三步对应的工程含义很直接：

1. 先为每个通道单独求一个动态范围。
2. 再把浮点值映射到有限整数格点。
3. 需要用时再做反量化。

仓库里这条线最直接对应的是：

- `../src/kv_cache/compression/quantizer.py` 的 `quantize_per_channel_symmetric()`
- `../src/kv_cache/compression/quantizer.py` 的 `dequantize()`

### 2.2 量化误差为什么能先看上界

单个标量量化到最近格点时，误差不会超过半个量化步长，因此可以直接写成：

$$
|x_j - \hat{x}_j| \le \frac{1}{2} \text{scale}_j
$$

如果把一个通道的误差看成向量误差，最粗但很实用的上界是：

$$
\|x - \hat{x}\|_2^2 \le \frac{1}{4} \sum_j \text{scale}_j^2
$$

这也是为什么 `per-channel` 往往比 `per-tensor` 更稳：不同通道的动态范围差异被拆开处理后，少数超大值不会把整块量化网格拉得过粗。

### 2.3 非对称量化什么时候更合适

如果数据分布明显偏向某一侧，例如最小值和最大值离原点并不对称，那么更自然的写法是：

$$
\text{scale}_j = \frac{x_j^{\max} - x_j^{\min}}{2^b - 1}
$$

$$
\text{zero\_point}_j = \operatorname{round}\!\left(-\frac{x_j^{\min}}{\text{scale}_j}\right)
$$

$$
q_j = \operatorname{round}\!\left(\frac{x_j}{\text{scale}_j} + \text{zero\_point}_j\right)
$$

它在仓库里对应的是 `../src/kv_cache/compression/quantizer.py` 的 `quantize_per_channel_asymmetric()`。

### 2.4 量化压缩比怎么算才不自欺欺人

忽略 `scale` 和 `zero_point` 的元数据时，量化压缩比最容易写成：

$$
R_{\mathrm{quant}} \approx \frac{s_{\mathrm{fp}}}{s_{\mathrm{q}}}
$$

例如从 BF16 到 INT8，就是：

$$
R_{\mathrm{quant}} \approx \frac{2}{1} = 2
$$

但线上估算最好再补一层元数据项：

$$
\text{bytes}_{\text{token}}^{\mathrm{quant}} \approx 2 \times L_{\mathrm{layers}} \times n_{\mathrm{kv\_heads}} \times d_{\mathrm{head}} \times s_{\mathrm{q}} + M_{\mathrm{scales}} + M_{\mathrm{zero\_points}}
$$

这也是为什么非常短的上下文、或者通道数不大的场景里，理论压缩比和真实压缩比会有偏差。

## 3. 路线二：不是所有 token 都值得一直保留

### 3.1 H2O 风格的核心想法

如果某些历史 token 在很多步里都被反复关注，那么它们就更像“长期有价值的重击点”。最直接的累计打分就是：

$$
\text{score}_i = \sum_{t=1}^{T_{\mathrm{obs}}} a_{t,i}
$$

其中 `a_t,i` 可以理解成第 `t` 步对第 `i` 个历史 token 的注意力权重。

如果总预算是 `B`，最近窗口要保留 `r` 个 token，那么保留集合可以写成：

$$
\mathcal{K}_{\mathrm{H2O}} = \operatorname{TopK}(\text{score}, B-r) \cup \{T-r+1, \ldots, T\}
$$

也就是说，它做的是“两段式保留”：

- 一段保住最近 token，避免把短期局部依赖砍掉。
- 一段保住历史重击点，避免把长期关键上下文忘掉。

仓库里对应的是：

- `../src/kv_cache/compression/sparsifier.py` 的 `cumulative_attention_scores()`
- `../src/kv_cache/compression/sparsifier.py` 的 `keep_recent_and_heavy_hitters()`

### 3.2 SnapKV 风格更像一次性选拔

有些方案不会持续累计全历史，而是用一个观察窗口估计“谁值得留下”。它的主式可以写成：

$$
\text{score}_i^{\mathrm{snap}} = \sum_{t \in \mathcal{W}_{\mathrm{obs}}} a_{t,i}
$$

然后同样保留最近窗口，再从更老的 token 里挑高分者：

$$
\mathcal{K}_{\mathrm{SnapKV}} = \operatorname{TopK}(\text{score}^{\mathrm{snap}}, B-r) \cup \{T-r+1, \ldots, T\}
$$

它在仓库里对应 `../src/kv_cache/compression/sparsifier.py` 的 `snapkv_select()`。

### 3.3 稀疏化压缩比的最简单写法

如果原始历史长度是 `T_total`，最后只保留 `T_kept`，那么 token 维度上的压缩比就是：

$$
R_{\mathrm{token}} = \frac{T_{\mathrm{total}}}{T_{\mathrm{kept}}}
$$

仓库里直接给了对应实现：`../src/kv_cache/compression/sparsifier.py` 的 `compression_ratio()`。

## 4. 量化和稀疏化真正该怎么合起来看

很多讨论会把量化和稀疏化分开讲，但真正做容量预算时，更有用的是把它们合成一条式子：

$$
R_{\mathrm{overall}} \approx \frac{T_{\mathrm{total}} \times \text{bytes}_{\text{token}}^{\mathrm{fp}}}{T_{\mathrm{kept}} \times \text{bytes}_{\text{token}}^{\mathrm{quant}} + M_{\mathrm{meta}} + M_{\mathrm{index}}}
$$

这条式子背后的工程判断是：

- 如果你已经把 `bytes_token` 压得很低，再继续抠位宽，收益可能不如减少 `T_kept`。
- 如果你已经把 `T_kept` 砍得很狠，再继续减 token，质量风险往往比继续量化更大。
- 元数据不是零成本，特别是分组量化、索引表、保留集合掩码都会吃预算。

## 5. 怎么从公式一路对到仓库源码

这一页最建议按下面顺序对照：

1. 先看 `../src/kv_cache/compression/quantizer.py`。
   - `quantize_per_channel_symmetric()` 对应对称量化主式。
   - `quantize_per_channel_asymmetric()` 对应非对称量化主式。
   - `quantization_error()` 对应误差统计与回放。
2. 再看 `../src/kv_cache/compression/sparsifier.py`。
   - `cumulative_attention_scores()` 对应累计注意力打分。
   - `keep_recent_and_heavy_hitters()` 对应 H2O 风格保留规则。
   - `snapkv_select()` 对应观察窗口打分与一次性选择。
3. 最后看 `../tests/test_kv_compression.py`。
   - 这组测试把量化往返、H2O 风格选择、SnapKV 风格选择和压缩比都做了最小验证。

如果你想按专题继续深入，可以接：

- `../notes/kv-compression/formula-to-code-walkthrough.md`
- `kv-memory.md`
- `kv-eviction-math.md`

## 6. 什么时候该优先量化，什么时候该优先选 token

| 场景 | 更先考虑什么 | 原因 |
|------|--------------|------|
| 长上下文很多，但重要信息分布很稀疏 | 稀疏化与 token 选择 | `T_kept` 才是主要矛盾 |
| 长度中等，但 KV 明显占满显存 | 量化 | 先把每个 token 变便宜最稳 |
| 多租户、高波动、上下文差异极大 | 二者结合 | 只靠单一手段很难稳定 |
| 对质量极其敏感，无法接受错删 | 轻量量化优先 | 比大幅稀疏化风险更可控 |

## 7. 这页真正该记住什么

- KV Compression 不是单一算法，而是“字节压缩”和“保留压缩”两条线的组合题。
- `per-channel` 量化的关键收益，不只是压缩率，更是误差更可控。
- H2O 和 SnapKV 的核心都不是“神秘论文技巧”，而是显式定义一套 token 重要性评分函数。
- 真正上线时，压缩比一定要把元数据和索引开销一起算进去。
