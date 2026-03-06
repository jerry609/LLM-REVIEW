# 从公式到源码：Attention / GQA / RoPE / FlashAttention 对照手册

> 这页专门用来解决“公式能背下来，但落到代码就断层”的问题。阅读顺序固定为：公式 -> 张量形状 -> 核心变量 -> 仓库源码。

## 这页覆盖哪些源码

- [../../src/attention/mha_gqa.py](../../src/attention/mha_gqa.py)：缩放点积注意力、分头、GQA 共享 KV。
- [../../src/attention/rope_rmsnorm.py](../../src/attention/rope_rmsnorm.py)：RoPE cache、旋转、RMSNorm。
- [../../src/attention/flash_attn_sim.py](../../src/attention/flash_attn_sim.py)：分块 attention 和在线 Softmax。

## 1. 缩放点积注意力对应 `mha_gqa.py`

### 1.1 线性投影

输入张量写成：

$$
X \in \mathbb{R}^{B \times T \times D}
$$

线性投影为：

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

其中：

- $Q \in \mathbb{R}^{B \times T \times D}$
- $K, V \in \mathbb{R}^{B \times T \times D_{kv}}$
- MHA 时 $D_{kv} = D$；GQA 时 $D_{kv} = H_{kv} \cdot d_h$

对应源码：

```python
q = x @ w_q
k = x @ w_k
v = x @ w_v
```

这三行正对应 `mha_gqa_forward()` 里的投影部分。这里先在最后一维完成线性映射，暂时还没有拆成多个 head。

### 1.2 从 `[B, T, D]` 拆成 `[B, H, T, d_h]`

定义：

$$
d_h = \frac{D}{H_q}, \qquad
Q_h \in \mathbb{R}^{B \times H_q \times T \times d_h}
$$

`mha_gqa.py` 里的 `_split_heads()` 做的是一次 `reshape + transpose`：

```python
def _split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    bsz, seqlen, dim = x.shape
    head_dim = dim // num_heads
    return x.reshape(bsz, seqlen, num_heads, head_dim).transpose(0, 2, 1, 3)
```

它对应的数学动作是：

$$
\mathbb{R}^{B \times T \times D}
\xrightarrow{\text{reshape}}
\mathbb{R}^{B \times T \times H \times d_h}
\xrightarrow{\text{transpose}}
\mathbb{R}^{B \times H \times T \times d_h}
$$

### 1.3 缩放点积和数值稳定 Softmax

单个 head 的注意力公式：

$$
S = \frac{QK^\top}{\sqrt{d_h}}, \qquad
P = \operatorname{softmax}(S), \qquad
O = PV
$$

仓库实现：

```python
def _scaled_dot_product_attention(q, k, v, mask=None):
    head_dim = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(head_dim)
    if mask is not None:
        scores = np.where(mask, scores, -1e30)
    probs = softmax(scores, axis=-1)
    return np.matmul(probs, v)
```

这里有三个关键点：

- `np.swapaxes(k, -1, -2)` 对应 $K^\top$
- `/ np.sqrt(head_dim)` 对应缩放因子 $1 / \sqrt{d_h}$
- `softmax()` 里先减去 `x_max`，对应安全 Softmax 的数值稳定写法

安全 Softmax 的公式是：

$$
\operatorname{softmax}(z)_i =
\frac{\exp(z_i - \max(z))}{\sum_j \exp(z_j - \max(z))}
$$

### 1.4 GQA 如何共享 KV

定义：

$$
H_q = \text{Query 头数}, \qquad
H_{kv} = \text{KV 头数}, \qquad
G = \frac{H_q}{H_{kv}}
$$

GQA 的核心不是减少 Query 头，而是让每个 KV 头服务 $G$ 个 Query 头：

$$
K'_h = K_{\lceil h / G \rceil}, \qquad
V'_h = V_{\lceil h / G \rceil}
$$

对应源码：

```python
group_size = num_heads // num_kv_heads
if group_size > 1:
    kh = np.repeat(kh, repeats=group_size, axis=1)
    vh = np.repeat(vh, repeats=group_size, axis=1)
```

这里的 `np.repeat(..., axis=1)` 是把 KV 头在“head 维”上逻辑展开到与 Query 头对齐。最重要的工程意义是：

$$
\text{KV cache}_{\text{GQA}} = \frac{H_{kv}}{H_q} \cdot \text{KV cache}_{\text{MHA}}
$$

比如 $H_q = 32, H_{kv} = 8$ 时，KV Cache 直接缩小为原来的 $1/4$。

### 1.5 输出合并

attention 结果还是 `[B, H, T, d_h]`，最终要回到模型维度：

$$
\mathbb{R}^{B \times H \times T \times d_h}
\rightarrow
\mathbb{R}^{B \times T \times (H d_h)}
=
\mathbb{R}^{B \times T \times D}
$$

对应源码：

```python
out = _scaled_dot_product_attention(qh, kh, vh, mask=mask)
out = _merge_heads(out)
return out @ w_o
```

这一步把所有 head 拼接回去，再乘输出投影 $W_O$。

## 2. RoPE 与 RMSNorm 对应 `rope_rmsnorm.py`

### 2.1 RMSNorm

RMSNorm 先算均方根：

$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
$$

再做缩放：

$$
\operatorname{RMSNorm}(x) = \frac{x}{\operatorname{RMS}(x)} \odot w
$$

对应源码：

```python
def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight
```

这和公式几乎一一对应：

- `x * x` 对应平方
- `np.mean(..., axis=-1)` 对应对最后一维求均值
- `np.sqrt(... + eps)` 对应均方根
- `* weight` 对应可学习缩放参数 $w$

### 2.2 RoPE 的频率缓存

RoPE 先构造每个二维平面的旋转角频率。常见写法是：

$$
\omega_i = \theta^{-2i / d_h}, \qquad i = 0, 1, \ldots, d_h/2 - 1
$$

每个位置 $p$ 的相位为：

$$
\phi_{p,i} = p \cdot \omega_i
$$

对应源码：

```python
idx = np.arange(0, head_dim, 2, dtype=np.float32)
inv_freq = 1.0 / (theta ** (idx / head_dim))
pos = np.arange(seqlen, dtype=np.float32)
freqs = np.outer(pos, inv_freq)
cos = np.repeat(np.cos(freqs), 2, axis=-1)
sin = np.repeat(np.sin(freqs), 2, axis=-1)
```

这里的 `np.outer(pos, inv_freq)` 就是一次性构造所有位置、所有频率的相位表 `freqs`。

### 2.3 RoPE 为什么需要 `_rotate_half`

对每一对偶数 / 奇数维度，RoPE 做的是二维旋转：

$$
\begin{pmatrix}
x_{2i}' \\
x_{2i+1}'
\end{pmatrix}
=
\begin{pmatrix}
\cos \phi_i & -\sin \phi_i \\
\sin \phi_i & \cos \phi_i
\end{pmatrix}
\begin{pmatrix}
x_{2i} \\
x_{2i+1}
\end{pmatrix}
$$

`_rotate_half()` 做的就是把

$$
(x_{2i}, x_{2i+1}) \mapsto (-x_{2i+1}, x_{2i})
$$

从而能写成向量化形式：

$$
x' = x \odot \cos \phi + \operatorname{rotate}(x) \odot \sin \phi
$$

对应源码：

```python
def _rotate_half(x: np.ndarray) -> np.ndarray:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = np.empty_like(x)
    out[..., ::2] = -x2
    out[..., 1::2] = x1
    return out

q_out = q * cos + _rotate_half(q) * sin
k_out = k * cos + _rotate_half(k) * sin
```

## 3. FlashAttention 对应 `flash_attn_sim.py`

### 3.1 标准注意力为什么会有大中间矩阵

标准 attention 需要显式构造：

$$
S = QK^\top \in \mathbb{R}^{T \times T}, \qquad
P = \operatorname{softmax}(S), \qquad
O = PV
$$

当 $T$ 很大时，$S$ 和 $P$ 都会变成巨大的中间矩阵，带来显著的 HBM 读写压力。

### 3.2 分块后的三个状态量

FlashAttention 不保存整个 $S$ 和 $P$，而是对每个 Query block 维护三个量：

$$
m_i = \text{当前已扫描块中的行最大值}
$$

$$
l_i = \sum \exp(s - m_i) \text{ 的累计值}
$$

$$
o_i = \text{归一化后的输出累计值}
$$

在扫描到当前块 $(i, j)$ 时，先得到局部统计：

$$
m_{ij} = \max(S_{ij}), \qquad
l_{ij} = \sum \exp(S_{ij} - m_{ij})
$$

再做在线更新：

$$
m_i^{\text{new}} = \max(m_i, m_{ij})
$$

$$
\alpha = \exp(m_i - m_i^{\text{new}}), \qquad
\beta = \exp(m_{ij} - m_i^{\text{new}})
$$

$$
l_i^{\text{new}} = \alpha l_i + \beta l_{ij}
$$

$$
o_i^{\text{new}} =
\frac{\alpha l_i o_i + \beta (P_{ij} V_j)}{l_i^{\text{new}}}
$$

对应源码：

```python
m_new = np.maximum(m_i, m_ij)
alpha = np.exp(m_i - m_new)
beta = np.exp(m_ij - m_new)
l_new = alpha * l_i + beta * l_ij

o_i = (alpha[:, None] * l_i[:, None] * o_i + (beta[:, None] * (p @ v_blk))) / l_new[:, None]
m_i, l_i = m_new, l_new
```

这几行就是整套在线 Softmax 的核心，和论文公式严格对应。

### 3.3 为什么它和标准 Softmax 等价

关键原因是：不同块虽然各自减去了不同的局部最大值，但在合并时又通过 $\alpha$ 和 $\beta$ 把它们重新 rescale 到共同基准 $m_i^{\text{new}}$ 下，所以最终结果和“先看完整一行再做 Softmax”完全一致。

换句话说，FlashAttention 改变的是计算顺序和数据流，不改变数学定义：

$$
\operatorname{FlashAttention}(Q, K, V) = \operatorname{Attention}(Q, K, V)
$$

### 3.4 代码里每个循环分别在做什么

```python
for i in range(0, seqlen, block_size):
    q_blk = q[i:i_end]
    ...
    for j in range(0, seqlen, block_size):
        k_blk = k[j:j_end]
        v_blk = v[j:j_end]
        scores = (q_blk @ k_blk.T) * scale
        ...
```

- 外层循环：固定一个 Query block，维护这一小块输出的在线统计量。
- 内层循环：顺序扫描所有 Key / Value block，把每个块的局部结果并进来。
- `scores = (q_blk @ k_blk.T) * scale`：对应局部块的缩放点积。
- `out[i:i_end] = o_i`：扫描完整个 KV 轴后，写回当前 Query block 的最终输出。

## 4. 推荐对照顺序

1. 先读 [../../math_dictionary/transformer-attention-math.md](../../math_dictionary/transformer-attention-math.md)
2. 再读 [../../math_dictionary/flashattention-math.md](../../math_dictionary/flashattention-math.md)
3. 然后对照：
   - [../../src/attention/mha_gqa.py](../../src/attention/mha_gqa.py)
   - [../../src/attention/rope_rmsnorm.py](../../src/attention/rope_rmsnorm.py)
   - [../../src/attention/flash_attn_sim.py](../../src/attention/flash_attn_sim.py)
4. 最后回看 [mha-vs-gqa-full-derivation.md](mha-vs-gqa-full-derivation.md) 和 [mha-vs-mla-full-derivation.md](mha-vs-mla-full-derivation.md)，把工程结论串起来。
