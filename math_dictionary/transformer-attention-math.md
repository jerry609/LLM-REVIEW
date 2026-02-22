# Transformer 注意力机制与核心组件公式详解

> **核心定位**：本手册系统性收录 Transformer 架构最核心的数学公式，包括 Attention 的缩放点积、各类归一化（LayerNorm/RMSNorm）、激活函数（SwiGLU）以及架构连接（Pre/Post-Norm）。全篇使用严谨的 LaTeX 语法，方便背诵与推导。

---

## 1. 缩放点积注意力 (Scaled Dot-Product Attention)

### 1.1 核心计算公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}} + M\right) V
$$

- $Q \in \mathbb{R}^{T \times d_{\text{head}}}$: 查询矩阵（Query）
- $K \in \mathbb{R}^{T \times d_{\text{head}}}$: 键矩阵（Key）
- $V \in \mathbb{R}^{T \times d_{\text{head}}}$: 值矩阵（Value）
- $M \in \mathbb{R}^{T \times T}$: 掩码矩阵（Mask），如因果掩码（Causal Mask）或填充掩码（Padding Mask）。
- $d_{\text{head}}$: 单个注意力头的特征维度。

### 1.2 缩放因子 $\frac{1}{\sqrt{d_{\text{head}}}}$ 的数学意义

**为什么需要除以 $\sqrt{d_{\text{head}}}$？**
假设 $Q$ 和 $K$ 中的元素均为独立同分布（i.i.d.）的随机变量，均值为 $0$，方差为 $1$。
它们的内积 $q \cdot k = \sum_{i=1}^{d_{\text{head}}} q_i k_i$。

- 均值：$\mathbb{E}[q \cdot k] = \sum \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$
- 方差：$\text{Var}(q \cdot k) = \sum \text{Var}(q_i k_i) = d_{\text{head}}$

随着维度 $d_{\text{head}}$ 变大，点积的方差也会线性增加（例如从 $64$ 变大）。方差变大意味着输入到 Softmax 的数值 $z_i$ 的绝对值会变得极大。
Softmax 函数对极大值的梯度接近于 $0$（梯度消失）。为了将方差重新缩放回 $1$，保持 Softmax 处于对数值敏感的梯度区域，必须除以标准差 $\sqrt{d_{\text{head}}}$。

---

## 2. 数值稳定的 Softmax (Safe Softmax)

直接计算 $\exp(z_i)$ 容易导致数值溢出（Overflow），例如当 $z_i > 88$ 时，在 FP16 下会超出表示范围变成 `NaN`。

标准的工程实现是**减去当前行的最大值**：

$$
\text{softmax}(z)_i = \frac{\exp(z_i - \max(z))}{\sum_{j} \exp(z_j - \max(z))}
$$

在 FlashAttention 中使用的 Online Softmax 进一步拓展了这一公式，通过分块维护局部的 `max` 和 `sum` 来实现流式计算，避免 $O(T^2)$ 内存占用。

---

## 3. 多头注意力 (Multi-Head Attention, MHA)

将隐藏层维度 $d_{\text{model}}$ 拆分为 $H$ 个独立的头（Head），每个头维度 $d_{\text{head}} = d_{\text{model}} / H$。

1. **线性投影**：
   $$
   Q_i = X W_i^Q, \quad K_i = X W_i^K, \quad V_i = X W_i^V
   $$
2. **独立计算注意力**：
   $$
   \text{head}_i = \text{Attention}(Q_i, K_i, V_i)
   $$
3. **拼接与输出映射**：
   $$
   \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W^O
   $$
   其中 $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ 负责融合各头信息。

---

## 4. 归一化层 (Normalization)

### 4.1 Layer Normalization (LayerNorm)

对特征的最后一维（特征维 $d_{\text{model}}$）进行均值方差归一化。

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

- $\mu = \frac{1}{d} \sum_{i=1}^d x_i$ （均值）
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$ （方差）
- $\gamma, \beta \in \mathbb{R}^d$ 是可学习的缩放（Scale）和平移（Shift）参数。

### 4.2 RMSNorm (Root Mean Square Normalization)

> **LLaMA 等现代大模型标配。**

RMSNorm 去掉了 LayerNorm 中的均值中心化（即假设均值 $\mu \approx 0$），也不使用平移参数 $\beta$。

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}
$$

**优势**：
1. **速度更快**：省去了计算均值 $\mu$ 的额外一步 Reduce 操作。
2. **效果相当**：实验证明去均值中心化不影响模型的收敛能力和最终表现。

---

## 5. 激活函数 (Activation Functions)

### 5.1 Swish / SiLU

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + \exp(-x)}
$$
- 平滑且非单调，相比 ReLU 有更好的梯度流动。

### 5.2 SwiGLU (Swish Gated Linear Unit)

> **大语言模型（LLaMA/Qwen/DeepSeek 等）的前馈网络 (FFN) 标配。**

原始 Transformer 的 FFN 是 $\text{ReLU}(xW_1)W_2$。SwiGLU 引入了门控机制（Gate），需要三个权重矩阵：$W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$。

$$
\text{SwiGLU}(x) = \left( \text{SiLU}(xW_{\text{gate}}) \odot (xW_{\text{up}}) \right) W_{\text{down}}
$$

- $W_{\text{gate}}$ 控制信息的激活比例。
- $\odot$ 是逐元素相乘。
- 维度通常从 $d_{\text{model}}$ 投影到中间隐藏维 $d_{\text{ff}}$（通常为 $\frac{8}{3} d_{\text{model}}$ 或 $4d_{\text{model}}$），然后再投回 $d_{\text{model}}$。

---

## 6. 架构连接：Pre-Norm vs Post-Norm

### 6.1 Post-Norm (原始 Transformer)

$$
x_{l+1} = \text{Norm}(x_l + \text{SubLayer}(x_l))
$$
- 梯度必须穿过 $\text{Norm}$ 函数。随着层数增加，靠近底层的梯度变小，导致深层网络训练极不稳定，常需要 Warmup 预热。

### 6.2 Pre-Norm (现代大模型如 LLaMA / GPT 标配)

$$
x_{l+1} = x_l + \text{SubLayer}(\text{Norm}(x_l))
$$
- 主干路径（$x_l$ 的累加）没有任何阻碍，梯度可以通过加法残差直接流向底层。
- **极大提升了训练稳定性**，允许训练百层以上的超深网络，并减少对 Warmup 的依赖。

---

## 7. 采样生成与解码公式

在自回归解码的最后一步，logits 向量会通过带有**温度 $T$ (Temperature)** 的 Softmax 转化为概率分布：

$$
p_i = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}
$$

- 当 $T \to 0$：退化为 Greedy Decoding（贪心策略），永远选概率最大的词。
- 当 $T = 1$：标准的概率采样。
- 当 $T \to \infty$：变成完全均匀分布的随机乱猜。

**Top-p (Nucleus Sampling) 截断**：
先按概率降序排列词表，累加概率直到超过阈值 $p$（例如 0.9），截断并丢弃剩余词，然后在保留的词集中重新做 Softmax 采样。