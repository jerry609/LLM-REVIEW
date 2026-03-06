# 门控注意力 (Gated Attention) 与相关变体数学推导

> **核心定位**：把“门”看成可学习的信息阀门，从特征维度的 GLU 风格门控，到序列维度的遗忘门，再到跨模态或 RAG 注入时的 gated cross-attention。本页按“为什么要加门 -> GAU -> GLA -> 交叉注意力扩展”的顺序组织，重点放在信息流控制而不是只罗列公式。

---

## 1. 为什么门控值得单独讨论

### 1.1 标准注意力擅长加权，不擅长硬性遗忘

标准 attention 会把历史 value 按权重加总，因此它很擅长“强调重要信息”，但不擅长“彻底关闭无关信息”。常见的 softmax 权重是非负且归一化的，它更像连续重分配权重，而不是执行一个明确的开关操作。

### 1.2 特征维度上的门控

门控最常见的写法，是让一条分支产生内容，另一条分支产生 gate，再做逐元素相乘：

$$
Y = G(X) \odot H(X)
$$

这类门控直接决定哪些通道可以通过，GLU、SwiGLU、GAU 都属于这一类思想。

### 1.3 序列维度上的遗忘

如果门控直接作用在时序状态更新上，模型就不只是“压暗某些特征”，而是能主动遗忘旧状态。GLA 的核心更新式就是这种形式：

$$
S_t = \alpha_t \odot S_{t-1} + K_tV_t^\top
$$

当 $\alpha_t$ 很小时，旧状态会被快速衰减；当 $\alpha_t$ 接近 $1$ 时，模型会保留更多长期上下文。

### 1.4 本页主线

| 机制 | Gate 作用位置 | 主要目标 |
|------|---------------|----------|
| GAU | 特征维度 | 把 attention 和 FFN 融成一个更轻的块 |
| GLA | 状态更新 | 在线性注意力里补上遗忘机制 |
| Gated Cross-Attention | 残差注入 | 按需吸收外部上下文、图像或检索结果 |

---

## 2. GAU (Gated Attention Unit)

> **出处**：《Transformer Quality in Linear Time》 (Hua et al., 2022)  
> **核心思想**：把 GLU 风格的门控与简化 attention 写进同一个块，用更少的结构完成“信息选择 + 非线性变换”。

### 2.1 门控分支先决定哪些特征能通过

对于输入序列 $X \in \mathbb{R}^{T \times d_{\mathrm{model}}}$，GAU 先投影出两个分支：

$$
U = \phi_u(XW_u), \quad V = \phi_v(XW_v)
$$

其中 $W_u, W_v \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{out}}}$，$\phi_u, \phi_v$ 通常取 SiLU 或 Swish。可以把 $V$ 看成待聚合的内容，把 $U$ 看成聚合结果最终允许通过多少的 gate。

### 2.2 简化 attention 分支负责建模位置关系

GAU 不再显式拆成多头，而是直接在较小的 attention 维度上计算相关性：

$$
Q = XW_q, \quad K = XW_k
$$

$$
A = \frac{1}{T}\operatorname{ReLU}^2\left(\frac{QK^\top}{\sqrt{d_{\mathrm{attn}}}}\right)
$$

原论文发现，使用 $\operatorname{ReLU}^2$ 可以在保持质量的同时省掉 softmax 的一部分代价。如果需要，也可以把这里替换回 $\operatorname{Softmax}$ 版本。

### 2.3 真正的关键信息流在融合步骤

GAU 的输出写成：

$$
O = \left(U \odot (AV)\right)W_o
$$

这里的三个量分别负责不同角色：

| 量 | 作用 |
|----|------|
| $AV$ | 把序列中的相关位置聚合成上下文表示 |
| $U$ | 控制每个输出通道是被放大、抑制还是关闭 |
| $W_o$ | 把融合结果映射回模型维度 |

这也是 GAU 和“先做 attention、再进 FFN”最本质的区别：门控不是后处理，而是直接嵌入信息路由过程里。

### 2.4 为什么说 GAU 接近“一层顶两层”

标准 Transformer 往往需要一层 attention 加一层 FFN，才能同时完成“跨位置聚合”和“通道级非线性变换”。GAU 把二者压进同一个模块里，因此常被概括成：

$$
\operatorname{GAU} \approx \operatorname{Attention} + \operatorname{GLU}
$$

工程上它的优势不是完全消灭 attention，而是用更紧凑的结构拿到接近 `MHA + FFN` 的表达能力。

---

## 3. GLA (Gated Linear Attention)

> **出处**：《Gated Linear Attention Transformers with Hardware-Efficient Training》 (Yang et al., 2023)  
> **核心思想**：在线性注意力的递归状态更新里加入数据依赖的遗忘门，让模型既保留 RNN 式高效推理，又不至于把所有历史信息无差别累加。

### 3.1 朴素线性注意力的问题是“只积累，不遗忘”

把线性注意力写成递归形式时，核心状态更新通常是：

$$
S_t = S_{t-1} + K_tV_t^\top
$$

这个式子的好处是推理复杂度低，但问题也很直接：所有历史项都被直接叠加进状态里。只要上下文足够长，旧信息和噪声都会持续滞留，模型没有明确机制决定哪些记忆应该尽快过期。

### 3.2 遗忘门把线性注意力变成可控记忆

GLA 先从当前输入生成 gate：

$$
g_t = X_tW_g, \quad \alpha_t = \sigma(g_t)
$$

再用 $\alpha_t$ 衰减旧状态：

$$
S_t = \alpha_t \odot S_{t-1} + K_tV_t^\top
$$

$$
O_t = Q_t \odot S_t
$$

这几个量的直觉可以直接对应成一套记忆系统：

| 量 | 解释 |
|----|------|
| $\alpha_t$ | 当前位置对历史状态的保留比例 |
| $K_tV_t^\top$ | 当前 token 注入的新证据 |
| $S_t$ | 累积后的记忆状态 |
| $O_t$ | 查询当前状态后得到的输出 |

当 $\alpha_t \to 0$ 时，模型几乎清空旧状态；当 $\alpha_t \to 1$ 时，模型更接近朴素线性注意力。

### 3.3 Chunkwise 训练把递归状态重新并行化

如果直接按递归式训练，GPU 并行度会很差。GLA 的关键工程贡献是把序列切成 chunk，在块内并行、块间递归。对任意 $j \le i$，累积衰减因子写成：

$$
\alpha_{i,j} = \prod_{k=j+1}^{i} \alpha_k, \quad \alpha_{i,i} = 1
$$

于是第 $i$ 个位置的输出可以写成：

$$
O_i = Q_i \sum_{j=1}^{i} \alpha_{i,j}\left(K_jV_j^\top\right)
$$

这个形式保留了“先忘、再累积”的语义，同时把大量计算重新搬回适合 GPU 的块并行范式。

### 3.4 GLA 的工程意义

GLA 的价值不只在公式上更优雅，而在于它回答了一个长期存在的问题：为什么很多线性注意力在长文本里质量不稳定。答案通常不是线性化本身错了，而是缺少一套数据依赖的遗忘机制。GLA 补上的正是这部分，因此它也成为后续混合架构和长上下文模型的重要基础。

---

## 4. 门控机制在自注意力之外的扩展

### 4.1 为什么 Cross-Attention 也需要 gate

在多模态和 RAG 场景里，外部上下文并不总是有用。如果每一层都无条件吸收图像特征或检索片段，模型很容易被噪声拖偏。因此更合理的做法不是永远开着 cross-attention，而是让模型自己决定某一层、某一通道是否需要外部信息。

### 4.2 Gated Cross-Attention 的基本写法

设外部上下文为 $C$，则普通 cross-attention 可写成：

$$
\operatorname{Attn}_{\mathrm{cross}} = \operatorname{Softmax}\left(\frac{XW_q(CW_k)^\top}{\sqrt{d_k}}\right)(CW_v)
$$

再加上一个可学习 gate：

$$
O = X + \tanh(\gamma) \odot \operatorname{Attn}_{\mathrm{cross}}
$$

其中 $\gamma$ 可以是标量，也可以是向量。它决定外部信息是被大幅引入，还是只做很弱的残差修正。

### 4.3 为什么常把 gate 初始化为零

如果初始化时 $\gamma \approx 0$，那么模型一开始几乎等价于原始语言模型，只是在残差支路上留了一个未来可以打开的阀门。这会让训练更稳定，因为模型不需要一上来就同时适应新模态、新检索上下文和新损失。

---

## 5. 面试实战总结

### 5.1 什么是 gating

可以直接回答：门控机制本质上是可学习的信息阀门。它既可以作用在特征维度上，决定哪些通道通过，也可以作用在时间维度上，决定旧状态要保留多少。

### 5.2 为什么 GAU 经常拿来对比 MHA + FFN

因为 GAU 把跨位置聚合和通道级非线性门控压进同一个块里。它不是简单把 FFN 换个名字，而是通过 $U \odot (AV)$ 这种写法，让 gate 直接参与 attention 输出的路由。

### 5.3 线性注意力为什么需要 forget gate

朴素线性注意力的问题是状态只会加不会忘，长上下文里噪声会不断积累。GLA 用 $\alpha_t$ 给状态更新加上衰减项，本质上是在给线性 attention 补一个可学习的记忆管理器。

### 5.4 Gated Cross-Attention 适合什么场景

当外部上下文质量不稳定时，gated cross-attention 很有价值。它常见于多模态和 RAG，因为模型需要的是按需吸收外部信息，而不是每层都无条件放大外部特征。
