# 门控注意力 (Gated Attention) 与相关变体数学推导

> **核心定位**：从最简单的通用门控注意力（Gated Cross-Attention），到深度融合 GLU 的 Gated Attention Unit (GAU)，再到目前支持长文本和 sub-quadratic 训练的 Gated Linear Attention (GLA)。全篇使用严格的 LaTeX 数学公式渲染，并提供论文出处。

---

## 1. 为什么需要门控机制？(Why Gating?)

在标准的 Attention 机制中，模型会将不同位置的 $V$（Value）加权求和。但并非所有的历史上下文都是有用的噪声，传统的 Softmax 注意力很难做到“绝对的遗忘”（因为 Softmax 产生的值总是大于 0）。

**门控机制（Gating Mechanism）**的引入，让模型具备了**动态控制信息流**的能力：
1. **阻断无关信息**：通过 $\sigma(X) \approx 0$ 的 Sigmoid 或 SiLU 门控，强行截断部分特征通道的传播。
2. **缓解梯度消失**：门控结构（如 GLU）天然具备更好的梯度流，有助于训练极深的网络。
3. **引入序列维度的遗忘**：像 RNN 一样，让模型能够主动“遗忘”某些过期的前文状态（如 GLA 的核心设计）。

---

## 2. GAU (Gated Attention Unit)

> **出处**：《Transformer Quality in Linear Time》 (Hua et al., 2022)
> **核心思想**：将 Gated Linear Unit (GLU) 与 Attention 强行融合在一个层里，用极其简化的单头注意力搭配强大的 GLU 门控，达到比多头注意力（MHA）更好且更快的性能。

### 2.1 核心公式推导

对于输入序列矩阵 $X \in \mathbb{R}^{T \times d_{\text{model}}}$，GAU 首先生成两个门控分支 $U$ 和 $V$（类似于 GLU 的做法）：

$$
U = \phi_u(X W_u), \quad V = \phi_v(X W_v)
$$

其中 $W_u, W_v \in \mathbb{R}^{d_{\text{model}} \times d_{\text{out}}}$，$\phi_u, \phi_v$ 通常是 SiLU 或 Swish 激活函数。

接下来，计算一个极简的**单头注意力（Single-Head Attention）**，它不需要映射到多头，而是直接作用在较小的维度 $d_{\text{attn}}$ 上（如 $d_{\text{attn}} = 64$）：

$$
Q = X W_q, \quad K = X W_k
$$
$$
A = \frac{1}{T} \text{ReLU}^2\left( \frac{Q K^\top}{\sqrt{d_{\text{attn}}}} \right)
$$
*(注：原论文发现用 $\text{ReLU}^2$ 替代 Softmax 可以在保持质量的同时极大提升计算速度，也可以换回 Softmax。)*

最后，将**注意力算出的加权结果**，被**门控分支 $U$ 进行特征维度上的逐元素缩放**：

$$
O = \left( U \odot (A \cdot V) \right) W_o
$$

- $\odot$ 是逐元素乘法（Hadamard Product）。
- $W_o \in \mathbb{R}^{d_{\text{out}} \times d_{\text{model}}}$ 是输出投影矩阵。

### 2.2 架构优势
传统 Transformer 每层需要跑两遍：`MHA -> LayerNorm -> FFN -> LayerNorm`。
GAU 证明了：把 Attention 放进 FFN（更确切地说是 GLU）的门控分支里，**一层顶两层**，不仅去掉了 MHA 的多头冗余计算，还因为融合了 GLU 获得了极强的非线性表达能力。

---

## 3. GLA (Gated Linear Attention)

> **出处**：《Gated Linear Attention Transformers with Hardware-Efficient Training》 (Yang et al., 2023)
> **核心思想**：纯线性注意力（Linear Attention）没有长距离遗忘机制，效果一直被标准 Softmax 注意力碾压。GLA 为线性注意力引入了一维的**数据依赖型遗忘门（Forget Gate）**，使其兼具 RNN 的推理效率和 Transformer 的硬件训练效率。

### 3.1 线性注意力的致命缺陷
标准线性注意力以 RNN 形式写出来时，其隐藏状态 $S_t$ 的更新公式是：
$$
S_t = S_{t-1} + K_t V_t^\top
$$
这意味着所有的历史信息被**无差别地**全部累加到 $S_t$ 中，模型无法遗忘早期的无用信息（比如上一个段落的句号）。

### 3.2 GLA 的遗忘门机制 (RNN 递归形式)

GLA 引入了一个依赖于当前输入的门控向量 $g_t \in \mathbb{R}^{d_{\text{head}}}$，并将其转化为遗忘系数 $\alpha_t$：

$$
g_t = X_t W_g \quad \text{(计算门控)}
$$
$$
\alpha_t = \sigma(g_t) \quad \text{(Sigmoid 激活限制在 0~1 之间)}
$$

然后用 $\alpha_t$ 作为衰减因子（Decay Factor）更新隐藏状态：

$$
S_t = \alpha_t \odot S_{t-1} + K_t V_t^\top \quad (\text{状态更新})
$$
$$
O_t = Q_t \odot S_t \quad (\text{输出查询})
$$

- 当 $\alpha_t \to 0$ 时，模型强行遗忘之前的所有的 $S_{t-1}$，只保留新的 $K_t V_t^\top$。
- $\odot$ 代表对特征维度的逐通道独立门控。

### 3.3 Chunkwise Parallel 形式 (块并行训练)

直接用 RNN 递归形式在 GPU 上训练极慢（因为不可并行）。GLA 的伟大之处在于提出了 Chunkwise 算法（类似 FlashAttention 的分块）：

将长度为 $T$ 的序列分成多个块（Chunk）。块内部使用类似标准 Transformer 的并行计算，块之间使用 RNN 方式传递状态。

对于任意位置 $i$ 和 $j$（其中 $j \le i$），从时间步 $j$ 到 $i$ 的**累积遗忘因子（Cumulative Decay）**为：
$$
\alpha_{i,j} = \prod_{k=j+1}^i \alpha_k \quad (\text{若 } j = i \text{，则 } \alpha_{i,i} = 1)
$$

则第 $i$ 步的完整注意力输出可并行计算为：
$$
O_i = Q_i \sum_{j=1}^i \alpha_{i,j} \left( K_j V_j^\top \right)
$$

**GLA 的意义**：它让线性注意力在语言建模任务上首次缩小了与 Transformer 的差距，并为之后的 Mamba / GLA 混合架构（如 Jamba）奠定了硬件友好型 RNN 训练的理论基础。

---

## 4. 交叉注意力门控 (Gated Cross-Attention)

> **典型应用**：多模态模型（如 DeepMind 的 Flamingo、Perceiver），或用于将外部检索信息（RAG）注入大模型的场景。

### 4.1 公式原理
在语言模型的主干网络中，我们希望**有条件地**吸收外部信息（例如图片特征或检索出的文档 $C$）。如果在某些层外部信息全是噪声，模型应当能够完全“关闭”交叉注意力。

$$
\text{Attn}_{\text{cross}} = \text{Softmax}\left(\frac{X W_q \cdot (C W_k)^\top}{\sqrt{d_k}}\right) (C W_v)
$$

引入一个**可学习的标量或向量门控参数** $\tanh(\gamma)$（初始值通常设为 $0$）：

$$
O = X + \tanh(\gamma) \odot \text{Attn}_{\text{cross}}
$$

- **初始化为 0**：在训练初期，$\tanh(\gamma) \approx 0$，模型等价于纯语言模型，保持了预训练模型的稳定性。
- **渐进式开放**：随着训练，$\gamma$ 逐渐学习到何时放大交叉注意力的特征，动态吸收外部信息。

---

## 5. 面试实战总结

1. **什么是门控（Gating）机制？**
   答：“门控机制本质上是一个可学习的非线性信息阀门，通常用 Sigmoid/SiLU 实现。在 LLM 中，GLU/SwiGLU 是特征维度的门控，而 GLA 则是序列时间维度的遗忘门。”
2. **为什么 GAU (Gated Attention Unit) 比 MHA 更好？**
   答：“GAU 把 Attention 操作放进了 GLU 的门控分支里，用极简的单头注意力配合强大的 SiLU 门控，既省去了多头注意力的维度切分与合并开销，又在表达能力上达到了 $1 \text{ 级 GAU} \approx 1 \text{ 级 MHA} + 1 \text{ 级 FFN}$ 的效果。”
3. **线性注意力为什么需要加 Gate（如 GLA）？**
   答：“朴素线性注意力将历史 KV 简单求和，缺乏遗忘机制。GLA 引入数据依赖的 decay 因子（Forget Gate），让状态具备类似 RNN 的短期记忆能力，并通过 Chunkwise 实现了硬件友好的并行训练。”