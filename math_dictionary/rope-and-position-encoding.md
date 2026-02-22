# 旋转位置编码 (RoPE) 与长度外推数学详解

> **核心定位**：从 RoPE 的底层推导，到解决长文本痛点的 PI、NTK-aware，再到当前最先进的 YaRN 算法，提供极其详细的公式推导。全篇支持 LaTeX 渲染，适合直接作为学术参考或面试深度背诵。

---

## 1. 旋转位置编码 (RoPE) 基础推导

RoPE (Rotary Position Embedding) 的核心思想是**通过绝对位置的旋转变换，来实现相对位置的编码**。

### 1.1 核心公式

对于第 $m$ 个位置的输入向量，假设其维度为 $d$（且 $d$ 为偶数），我们将其划分为 $d/2$ 个二维平面。在每个二维平面 $i \in \{0, 1, ..., d/2-1\}$ 上，RoPE 对 Query 和 Key 应用如下旋转变换：

$$
\begin{pmatrix}
q_{m, 2i}^{(R)} \\
q_{m, 2i+1}^{(R)}
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
q_{m, 2i} \\
q_{m, 2i+1}
\end{pmatrix}
$$

其中，旋转角度的基频（Base Frequency）$\theta_i$ 定义为：

$$
\theta_i = b^{-\frac{2i}{d}}
$$

默认情况下，基底 $b = 10000$。

### 1.2 相对位置的优雅体现

Transformer 的核心操作是 Q 和 K 的内积。我们来看看应用 RoPE 之后，位于位置 $m$ 的 Query 和位于位置 $n$ 的 Key 的内积：

$$
\mathbf{q}_m^\top \mathbf{k}_n = \sum_{i=0}^{d/2-1} \left[ q_{m, 2i}k_{n, 2i} \cos((m-n)\theta_i) + q_{m, 2i+1}k_{n, 2i+1} \cos((m-n)\theta_i) + \dots \right]
$$

可以严格证明，旋转变换后的内积**仅与相对位置 $(m-n)$ 有关**：

$$
\langle \mathbf{q}_m^{(R)}, \mathbf{k}_n^{(R)} \rangle = \mathbf{q}_m^\top \mathbf{R}_{d, m-n} \mathbf{k}_n
$$

**结论**：RoPE 在输入层（或每层 QK 投影后）注入绝对位置信息，但在 Attention 算分时自然退化为相对位置偏置，完美契合了语言的相对位置不变性。

---

## 2. 长上下文外推的演进 (Length Extrapolation)

当模型在长度为 $L$（如 4K）的数据上训练，但要在推理时处理长度为 $L' > L$（如 32K）的文本时，直接使用 RoPE 会导致困惑度 (PPL) 爆炸。因为模型从未见过 $m > L$ 产生的大旋转角度。

### 2.1 线性位置插值 (Position Interpolation, PI)

最直观的方法是把超出范围的物理位置，硬生生压缩回训练时见过的范围内。通过一个缩放因子 $s = \frac{L'}{L}$：

$$
m' = \frac{m}{s}
$$

这等价于修改了基频公式：

$$
\theta_i' = \frac{\theta_i}{s}
$$

- **优势**：极大地降低了 PPL 爆炸的问题，通常只需要微调（Fine-tune）极少的步数（<1000步）就能适应。
- **致命缺陷**：这种线性插值对所有频率 $\theta_i$ 一视同仁。高频分量（$i$ 较小，捕捉局部相邻词依赖）的周期被拉长，导致相邻词的位置分辨率下降，模型变得"近视"，细节召回能力变差。

### 2.2 NTK-aware Scaled RoPE

神经正切核 (Neural Tangent Kernel, NTK) 理论表明，神经网络学习低频信号快，学习高频信号慢。因此，外推时：
1. **高频分量（局部信息）**：不能动，必须保持原始分辨率。直接外推。
2. **低频分量（全局信息）**：需要插值，压缩到训练范围内。

NTK-aware 不直接缩放位置 $m$，而是通过放大基底 $b$ 来实现**非线性**的频率缩放。设定新的基底 $b'$：

$$
b' = b \cdot s^{\frac{d}{d-2}}
$$

代入基频公式后得到新的频率：

$$
\theta_i' = (b')^{-\frac{2i}{d}} = b^{-\frac{2i}{d}} \cdot s^{-\frac{2i}{d-2}}
$$

**数学分析**：
- 当 $i=0$（极高频）：$\theta_0' = \theta_0 = 1$（完全没有缩放，纯外推）。
- 当 $i \to d/2$（极低频）：$\theta_{d/2}' \approx \theta_{d/2} \cdot s^{-1}$（等价于 PI 的线性缩放，纯插值）。
- **优势**：这是一个 **Free Lunch**，甚至不需要微调就能让模型在更长上下文上工作，且比 PI 更好地保留了局部信息。

---

## 3. YaRN (Yet another RoPE extensioN) 核心机制详解

> **YaRN 是目前大模型（如 Qwen2.5, LLaMA-3）最主流的长上下文解决方案。** 它是对 NTK-aware 的进一步精细化和理论完善。

YaRN 认为，不仅高频低频需要区别对待，中间频率也应该平滑过渡，而且在修改频率后，注意力分数也需要温度校正。

### 3.1 波长与频率的分组 (Frequency Grouping)

YaRN 定义了波长（Wavelength）的概念 $\lambda_i$：

$$
\lambda_i = \frac{2\pi}{\theta_i} = 2\pi b^{\frac{2i}{d}}
$$

假设训练长度为 $L$，缩放因子 $s = \frac{L'}{L}$。我们定义两个阈值 $\alpha$ 和 $\beta$（YaRN 论文推荐 $\alpha=1, \beta=32$）。这把维度分为三组：

1. **高频组（波长极短，$\lambda_i < \alpha L$）**：这段波长即使放大了 $s$ 倍，也完全在训练长度 $L$ 内。所以**纯外推**，不修改频率。
2. **低频组（波长极长，$\lambda_i > \beta L$）**：波长长于训练长度，模型没见过，必须**纯插值**，等比例缩小频率。
3. **中频组（$\alpha L \le \lambda_i \le \beta L$）**：使用一个 Ramp 函数 $\gamma_i$ 进行平滑过渡。

### 3.2 YaRN 频率修改公式

我们定义每一维的缩放因子 $h_i$：

$$
\gamma_i = \frac{i - i_{\text{low}}}{i_{\text{high}} - i_{\text{low}}}
$$

其中 $i_{\text{low}}$ 和 $i_{\text{high}}$ 分别是对应 $\lambda_i = \alpha L$ 和 $\lambda_i = \beta L$ 的维度索引。

$$
h_i = \begin{cases}
1 & \text{if } i < i_{\text{low}} \quad (\text{高频：不变}) \\
s & \text{if } i > i_{\text{high}} \quad (\text{低频：线性插值}) \\
(1-\gamma_i) \cdot s + \gamma_i \cdot 1 & \text{if } i_{\text{low}} \le i \le i_{\text{high}} \quad (\text{中频：线性组合})
\end{cases}
$$

最终，YaRN 修改后的基频 $\theta_i'$ 为：

$$
\theta_i' = \frac{\theta_i}{h_i}
$$

### 3.3 注意力温度校正 (Temperature Scaling)

YaRN 的另一个极大创新是发现了长度外推时的**熵崩塌**问题。随着序列变长，Attention 矩阵变得越来越大，Softmax 的分母变大，导致注意力分数变得极其平滑（趋于均匀分布），模型丧失了关注重点的能力。

为了对抗这种平滑化，YaRN 在计算 Softmax 之前，对所有的 Attention Score 乘以一个大于 1 的温度调节因子 $t$：

$$
t = \sqrt{0.1 \ln(s) + 1}
$$

最终的 Attention 计算变为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{t \cdot QK^\top}{\sqrt{d_k}}\right)V
$$

**YaRN 总结**：
1. **分段频率缩放**：解决高频丢失和低频未见的问题。
2. **温度校正 $t$**：解决序列变长导致 Softmax 熵崩塌、注意力不集中的问题。
3. **效果**：不仅能在极少微调下实现 128K 甚至 1M 的上下文外推，还能在极长上下文中保持极高的准确率（Needle in a Haystack 表现优异）。

---

## 4. 面试常见追问

**Q1：为什么 RoPE 的 base 默认是 10000？Llama 3 为什么调大到了 500,000？**
> base 决定了波长的衰减速度。base 越大，频率衰减越慢，高频向低频的过渡越平滑，意味着它能编码更长的物理距离而不发生混叠。Llama 3 预训练时就设定在 8K 上下文，增大 base 到 500,000 是为了在预训练阶段就赋予模型更强的原生长度感知和长程外推潜力。

**Q2：YaRN 和 ALiBi 到底谁更强？为什么大模型基本都在用 YaRN？**
> ALiBi 在 attention score 上加距离惩罚，确实能做到 "Zero-shot 外推" 且没有超参，但在处理超过 100K 甚至 1M 的上下文时，ALiBi 对远距离信息的强衰减导致模型"遗忘"远距离线索。而 YaRN 建立在 RoPE 之上，不仅保留了精准的相对位置，还通过温度 $t$ 控制了熵，使得在超长文本的检索（Recall）任务上，YaRN 的表现远好于 ALiBi。