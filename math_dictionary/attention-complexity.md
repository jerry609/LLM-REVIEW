# 复杂度、FLOPs 与算力瓶颈分析数学详解

> **核心定位**：从最底层的时间和空间复杂度推导开始，精确计算 Transformer 每层的浮点运算次数（FLOPs），并结合 Roofline 模型深度分析大模型推理的算力与带宽瓶颈。全篇使用严格的 LaTeX 公式渲染。

---

## 1. Transformer 注意力机制的复杂度推导

### 1.1 无 KV Cache 的自回归生成复杂度

在没有 KV Cache 的朴素自回归解码中，生成第 $t$ 个 token 时，模型需要重新计算前面所有 $t$ 个 token 的注意力。

单步 Attention 复杂度：

$$
\mathcal{O}(t \cdot t \cdot d_{\text{head}}) = \mathcal{O}(t^2 d_{\text{head}})
$$

生成长度为 $T$ 的整个序列时，累积 Attention 计算量为：

$$
\operatorname{TotalFLOPs} \propto \sum_{t=1}^{T} t^2 = \frac{T(T+1)(2T+1)}{6} = \mathcal{O}(T^3)
$$

结论：如果没有 KV Cache，长序列自回归生成的计算量会呈立方级爆炸。

### 1.2 引入 KV Cache 的自回归生成复杂度

引入 KV Cache 后，历史 token 的 Key 和 Value 被缓存下来。在生成第 $t$ 个 token 时，只需计算当前步的查询向量 $q_t$。

新查询投影的开销为：

$$
\mathcal{O}(d_{\text{model}} \cdot d_{\text{head}})
$$

当前步查询向量 $q_t$ 与历史 Key 做内积的复杂度为：

$$
\mathcal{O}(t \cdot d_{\text{head}})
$$

生成长度为 $T$ 的整个序列时，累积 Attention 计算量变为：

$$
\operatorname{TotalFLOPs} \propto \sum_{t=1}^{T} t = \frac{T(T+1)}{2} = \mathcal{O}(T^2)
$$

结论：KV Cache 将总时间复杂度从 $\mathcal{O}(T^3)$ 降至 $\mathcal{O}(T^2)$，单步复杂度降至线性 $\mathcal{O}(T)$。

---

## 2. Prefill 与 Decode 阶段的 FLOPs 精确计算

假设批次大小为 $B$，序列长度为 $T$（在 Decode 时 $T=1$），模型隐藏层维度为 $d_{\text{model}}$，注意力头数为 $H$，每个头的维度为 $d_{\text{head}}$，前馈网络中间层维度为 $d_{\text{ff}}$。

**注：在深度学习中，一次乘法加一次加法（Multiply-Accumulate, MAC）计为 2 个 FLOPs。**

### 2.1 单层 Attention 的计算量 (Prefill, 长度 $T$)

#### QKV 投影

$$
\operatorname{FLOPs}_{QKV} = 3 \times \left( 2 \cdot B \cdot T \cdot d_{\text{model}} \cdot (H \cdot d_{\text{head}}) \right) = 6 \cdot B \cdot T \cdot d_{\text{model}} \cdot H \cdot d_{\text{head}}
$$

若是 GQA 架构，KV 头数变为 $H_{\text{KV}}$，乘数 $3$ 会按共享比例变化。

#### Attention 分数矩阵 $QK^\top$

$Q$ 的形状为 $(B, H, T, d_{\text{head}})$，$K^\top$ 的形状为 $(B, H, d_{\text{head}}, T)$。

$$
\operatorname{FLOPs}_{\mathrm{score}} = 2 \cdot B \cdot H \cdot T \cdot T \cdot d_{\text{head}} = 2 \cdot B \cdot H \cdot T^2 \cdot d_{\text{head}}
$$

#### Attention 权重乘以 $V$

分数矩阵的形状为 $(B, H, T, T)$，$V$ 的形状为 $(B, H, T, d_{\text{head}})$。

$$
\operatorname{FLOPs}_{\mathrm{weighted}} = 2 \cdot B \cdot H \cdot T \cdot T \cdot d_{\text{head}} = 2 \cdot B \cdot H \cdot T^2 \cdot d_{\text{head}}
$$

#### 输出投影 $W^O$

$$
\operatorname{FLOPs}_{\mathrm{out}} = 2 \cdot B \cdot T \cdot (H \cdot d_{\text{head}}) \cdot d_{\text{model}}
$$

### 2.2 单层 FFN (以 SwiGLU 为例) 计算量

现代大模型（如 LLaMA）使用 SwiGLU 激活函数，包含三个权重矩阵：$W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$。

#### Gate 与 Up 投影

$$
\operatorname{FLOPs}_{\mathrm{gate+up}} = 2 \times \left( 2 \cdot B \cdot T \cdot d_{\text{model}} \cdot d_{\text{ff}} \right) = 4 \cdot B \cdot T \cdot d_{\text{model}} \cdot d_{\text{ff}}
$$

#### Down 投影

$$
\operatorname{FLOPs}_{\mathrm{down}} = 2 \cdot B \cdot T \cdot d_{\text{ff}} \cdot d_{\text{model}}
$$

#### 单层 FFN 总 FLOPs

$$
\text{FLOPs}_{\text{FFN}} = 6 \cdot B \cdot T \cdot d_{\text{model}} \cdot d_{\text{ff}}
$$

### 2.3 全模型计算量粗估公式

如果模型的总参数量为 $N$（仅算线性层的权重参数），在不考虑 Attention 矩阵运算的 $T^2$ 项时，生成（或处理）单个 token 的前向计算量大约为：

$$
\text{FLOPs}_{\text{per token}} \approx 2N
$$

对于长度为 $T$ 的 prompt 的 Prefill 阶段，总计算量粗估为：

$$
\text{FLOPs}_{\text{Prefill}} \approx 2N \cdot B \cdot T
$$

---

## 3. Roofline 模型与 AI (Arithmetic Intensity)

大模型推理的性能往往受限于硬件。Roofline 模型通过计算**算术强度 (Arithmetic Intensity, AI)** 来判断当前处于什么瓶颈。

### 3.1 核心定义

算术强度衡量的是**从显存中每读取 1 Byte 数据，能进行多少次浮点运算**：

$$
\text{AI} = \frac{\text{Total FLOPs}}{\text{Total Bytes Accessed}} \quad \text{(单位: FLOPs/Byte)}
$$

硬件实际性能公式：

$$
\text{Actual Performance} = \min \big( \text{Peak Compute}, \; \text{Peak Bandwidth} \times \text{AI} \big)
$$

### 3.2 硬件理论拐点 (以 NVIDIA A100/H100 为例)

| 硬件 | 峰值算力 (BF16 Tensor Core) | 峰值显存带宽 (HBM) | 拐点算术强度 (Ridge Point) |
|------|---------------------------|------------------|--------------------------|
| **A100 80GB** | $\approx 312 \text{ TFLOPS}$ | $\approx 2.0 \text{ TB/s}$ | $\frac{312 \times 10^{12}}{2.0 \times 10^{12}} = \mathbf{156 \text{ FLOPs/Byte}}$ |
| **H100 80GB** | $\approx 990 \text{ TFLOPS}$ | $\approx 3.35 \text{ TB/s}$ | $\frac{990 \times 10^{12}}{3.35 \times 10^{12}} = \mathbf{295 \text{ FLOPs/Byte}}$ |

如果当前计算的 $\text{AI} > \text{Ridge Point}$，则是**算力受限 (Compute-bound)**；反之则是**带宽受限 (Memory-bound)**。

### 3.3 Prefill vs Decode 瓶颈定性分析

#### Prefill 阶段：Compute-bound

在处理长 prompt 时，$T$ 很大，矩阵乘法会重复复用权重。从 HBM 读取一次模型权重 $W$，可以与 $B \times T$ 个 token 相乘。

$$
\operatorname{AI}_{\mathrm{Prefill}} \propto \frac{\mathcal{O}(N \cdot B \cdot T)}{\mathcal{O}(N)} \propto B \cdot T
$$

因此 Prefill 的算术强度很高，通常落在 Roofline 右侧，更接近算力受限。

#### Decode 阶段：Memory-bound

每次只生成 $T=1$ 个新 token，却仍然要搬运全部权重 $N$ 以及庞大的 KV Cache，只做很少的点积计算。

$$
\operatorname{AI}_{\mathrm{Decode}} \propto \frac{\mathcal{O}(N \cdot B \cdot 1)}{\mathcal{O}(N + \text{KVCache\_Size})} \approx \mathcal{O}(B)
$$

此时 AI 极低，通常远低于硬件拐点，因此 Decode 更容易受带宽限制。这也是 Continuous Batching 能明显改善吞吐量的理论基础。

---

## 4. FlashAttention 的 IO 复杂度优化

FlashAttention 并不减少 FLOPs 数，而是直接对准了 Roofline 左侧的 Memory-bound 问题，通过降低 HBM 读写次数（IO 复杂度）来加速。

### 4.1 传统 Attention 的 IO 致命伤

为了计算 $\text{Softmax}(QK^\top)V$，标准实现必须将巨大的中间矩阵 $S = QK^\top$（大小为 $T \times T$）写回 HBM，然后再读出来算 Softmax。

$$
\text{IO Complexity} = \mathcal{O}(T^2)
$$

当 $T=8192$ 时，这个 $\mathcal{O}(T^2)$ 读写的耗时甚至超过了计算本身的耗时。

### 4.2 FlashAttention 优化机制

FlashAttention 使用**分块 (Tiling)** 和 **在线 Softmax (Online Softmax)**，将计算拆分，使中间结果永远只停留在高速 SRAM 中（大小为 $M$）。
其 IO 复杂度为：

$$
\text{IO}_{\text{FlashAttention}} = \mathcal{O}\left( \frac{T^2 \cdot d_{\text{head}}}{M_{\text{sram}}} \right)
$$

由于 $M_{\text{sram}}$（如 A100 的 20MB 共享内存）远大于单个块，它使得 HBM 访存量呈数量级下降，把 Attention 从内存受限强行提到了更接近算力受限的水平。