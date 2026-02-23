# 传统 MHA vs MLA：全流程数学与矩阵推导对比

> **摘要**：本文以严格的数学形式，完整推导 Multi-Head Attention (MHA) 与 Multi-head Latent Attention (MLA) 在 **训练/Prefill** 和 **推理/Decode** 两个阶段的每一步计算。重点阐明 MLA 在 Decode 阶段通过**矩阵吸收（Matrix Absorption）**技术，实现"永不解压 KV Cache"这一核心工程洞见的数学原理。

---

## 符号约定

| 符号 | 含义 | 典型值 (DeepSeek-V2) |
|------|------|---------------------|
| $L$ | 序列长度 (Sequence Length) | 可变 |
| $d$ | 模型隐藏维度 (d_model) | 5120 |
| $n_h$ | 注意力头数 | 128 |
| $d_h$ | 每头维度 $= d / n_h$ | 128 |
| $d_c$ | MLA 的 KV 潜在空间维度 | 512 |
| $d_c'$ | MLA 的 Q 潜在空间维度 | 1536 |
| $d_r$ | 解耦 RoPE 维度 | 64 |

---

## 一、输入定义

两种机制共享完全相同的输入：

$$X \in \mathbb{R}^{L \times d}$$

其中 $X$ 是经过 Embedding + 前序层 Transformer Block 计算后、进入当前注意力层的隐状态矩阵。每一行 $x_t \in \mathbb{R}^{1 \times d}$ 代表第 $t$ 个 token 的表示。

---

## 二、训练 / Prefill 阶段

在 Prefill 阶段，模型一次性处理完整序列 $X$（长度为 $L$），计算所有位置的注意力并写入 KV Cache。

### 2.1 Query 投影

#### MHA

对每个头 $i \in \{1, \ldots, n_h\}$ 独立投影：

$$Q_i = X \, W_i^Q \quad \in \mathbb{R}^{L \times d_h}$$

其中 $W_i^Q \in \mathbb{R}^{d \times d_h}$。实际实现中，所有头的 $W_i^Q$ 被拼接为一个大矩阵 $W^Q \in \mathbb{R}^{d \times (n_h \cdot d_h)}$，一次矩阵乘法完成。

#### MLA

MLA 对 Query 也引入潜在空间压缩（与 KV 路径对称）：

$$c^Q = X \, W^{DQ} \quad \in \mathbb{R}^{L \times d_c'}$$

$$Q_i^C = c^Q \, W_i^{UQ} \quad \in \mathbb{R}^{L \times d_h}$$

其中：
- $W^{DQ} \in \mathbb{R}^{d \times d_c'}$ 是 Q 的**下投影矩阵**（Down-projection）
- $W_i^{UQ} \in \mathbb{R}^{d_c' \times d_h}$ 是第 $i$ 个头的**上投影矩阵**（Up-projection）
- $d_c'$ 是 Q 的潜在维度（DeepSeek-V2 中 $d_c' = 1536$）

> **💡 为什么 Q 也要压缩？** 在 Prefill 阶段，Q 压缩节省的计算量有限。但这为后续 Decode 阶段的矩阵吸收提供了关键的数学前提——使得 Q 和 K 能在同一个低维空间里直接做点积。

### 2.2 Key、Value 投影（⭐ 核心差异点）

#### MHA：每个头独立投影，生成高维 K、V

$$K_i = X \, W_i^K \quad \in \mathbb{R}^{L \times d_h}$$

$$V_i = X \, W_i^V \quad \in \mathbb{R}^{L \times d_h}$$

其中 $W_i^K, W_i^V \in \mathbb{R}^{d \times d_h}$。全部 $n_h$ 个头产生 $n_h$ 套独立的 K、V 矩阵。

#### MLA：统一下投影到潜在空间（不分头！）

$$\boxed{C^{KV} = X \, W^{DKV} \quad \in \mathbb{R}^{L \times d_c}}$$

其中 $W^{DKV} \in \mathbb{R}^{d \times d_c}$。这是 MLA 最核心的一步——所有 $n_h$ 个头的 Key 和 Value 信息，被**联合压缩**到一个维度仅为 $d_c$ 的共享潜在向量中。

> **数学本质**：$W^{DKV}$ 等价于对 $\text{Concat}(W^K, W^V) \in \mathbb{R}^{d \times 2n_h d_h}$ 做了一次最优低秩近似。前提条件是 KV 矩阵的固有秩（Intrinsic Rank）远小于 $2n_h d_h$——这已被 SVD 实验反复验证（见 `mla_latent_space_analysis.ipynb` 实验 1）。

### 2.3 K、V 解压

#### MHA

**不需要**——直接使用 §2.2 中投影得到的高维矩阵。

#### MLA：对每个头独立上投影

$$K_i^C = C^{KV} \, W_i^{UK} \quad \in \mathbb{R}^{L \times d_h}$$

$$V_i^C = C^{KV} \, W_i^{UV} \quad \in \mathbb{R}^{L \times d_h}$$

其中 $W_i^{UK}, W_i^{UV} \in \mathbb{R}^{d_c \times d_h}$。

> **理解要点**：在 Prefill 阶段，MLA 确实需要"解压"出完整的 K 和 V 来计算注意力分数。但关键在于——**解压后的结果不缓存**！KV Cache 中只保存压缩后的 $C^{KV}$。

### 2.4 位置编码（RoPE）

#### MHA：直接应用在全量 Q、K 上

$$\hat{Q}_i = \text{RoPE}(Q_i) \quad \in \mathbb{R}^{L \times d_h}$$

$$\hat{K}_i = \text{RoPE}(K_i) \quad \in \mathbb{R}^{L \times d_h}$$

RoPE 对每一对相邻维度应用旋转：

$$\text{RoPE}(x)_{2j:2j+1} = \begin{pmatrix} \cos(t \theta_j) & -\sin(t \theta_j) \\ \sin(t \theta_j) & \cos(t \theta_j) \end{pmatrix} \begin{pmatrix} x_{2j} \\ x_{2j+1} \end{pmatrix}$$

其中 $t$ 是 token 的位置索引，$\theta_j = 10000^{-2j/d_h}$。

#### MLA：解耦 RoPE（⭐ 关键设计）

MLA 不能对 $C^{KV}$ 直接施加 RoPE，因为：
1. $C^{KV}$ 是共享的潜在向量，不是任何特定头的 K
2. RoPE 是位置相关的旋转，会破坏潜在空间的低秩结构（见 `mla_latent_space_analysis.ipynb` 实验 3：Toeplitz 偏差增大 5.6 倍）

**解决方案——解耦（Decouple）**：

单独投影出一组小维度的 RoPE Key 和 RoPE Query：

$$K^R = X \, W^{KR} \quad \in \mathbb{R}^{L \times d_r}$$

$$Q^R = X \, W^{QR} \quad \in \mathbb{R}^{L \times d_r}$$

分别施加 RoPE 后，与语义部分拼接：

$$\hat{Q}_i = \left[\; Q_i^C \;\|\; \text{RoPE}(Q^R) \;\right] \quad \in \mathbb{R}^{L \times (d_h + d_r)}$$

$$\hat{K}_i = \left[\; K_i^C \;\|\; \text{RoPE}(K^R) \;\right] \quad \in \mathbb{R}^{L \times (d_h + d_r)}$$

> **设计哲学**：语义信息（who/what）走压缩路径；位置信息（where）走独立路径。两条路径在注意力计算时自然融合。

### 2.5 Attention 计算

#### MHA

$$S_i = \text{softmax}\!\left(\frac{\hat{Q}_i \, \hat{K}_i^\top}{\sqrt{d_h}} + M\right) \quad \in \mathbb{R}^{L \times L}$$

$$O_i = S_i \, V_i \quad \in \mathbb{R}^{L \times d_h}$$

其中 $M$ 是因果掩码（$M_{st} = 0$ if $s \geq t$, else $-\infty$）。

#### MLA

$$S_i = \text{softmax}\!\left(\frac{\hat{Q}_i \, \hat{K}_i^\top}{\sqrt{d_h + d_r}} + M\right) \quad \in \mathbb{R}^{L \times L}$$

$$O_i = S_i \, V_i^C \quad \in \mathbb{R}^{L \times d_h}$$

> **注意缩放因子**：MLA 的分母是 $\sqrt{d_h + d_r}$ 而非 $\sqrt{d_h}$，因为 Q 和 K 的拼接维度是 $d_h + d_r$。

### 2.6 Output 投影

两种方式完全一致：

$$O = \text{Concat}(O_1, O_2, \ldots, O_{n_h}) \, W^O \quad \in \mathbb{R}^{L \times d}$$

其中 $W^O \in \mathbb{R}^{(n_h \cdot d_h) \times d}$。

---

## 三、KV Cache 阶段

这是 MLA 相对 MHA 的核心优势所在。

### 3.1 物理存储内容

#### MHA

缓存所有头的完整 K 和 V（已施加 RoPE 的 K）：

$$\text{Cache}_{\text{MHA}} = \left\{\; \hat{K}_i,\; V_i \;\right\}_{i=1}^{n_h}$$

#### MLA

仅缓存唯一的潜在向量和解耦的 RoPE Key：

$$\boxed{\text{Cache}_{\text{MLA}} = \left\{\; C^{KV},\; \text{RoPE}(K^R) \;\right\}}$$

> **⭐ 关键洞见**：MLA 的 Cache 是**头无关的（Head-agnostic）**。无论有多少个头，Cache 体积不变！

### 3.2 显存占用对比（每层每 token）

#### MHA

$$\text{Mem}_{\text{MHA}} = 2 \times n_h \times d_h = 2d$$

以 DeepSeek-V2 为例（$d = 5120, n_h = 128, d_h = 128$）：

$$2 \times 128 \times 128 = 32768 \;\text{维} \quad \xrightarrow{\text{FP16}} \quad 65536 \;\text{bytes} = 64 \;\text{KB/token/layer}$$

#### MLA

$$\text{Mem}_{\text{MLA}} = d_c + d_r$$

$$512 + 64 = 576 \;\text{维} \quad \xrightarrow{\text{FP16}} \quad 1152 \;\text{bytes} \approx 1.1 \;\text{KB/token/layer}$$

**压缩比：$32768 / 576 \approx 56.9 \times$**

### 3.3 全模型 KV Cache 对比

假设 60 层，序列长度 128K，FP16：

| 指标 | MHA | MLA | 压缩比 |
|------|-----|-----|-------|
| 每 token 总维度 | $60 \times 32768 = 1,966,080$ | $60 \times 576 = 34,560$ | 56.9× |
| 128K 序列总显存 | $\approx 477$ GB | $\approx 8.4$ GB | 56.9× |

> MHA 方案在 128K 上下文时需要 **477 GB** 仅用于 KV Cache——这远超任何单卡的显存容量。MLA 将其压缩到 **8.4 GB**，在单张 A100 80GB 上从容运行。

---

## 四、推理 / Decode 阶段（⭐⭐ 矩阵吸收的核心）

Decode 阶段是自回归生成：每一步仅输入一个新 token $x_t \in \mathbb{R}^{1 \times d}$，需要与所有历史 token 的 KV Cache 交互。

**这是 MLA 工程实现中最精妙的部分——通过矩阵结合律，将"解压 Cache"的计算完全消除。**

### 4.1 计算新 Query

#### MHA

$$q_{t,i} = \text{RoPE}\!\left(x_t \, W_i^Q\right) \quad \in \mathbb{R}^{1 \times d_h}$$

#### MLA

$$c_t^Q = x_t \, W^{DQ} \quad \in \mathbb{R}^{1 \times d_c'}$$

$$q_{t,i}^C = c_t^Q \, W_i^{UQ} \quad \in \mathbb{R}^{1 \times d_h}$$

$$q_{t,i}^R = \text{RoPE}\!\left(x_t \, W^{QR}\right) \quad \in \mathbb{R}^{1 \times d_r}$$

### 4.2 重构 K（准备点积）

#### MHA

直接从 Cache 读取高维 K：

$$\hat{K}_i^{\text{cached}} \in \mathbb{R}^{L \times d_h} \quad \text{（已存储）}$$

#### MLA

理论上需要解压：

$$K_i^C = C^{KV} \, W_i^{UK} \quad \in \mathbb{R}^{L \times d_h}$$

**但实际推理中——绝对不执行这一步！**

### 4.3 矩阵吸收：Q 融合（⭐⭐⭐ MLA 的核心魔法）

#### MHA

无此步骤。

#### MLA

**关键推导**——先写出语义部分的注意力分数：

$$\text{Score}_{\text{semantic}} = q_{t,i}^C \cdot (K_i^C)^\top$$

展开 $K_i^C = C^{KV} \, W_i^{UK}$：

$$= q_{t,i}^C \cdot \left(C^{KV} \, W_i^{UK}\right)^\top$$

$$= q_{t,i}^C \cdot (W_i^{UK})^\top \cdot (C^{KV})^\top$$

**利用矩阵乘法的结合律**，将 $q_{t,i}^C$ 和 $(W_i^{UK})^\top$ 先结合：

$$\boxed{\tilde{q}_{t,i}^C = q_{t,i}^C \cdot (W_i^{UK})^\top \quad \in \mathbb{R}^{1 \times d_c}}$$

此时：

$$\text{Score}_{\text{semantic}} = \tilde{q}_{t,i}^C \cdot (C^{KV})^\top$$

> **深刻理解**：$\tilde{q}_{t,i}^C$ 是一个 $d_c = 512$ 维的向量——**它已经从高维的 Query 空间变换到了低维的潜在空间**。这意味着我们可以直接用这个低维 Query 与低维的 Cache $C^{KV}$ 做点积，**完全绕过了 K 的解压过程**！
>
> 从物理意义上讲：$W_i^{UK}$ 矩阵（原本负责"解压"K）被"吸收"到了 Query 的变换中。Q 自己戴上了一副"滤镜"，使得它能直接看懂压缩后的语言。

### 4.4 Attention 点积

#### MHA

$$\text{Score}_{t,i} = q_{t,i} \cdot \left(\hat{K}_i^{\text{cached}}\right)^\top \quad \in \mathbb{R}^{1 \times L}$$

计算复杂度：$O(d_h \times L)$。读取 Cache 体积：$O(n_h \times d_h \times L)$。

#### MLA

$$\boxed{\text{Score}_{t,i} = \underbrace{\tilde{q}_{t,i}^C \cdot (C^{KV})^\top}_{\text{语义分数（潜在空间内）}} + \underbrace{q_{t,i}^R \cdot \left(\text{RoPE}(K^R)\right)^\top}_{\text{位置分数}}}$$

语义部分：$\tilde{q}_{t,i}^C \in \mathbb{R}^{1 \times d_c}$，$C^{KV} \in \mathbb{R}^{L \times d_c}$ → 复杂度 $O(d_c \times L)$。

位置部分：$q_{t,i}^R \in \mathbb{R}^{1 \times d_r}$，$\text{RoPE}(K^R) \in \mathbb{R}^{L \times d_r}$ → 复杂度 $O(d_r \times L)$。

**读取 Cache 总体积：$O((d_c + d_r) \times L)$**——仅为 MHA 的 $1/56.9$！

> **工程意义**：Decode 阶段是 **Memory-Bandwidth Bound**（带宽瓶颈）——瓶颈在于从 HBM 读取 KV Cache 的速度，而非 GPU 的计算能力。MLA 将 Cache 读取量压缩 57 倍，直接将 Decode 的吞吐量提升了一个数量级。

### 4.5 计算 V 的加权和

#### MHA

$$o_{t,i} = \text{softmax}(\text{Score}_{t,i}) \cdot V_i^{\text{cached}} \quad \in \mathbb{R}^{1 \times d_h}$$

读取 Cache 中的 $V_i \in \mathbb{R}^{L \times d_h}$——庞大的内存 IO。

#### MLA

理论上需要：

$$o_{t,i} = \text{softmax}(\text{Score}_{t,i}) \cdot \underbrace{(C^{KV} \, W_i^{UV})}_{\text{解压出的 } V_i^C} \quad \in \mathbb{R}^{1 \times d_h}$$

**实际操作（第二次矩阵吸收）**：

利用结合律，**先在潜在空间完成加权求和，再解压**：

$$u_{t,i} = \text{softmax}(\text{Score}_{t,i}) \cdot C^{KV} \quad \in \mathbb{R}^{1 \times d_c}$$

此时 $u_{t,i}$ 是一个 $d_c = 512$ 维的"潜在空间中的注意力输出"。它仅需读取极小的 $C^{KV}$ 矩阵。

然后：

$$o_{t,i} = u_{t,i} \cdot W_i^{UV} \quad \in \mathbb{R}^{1 \times d_h}$$

> **第二次吸收的本质**：我们不是"读取大矩阵 $V_i^C$"，而是"读取小矩阵 $C^{KV}$，然后用固定权重 $W_i^{UV}$ 映射"。$W_i^{UV}$ 是模型的静态权重，常驻显存，不随序列长度增长。

### 4.6 矩阵吸收：O 融合（第三次吸收）

#### MHA

$$o_t = \text{Concat}(o_{t,1}, \ldots, o_{t,n_h}) \cdot W^O = \sum_{i=1}^{n_h} o_{t,i} \, W_i^O$$

其中 $W_i^O \in \mathbb{R}^{d_h \times d}$ 是 $W^O$ 的第 $i$ 个头对应的切片。

#### MLA

展开 $o_{t,i}$：

$$o_t = \sum_{i=1}^{n_h} \left(u_{t,i} \cdot W_i^{UV}\right) \cdot W_i^O$$

再次利用矩阵乘法结合律：

$$\boxed{o_t = \sum_{i=1}^{n_h} u_{t,i} \cdot \underbrace{\left(W_i^{UV} \cdot W_i^O\right)}_{W_i^{UV\_O} \;\in\; \mathbb{R}^{d_c \times d}}}$$

**预计算融合权重**：

$$W_i^{UV\_O} = W_i^{UV} \cdot W_i^O \quad \in \mathbb{R}^{d_c \times d}$$

这个融合权重在模型加载时**一次性计算并缓存**，推理过程中直接使用。

> **第三次吸收的意义**：消除了 $u_{t,i} \to o_{t,i}$（$d_c \to d_h$）和 $o_{t,i} \to$ output（$d_h \to d$）这两步中间计算，将路径缩短为 $u_{t,i} \to$ output（$d_c \to d$）。

---

## 五、三次矩阵吸收总结

| 吸收步骤 | 原始计算 | 吸收后计算 | 消除的中间量 |
|----------|---------|-----------|------------|
| **Q 吸收** | $q^C \cdot (C^{KV} W^{UK})^\top$ | $\tilde{q}^C \cdot (C^{KV})^\top$ | $K_i^C \in \mathbb{R}^{L \times d_h}$ |
| **V 吸收** | $\alpha \cdot (C^{KV} W^{UV})$ | $(\alpha \cdot C^{KV}) \cdot W^{UV}$ | $V_i^C \in \mathbb{R}^{L \times d_h}$ |
| **O 吸收** | $u \cdot W^{UV} \cdot W^O$ | $u \cdot W^{UV\_O}$ | $o_i \in \mathbb{R}^{1 \times d_h}$ |

> 其中 $\alpha = \text{softmax}(\text{Score}) \in \mathbb{R}^{1 \times L}$。

**数学上的统一视角**：三次吸收的本质都是**矩阵乘法的结合律** $(AB)C = A(BC)$。MLA 的设计使得 KV Cache 中的 $C^{KV}$ 总是出现在矩阵链的中间位置，因此可以通过改变结合顺序，将解压矩阵 $W^{UK}, W^{UV}$ "推"到 $C^{KV}$ 的另一侧——要么吸收到 Query 中（Q 吸收），要么吸收到输出权重中（V+O 吸收）。

---

## 六、Decode 阶段计算量与带宽对比

### 6.1 单步 Decode 内存读取量

| 操作 | MHA | MLA |
|------|-----|-----|
| 读 K Cache | $n_h \times d_h \times L$ | $d_c \times L + d_r \times L$ |
| 读 V Cache | $n_h \times d_h \times L$ | $d_c \times L$（与 K 共享同一 $C^{KV}$） |
| **总读取维度** | $2 \times n_h \times d_h \times L$ | $(2d_c + d_r) \times L$ |

以 DeepSeek-V2 参数为例（$n_h=128, d_h=128, d_c=512, d_r=64$）：

| | MHA | MLA | 比值 |
|--|-----|-----|------|
| 每 token 读取维度 | $2 \times 128 \times 128 = 32768$ | $2 \times 512 + 64 = 1088$ | **30.1×** |

> **注意**：MLA 在计算 Score 和 V 加权和时，$C^{KV}$ 需要被读取两次（一次算 Score，一次算 $u$）。但由于它非常小（$512$ 维 vs MHA 的 $128 \times 128 = 16384$ 维），**总读取量依然远小于 MHA**。在实际 GPU 实现中，$C^{KV}$ 的两次读取可以利用 L2 Cache 命中，进一步减少 HBM 访问。

### 6.2 Decode 吞吐量提升的数学解释

Decode 是典型的 **Memory-Bandwidth Bound** 计算。根据 Roofline 模型：

$$\text{Decode Latency} \propto \frac{\text{Total Bytes Read from HBM}}{\text{HBM Bandwidth}}$$

由于 MLA 将读取量压缩了约 $30\times$，在相同的 HBM 带宽下：

$$\text{Speedup}_{\text{MLA}} \approx \frac{\text{Mem}_{\text{MHA}}}{\text{Mem}_{\text{MLA}}} = \frac{32768}{1088} \approx 30\times$$

实际测试中的加速比会略低于理论值（因为还有模型权重读取、RoPE 计算等开销），但 DeepSeek 报告的 Decode 吞吐量提升在 **5-10 倍** 量级，已经是非常显著的提升。

---

## 七、预计算融合权重清单

在模型加载（Model Loading）阶段，推理引擎需要预计算以下融合权重：

| 原始权重 | 融合公式 | 融合后维度 | 用途 |
|---------|---------|-----------|------|
| $W_i^{UQ}, W^{DQ}$ | $W_i^{Q\_fused} = W^{DQ} \cdot W_i^{UQ}$ | $d \times d_h$ | Decode Q 计算 |
| $W_i^{UK}$ | $\tilde{W}_i^{QK} = W_i^{UQ\top} \cdot (W_i^{UK})^\top$ | $d_h \times d_c$ | Q 吸收 |
| $W_i^{UV}, W_i^O$ | $W_i^{UV\_O} = W_i^{UV} \cdot W_i^O$ | $d_c \times d$ | V+O 吸收 |

> **工程实践**：这些预计算可以在模型加载时用几秒钟完成（只需做一次矩阵乘法），之后整个推理过程中无需再操作 $W^{UK}$ 和 $W^{UV}$。某些实现甚至会将 $W^{UK}$ 和 $W^{UV}$ 从显存中释放，进一步节省空间。

---

## 八、MLA Decode 完整计算流（伪代码）

```python
# ─── 模型加载时（一次性预计算） ───
for each head i:
    W_QK_absorbed[i] = W_UQ[i].T @ W_UK[i].T     # Q 吸收 K 的解压矩阵
    W_UVO[i]         = W_UV[i] @ W_O[i]           # V 解压矩阵吸收 O 矩阵

# ─── 每步 Decode ───
def decode_step(x_t, cache_C_KV, cache_K_rope):
    """
    x_t:        [1, d]          当前 token 的隐状态
    cache_C_KV: [L, d_c]        历史潜在向量 (512 维)
    cache_K_rope: [L, d_r]      历史 RoPE Key (64 维)
    """
    # 1. Query 路径
    c_q = x_t @ W_DQ                              # [1, d_c']
    q_rope = RoPE(x_t @ W_QR)                     # [1, d_r]

    for each head i:
        # 2. Q 吸收：直接变换到潜在空间
        q_semantic = c_q @ W_QK_absorbed[i]        # [1, d_c]  ← 不是 d_h！

        # 3. Attention Score（在潜在空间计算）
        score_semantic = q_semantic @ cache_C_KV.T # [1, L]    ← 读取极小 Cache
        score_position = q_rope @ cache_K_rope.T   # [1, L]
        score = (score_semantic + score_position) / sqrt(d_h + d_r)

        # 4. V 加权和（在潜在空间计算）
        alpha = softmax(score)                     # [1, L]
        u_i = alpha @ cache_C_KV                   # [1, d_c]  ← 再次读取小 Cache

        # 5. V+O 吸收：直接从潜在空间映射到输出
        output_i = u_i @ W_UVO[i]                  # [1, d]    ← 一步到位

    # 6. 汇总所有头
    output = sum(output_i for all heads)
    return output
```

> **对比 MHA Decode**：在 MHA 中，步骤 3 需要读取 $\hat{K}_i \in \mathbb{R}^{L \times d_h}$（128 维 × $L$ 个 token × 128 个头），步骤 4 需要读取 $V_i \in \mathbb{R}^{L \times d_h}$。MLA 的步骤 3 和 4 都只读取 $C^{KV} \in \mathbb{R}^{L \times 512}$（共享，不分头）。

---

## 九、Why It Works：数学保证

MLA 的矩阵吸收能够"无损"工作，依赖以下数学事实：

### 9.1 结合律的等价性

对于任意矩阵 $A, B, C$（维度兼容时）：

$$(AB)C = A(BC)$$

这是线性代数的基本定理。矩阵吸收**不改变任何数值结果**——只改变了计算顺序。

### 9.2 低秩假设的合理性

MLA 的 $C^{KV} = X W^{DKV}$ 构成一个 $d \to d_c$ 的线性瓶颈。根据 Eckart-Young 定理，当 KV 矩阵的有效秩 $r_{\text{eff}} \leq d_c$ 时，投影误差为零：

$$\left\| \hat{K} - C^{KV} W^{UK} \right\|_F^2 = \sum_{j > d_c} \sigma_j^2$$

其中 $\sigma_j$ 是 KV 矩阵的第 $j$ 个奇异值。当 $\sigma_j$ 对 $j > d_c$ 迅速衰减到 $\approx 0$ 时（这已被实验 1 中的 SVD 分析证实），投影误差趋近于零。

### 9.3 端到端训练的自适应性

与固定的 SVD 截断不同，MLA 是**端到端训练**的。$W^{DKV}$ 和 $W^{UK}_i, W^{UV}_i$ 通过反向传播联合优化，使得模型**自动学习**最优的压缩-解压映射。这意味着即使 KV 矩阵的理论有效秩略大于 $d_c$，端到端训练也能通过放弃"对下游任务无贡献"的维度来适应瓶颈约束。

---

## 十、一句话总结

| | MHA | MLA |
|--|-----|-----|
| **KV Cache 存什么** | 每个头的完整 K 和 V | 一个共享的低维潜在向量 $C^{KV}$ |
| **Decode 时怎么读** | 逐头读取大量高维 K/V | 直接在潜在空间做点积和求和 |
| **核心数学技巧** | 无 | 矩阵乘法结合律：$(qW^{UK\top})C^{KV\top}$ |
| **瓶颈** | Memory Bandwidth Bound | 权重计算 Bound（更容易并行） |
| **KV Cache 大小** | $O(n_h \cdot d_h \cdot L)$ | $O(d_c \cdot L)$ |

> **MLA 的哲学**：不要在推理时做"解压"这种可以预计算的事情。通过数学上的结合律，**把所有需要解压的工作转嫁到了模型的静态权重上**（预计算一次，永久使用），从而让 Decode 阶段的内存 IO 降到理论最小值。

---

## 参考文献

1. DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* arXiv:2405.04434, 2024.
2. DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024.
3. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
4. Shazeer, N. *Multi-Query Attention Is All You Need.* arXiv:1911.02150, 2019.
5. Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023.
6. Su, J. et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing, 2024.
