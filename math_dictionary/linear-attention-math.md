# 线性注意力与高效注意力机制数学详解

> **核心定位**：从标准 Softmax 注意力的 $\mathcal{O}(T^2)$ 瓶颈出发，推导线性注意力的核函数近似原理，再深入 Mamba (SSM)、RetNet 和 RWKV 的递推更新公式。每种机制均给出严格的数学表达与复杂度分析。

---

## 1. 线性注意力的核心推导

### 1.1 从 Softmax 到核函数近似

标准注意力的核心计算为：
$$
O_i = \frac{\sum_{j=1}^T \exp\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right) v_j}{\sum_{j=1}^T \exp\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right)}
$$

**关键洞察**：$\exp(q^\top k / \sqrt{d})$ 本质上是一个**核函数 (Kernel)**。如果我们找到一个特征映射 $\phi: \mathbb{R}^d \to \mathbb{R}^D$，满足：
$$
\exp\!\left(\frac{q^\top k}{\sqrt{d}}\right) \approx \phi(q)^\top \phi(k)
$$

那么注意力公式可以重写为：
$$
O_i = \frac{\phi(q_i)^\top \sum_{j=1}^T \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_{j=1}^T \phi(k_j)}
$$

### 1.2 计算顺序的革命

定义两个**全局聚合量**（与查询位置 $i$ 无关，可以预先算好）：
$$
S = \sum_{j=1}^T \phi(k_j) v_j^\top \in \mathbb{R}^{D \times d_v}, \quad z = \sum_{j=1}^T \phi(k_j) \in \mathbb{R}^D
$$

那么每个位置 $i$ 的输出变为：
$$
O_i = \frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}
$$

- **标准注意力**：先算 $QK^\top$（$\mathcal{O}(T^2 d)$），再乘 $V$。
- **线性注意力**：先算 $\phi(K)^\top V$（$\mathcal{O}(T D d_v)$），再用 $\phi(Q)$ 查询。
- **复杂度**：从 $\mathcal{O}(T^2 d)$ 降为 $\mathcal{O}(T D d_v)$。当 $D, d_v \ll T$ 时，实现了对序列长度的**线性复杂度**。

---

## 2. RNN 递推视角 (Causal Linear Attention)

在因果（Causal）场景下，位置 $i$ 只能看到 $j \le i$。此时聚合量变为时间步相关的：

$$
S_t = S_{t-1} + \phi(k_t) v_t^\top \quad \text{(状态更新)}
$$
$$
z_t = z_{t-1} + \phi(k_t) \quad \text{(归一化项更新)}
$$
$$
O_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t} \quad \text{(输出查询)}
$$

这就是一个经典的 **RNN 递推形式**：
- **隐藏状态** $S_t \in \mathbb{R}^{D \times d_v}$，大小固定，不随序列长度增长。
- **单步推理复杂度**：$\mathcal{O}(D \cdot d_v)$，与序列长度 $T$ 完全无关。
- **致命缺陷**：累加 $S_t = S_{t-1} + \phi(k_t) v_t^\top$ 只增不减，模型**没有遗忘机制**，无法清除过期的历史信息。

---

## 3. 主流高效注意力变体

### 3.1 RetNet (Retentive Network)

> **出处**：Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models", 2023

RetNet 在线性注意力的递推中引入了**位置相关的指数衰减因子** $\gamma \in (0, 1)$：

$$
S_t = \gamma \cdot S_{t-1} + k_t v_t^\top
$$
$$
O_t = q_t^\top S_t
$$

- **物理意义**：距离当前位置越远的历史信息，衰减越严重（$\gamma^{t-j}$）。
- **三种等价计算形式**：
  1. **并行形式 (训练)**：$A_{ij} = q_i^\top k_j \cdot \gamma^{i-j}$，可用矩阵乘法并行计算。
  2. **递推形式 (推理)**：如上式，$\mathcal{O}(d^2)$ per token。
  3. **分块形式 (混合)**：块内并行 + 块间递推。

### 3.2 Mamba (Structured State Space Model, S6)

> **出处**：Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023

Mamba 基于连续状态空间模型 (SSM) 的离散化。其核心递推为：

$$
h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t x_t
$$
$$
y_t = C_t h_t + D x_t
$$

**与标准 SSM 的革命性区别**：矩阵 $\bar{A}_t, \bar{B}_t, C_t$ 都是**输入依赖的（Input-Dependent / Selective）**，即它们是当前 token $x_t$ 的函数。这赋予了 Mamba 根据内容决定"记住什么、遗忘什么"的能力。

- **隐藏状态** $h_t \in \mathbb{R}^{d \times N}$（$N$ 为 SSM 状态维度，通常 $16$–$64$）。
- **训练方式**：通过硬件高效的**并行前缀扫描 (Parallel Prefix Scan)** 算法在 GPU 上高效计算。
- **推理复杂度**：$\mathcal{O}(d \cdot N)$ per token，常数级，无 KV Cache。

### 3.3 RWKV

> **出处**：Peng et al., "RWKV: Reinventing RNNs for the Transformer Era", 2023

RWKV 使用 WKV (Weighted Key-Value) 机制替代标准注意力：

$$
\text{wkv}_t = \frac{\sum_{j=1}^{t-1} e^{-(t-1-j)w + k_j} v_j + e^{u+k_t} v_t}{\sum_{j=1}^{t-1} e^{-(t-1-j)w + k_j} + e^{u+k_t}}
$$

- $w > 0$ 是可学习的**时间衰减（Time Decay）**参数。
- $u$ 是可学习的**当前 token 加成 (Bonus)** 参数。
- 也可写成类似线性注意力的递推形式，推理时 $\mathcal{O}(d)$ per token。

---

## 4. 复杂度与 KV Cache 对比总表

| 方法 | 训练复杂度 | 推理 per token | 状态/Cache 大小 | 状态是否增长 |
|------|-----------|---------------|----------------|------------|
| **Softmax Attention** | $\mathcal{O}(T^2 d)$ | $\mathcal{O}(T \cdot d)$ | $\mathcal{O}(T \cdot d)$（KV Cache） | 线性增长 |
| **Linear Attention** | $\mathcal{O}(T d^2)$ | $\mathcal{O}(d^2)$ | $\mathcal{O}(d^2)$ | **固定** |
| **RetNet** | $\mathcal{O}(T d^2)$ | $\mathcal{O}(d^2)$ | $\mathcal{O}(d^2)$ | **固定** |
| **Mamba (S6)** | $\mathcal{O}(T d N)$ | $\mathcal{O}(d N)$ | $\mathcal{O}(d N)$ | **固定** |
| **RWKV** | $\mathcal{O}(T d)$ | $\mathcal{O}(d)$ | $\mathcal{O}(d)$ | **固定** |

---

## 5. 面试实战追问

**Q1：线性注意力为什么效果通常不如 Softmax 注意力？**
> 答：Softmax 注意力通过 $\exp$ 函数产生了极其**尖锐的（Sparse）** 注意力分布，能精确地聚焦到关键位置。线性注意力的核函数近似本质上会产生更平滑的注意力分布，在**关联检索 (Associative Recall)**——即"在大量干扰中精确找到特定信息"——这类任务上表现明显较差。这也是 Needle-in-a-Haystack 测试中线性模型的弱项。

**Q2：Mamba 的"选择性"机制到底解决了什么？**
> 答：传统 SSM 的 $A, B, C$ 矩阵对所有输入都一样（Input-Independent），这使得模型无法根据内容做出"该记住还是该遗忘"的判断。Mamba 让这些矩阵成为输入的函数（Input-Dependent），相当于给 SSM 装上了一个数据驱动的开关，按内容选择性地更新隐藏状态。

**Q3：混合架构（如 Jamba）为什么要"混搭" Attention 和 Mamba？**
> 答：Mamba 擅长高效处理长序列的局部和全局趋势，但在精确的长距离信息检索上不如 Softmax Attention。混合架构让部分层使用标准 Attention（保证检索精度），部分层使用 Mamba（降低整体复杂度和 KV Cache 需求），在质量和效率之间取得最优平衡。