# FlashAttention IO 优化数学详解

> **核心定位**：从 GPU 存储层次出发，严格推导 FlashAttention 为什么能在 FLOPs 不变的前提下实现 2–4× 的端到端加速。重点剖析 Online Softmax 的数学等价性证明、IO 复杂度的精确推导，以及 FA-2/3 的改进。

---

## 1. GPU 存储层次与 IO 瓶颈

理解 FlashAttention 之前，必须先理解 GPU 的**两级存储层次**：

| 存储层级 | 容量 (A100) | 带宽 | 特点 |
|---------|------------|------|------|
| **HBM** (High Bandwidth Memory) | 80 GB | $\sim 2 \text{ TB/s}$ | 大但相对慢 |
| **SRAM** (Shared Memory / L1) | $\sim 20 \text{ MB}$ | $\sim 19 \text{ TB/s}$ | 小但极快（$\sim 10\times$） |

所有矩阵 $Q, K, V$ 和输出 $O$ 都存储在 HBM 中。GPU 的计算单元（Tensor Core）只能直接访问 SRAM。因此，**每一次矩阵运算都需要先将数据从 HBM 搬到 SRAM，计算完成后再写回 HBM**。

---

## 2. 标准注意力的 IO 致命伤

标准注意力的计算分三步：
1. 计算注意力分数 $S = QK^\top \in \mathbb{R}^{T \times T}$ → 写回 HBM
2. 计算 Softmax $P = \text{softmax}(S) \in \mathbb{R}^{T \times T}$ → 读出 $S$，写回 $P$
3. 计算输出 $O = PV \in \mathbb{R}^{T \times d}$ → 读出 $P, V$

中间矩阵 $S$ 和 $P$ 都是 $T \times T$ 的巨大矩阵。以 $T = 8192, d = 128$ 为例：

$$
\text{中间矩阵大小} = T^2 = 67,108,864 \text{ 个元素} \approx 128 \text{ MB (FP16)}
$$

**总 HBM IO 复杂度**：
$$
\text{IO}_{\text{standard}} = \mathcal{O}(T^2 + T \cdot d) = \mathcal{O}(T^2) \quad (\text{因为通常 } T \gg d)
$$

当 $T$ 足够大时，光是搬运中间矩阵的 IO 开销，就已经远超计算本身的时间。

---

## 3. FlashAttention 的分块策略 (Tiling)

FlashAttention 的核心思想是：**永远不把 $T \times T$ 的中间矩阵写入 HBM**。

将 $Q, K, V$ 分别按行切分为大小为 $B_r$（行块）和 $B_c$（列块）的小块：

**外循环**遍历 $K, V$ 的列块 $(K_j, V_j)$；**内循环**遍历 $Q$ 的行块 $(Q_i)$：

$$
S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}
$$

这个小矩阵 $S_{ij}$ 完全在 SRAM 中，可以就地计算 Softmax 并累加到输出 $O_i$，无需写回 HBM。

---

## 4. Online Softmax 的数学等价性证明

分块计算 Softmax 的挑战在于：标准 Softmax 需要知道**整行**的最大值 $\max(z)$ 和求和 $\sum \exp(z)$。但分块时我们只能看到一部分。

### 4.1 流式更新公式

假设处理到第 $j$ 个列块后，我们维护了以下全局统计量：
- $m^{(j)}$：当前已见过的所有元素的行最大值
- $\ell^{(j)}$：当前的归一化分母（exp-sum）
- $O^{(j)}$：当前的输出累加值

当新的列块 $K_{j+1}, V_{j+1}$ 到来时：

$$
\tilde{S} = Q_i K_{j+1}^\top \quad \text{(局部注意力分数)}
$$
$$
\tilde{m} = \max(\tilde{S}) \quad \text{(局部最大值)}
$$
$$
m^{(j+1)} = \max\!\left(m^{(j)}, \tilde{m}\right) \quad \text{(全局最大值更新)}
$$
$$
\ell^{(j+1)} = \ell^{(j)} \cdot e^{m^{(j)} - m^{(j+1)}} + \sum \exp\!\left(\tilde{S} - m^{(j+1)}\right) \quad \text{(归一化分母更新)}
$$
$$
O^{(j+1)} = O^{(j)} \cdot \frac{\ell^{(j)}}{\ell^{(j+1)}} \cdot e^{m^{(j)} - m^{(j+1)}} + \frac{\exp\!\left(\tilde{S} - m^{(j+1)}\right)}{\ell^{(j+1)}} V_{j+1}
$$

### 4.2 等价性证明

当所有列块处理完毕后，$m^{(\text{final})} = \max_j(S_{ij})$ 就是整行的真实最大值，$\ell^{(\text{final})} = \sum_j \exp(S_{ij} - m^{(\text{final})})$ 就是真实的归一化分母。

因此：
$$
O_i^{(\text{final})} = \text{softmax}(S_i) \cdot V = \sum_j \frac{\exp(S_{ij} - m)}{\ell} V_j
$$

**数学上严格等价于全量 Softmax**（在浮点精度范围内）。

---

## 5. FlashAttention 的 IO 复杂度精确推导

设 SRAM 的大小为 $M$ 个元素。为了让 $Q_i, K_j, V_j, S_{ij}, O_i$ 都能装进 SRAM：

$$
B_c = \left\lceil \frac{M}{4d} \right\rceil, \quad B_r = \min\left(\left\lceil \frac{M}{4d} \right\rceil, d\right)
$$

外循环和内循环的总迭代次数为 $\frac{T}{B_r} \times \frac{T}{B_c}$。每次迭代从 HBM 读取的数据量为 $\mathcal{O}(B_r \cdot d + B_c \cdot d)$。

$$
\text{IO}_{\text{FlashAttention}} = \mathcal{O}\left(\frac{T^2 d^2}{M}\right)
$$

**与标准注意力的对比**：

$$
\frac{\text{IO}_{\text{standard}}}{\text{IO}_{\text{Flash}}} = \frac{T^2}{T^2 d^2 / M} = \frac{M}{d^2}
$$

以 A100 为例：$M \approx 10^5$ 个 FP16 元素，$d = 128$，所以 $M / d^2 \approx 6$。即 FlashAttention 的 HBM 读写量约为标准注意力的 $\frac{1}{6}$。

---

## 6. FlashAttention-2 改进

> Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023

核心改进：
1. **循环顺序调换**：外循环遍历 $Q$ 块，内循环遍历 $K, V$ 块。这使得输出 $O_i$ 在内循环中原地累加，减少了 HBM 写入次数。
2. **更好的 Warp 分工**：将工作在不同 Warp 之间按行分配（而非按列），减少了 Warp 间的同步和共享内存的 bank conflict。
3. **减少非矩阵乘法操作**：将 Softmax 的 rescale 操作延迟到最后统一执行。

**效果**：在 A100 上达到理论峰值 TFLOPS 的 **50%–73%**（FA-1 为 25%–40%）。

---

## 7. FlashAttention-3 改进 (H100 优化)

> Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", 2024

针对 H100 (Hopper 架构) 的三大优化：
1. **异步数据加载 (TMA)**：利用 Tensor Memory Accelerator 实现计算与 HBM→SRAM 数据搬运的**完全重叠**。
2. **FP8 低精度计算**：在 Tensor Core 上使用 FP8 进行矩阵乘法，并通过**非连贯处理（Incoherent Processing）**技巧控制量化误差。
3. **Warp 专业化**：将 Warp 分为 Producer（负责搬数据）和 Consumer（负责算矩阵乘），形成流水线。

---

## 8. FlashDecoding 与 Decode 阶段的特殊性

FlashAttention 主要加速的是 **Prefill 阶段**（$Q$ 矩阵很大）。在 Decode 阶段，$Q$ 只有 1 个 token（$T_q = 1$），此时并行度不在 $T_q$ 维度而在 $T_{\text{kv}}$ 维度。

**FlashDecoding** 的改进思路：
- 在 **KV 序列维度**上并行切分（而非 $Q$ 维度），每个线程块处理一段 KV Cache。
- 最后对各块的 partial output 做一次 Online Softmax 的 reduce 合并。
- 在长上下文场景下（$T_{\text{kv}} > 8K$），FlashDecoding 相比原始 FlashAttention 在 Decode 阶段可额外加速 $5$–$8\times$。