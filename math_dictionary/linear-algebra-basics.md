# 线性代数基础速查（LLM 相关）

> **核心定位**：为 LLM 推理系统中频繁出现的线性代数操作提供严格的数学定义、FLOPs 计算和几何直觉。覆盖矩阵乘法、SVD、范数、Softmax 梯度、余弦相似度等核心概念，每个概念都直接关联到 Transformer 中的实际应用。

---

## 1. 矩阵乘法与 FLOPs

### 1.1 基本公式

$$
C = AB, \quad A \in \mathbb{R}^{m \times k}, \; B \in \mathbb{R}^{k \times n} \quad \Rightarrow \quad C \in \mathbb{R}^{m \times n}
$$

$$
C_{ij} = \sum_{\ell=1}^{k} A_{i\ell} B_{\ell j}
$$

### 1.2 FLOPs 计算

每个输出元素 $C_{ij}$ 需要 $k$ 次乘法 + $(k-1)$ 次加法 $\approx 2k$ 次浮点操作。

$$
\boxed{\text{FLOPs}(A \times B) = 2mkn}
$$

### 1.3 在 Transformer 中的应用

| 操作 | 矩阵 $A$ | 矩阵 $B$ | FLOPs |
|------|----------|----------|:-----:|
| QKV 投影 | $X \in \mathbb{R}^{BT \times d}$ | $W_{QKV} \in \mathbb{R}^{d \times 3d}$ | $6BTd^2$ |
| 注意力分数 | $Q \in \mathbb{R}^{BHT \times d_h}$ | $K^\top \in \mathbb{R}^{d_h \times T}$ | $2BHT^2 d_h$ |
| 注意力输出 | $P \in \mathbb{R}^{BHT \times T}$ | $V \in \mathbb{R}^{T \times d_h}$ | $2BHT^2 d_h$ |
| FFN (SwiGLU) | $X \in \mathbb{R}^{BT \times d}$ | $W_{\text{gate/up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ | $2 \times 2BTd \cdot d_{\text{ff}}$ |

---

## 2. 向量点积与几何意义

$$
a \cdot b = \sum_i a_i b_i = \|a\| \|b\| \cos\theta
$$

- $\theta = 0$：完全同向（点积最大）。
- $\theta = \pi/2$：正交（点积为零）。
- $\theta = \pi$：完全反向（点积最小）。

**在注意力中**：$Q K^\top$ 本质上计算的是 Query 和 Key 的**余弦相似度**（经 $\sqrt{d}$ 缩放后）。

---

## 3. Softmax 函数与梯度

### 3.1 前向

$$
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}, \quad \sum_i p_i = 1, \; p_i > 0
$$

### 3.2 Jacobian 矩阵

$$
\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j) = \begin{cases}
p_i(1 - p_i) & i = j \\
-p_i p_j & i \ne j
\end{cases}
$$

其中 $\delta_{ij}$ 是 Kronecker delta。

### 3.3 Safe Softmax

为了数值稳定性，减去最大值：
$$
p_i = \frac{\exp(z_i - \max(z))}{\sum_j \exp(z_j - \max(z))}
$$

数学上等价，但避免了 $\exp$ 溢出。FlashAttention 的 Online Softmax 就是基于这个性质。

---

## 4. 范数（Norm）

### 4.1 向量范数

$$
\|x\|_1 = \sum_i |x_i|, \quad \|x\|_2 = \sqrt{\sum_i x_i^2}, \quad \|x\|_\infty = \max_i |x_i|
$$

### 4.2 矩阵范数

$$
\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2} = \sqrt{\text{tr}(A^\top A)} = \sqrt{\sum_i \sigma_i^2}
$$

$\sigma_i$ 是 $A$ 的奇异值。Frobenius 范数等价于将矩阵"展平"后的 L2 范数。

$$
\|A\|_* = \sum_i \sigma_i \quad (\text{核范数 / Nuclear Norm})
$$

核范数最小化 = 秩最小化的最佳**凸松弛**。GD 的隐式偏置倾向于核范数最小 → 权重倾向低秩。

### 4.3 梯度裁剪（Gradient Clipping）

$$
g' = \begin{cases}
g & \|g\|_2 \le G_{\max} \\
g \cdot \frac{G_{\max}}{\|g\|_2} & \|g\|_2 > G_{\max}
\end{cases}
$$

LLM 训练中通常 $G_{\max} = 1.0$，防止梯度爆炸。

---

## 5. SVD（奇异值分解）

### 5.1 定义

$$
A = U \Sigma V^\top
$$

- $U \in \mathbb{R}^{m \times m}$：左奇异向量（正交矩阵）
- $\Sigma \in \mathbb{R}^{m \times n}$：奇异值矩阵（对角，$\sigma_1 \ge \sigma_2 \ge \dots \ge 0$）
- $V \in \mathbb{R}^{n \times n}$：右奇异向量（正交矩阵）

### 5.2 最优低秩近似 (Eckart-Young 定理)

$$
\boxed{A_r = U_r \Sigma_r V_r^\top = \arg\min_{\text{rank}(B) \le r} \|A - B\|_F}
$$

保留前 $r$ 个奇异值就是 Frobenius 范数意义下的**最优秩-$r$ 近似**。

**LoRA 的理论基础**：微调的权重增量 $\Delta W$ 通常是低秩的，LoRA 等价于直接学习 $\Delta W = BA$（一个秩-$r$ 分解）。

### 5.3 条件数

$$
\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}
$$

条件数大 → 矩阵"病态" → 数值计算不稳定。

---

## 6. 特征值与特征向量

$$
Av = \lambda v, \quad v \ne 0
$$

- 对称矩阵的特征值全为实数。
- 协方差矩阵 $\Sigma = \frac{1}{n}X^\top X$ 的特征分解 = PCA。
- **LLM 中的意义**：激活值协方差矩阵的大特征值 → 离群维度（Outlier Dimension） → 量化难点。

---

## 7. 余弦相似度

$$
\cos(a, b) = \frac{a \cdot b}{\|a\|_2 \cdot \|b\|_2} \in [-1, 1]
$$

与 L2 距离的关系：
$$
\|a - b\|_2^2 = \|a\|_2^2 + \|b\|_2^2 - 2 a \cdot b
$$

当 $\|a\| = \|b\| = 1$（归一化后）：
$$
\|a - b\|_2^2 = 2(1 - \cos(a, b))
$$

**用途**：Embedding 相似度比较、BERTScore 计算、Head 子空间相似性分析（CKA）。

---

## 8. 广播（Broadcasting）

GQA 中 $K \in \mathbb{R}^{B \times T \times H_{\text{KV}} \times d}$ 广播到 $Q \in \mathbb{R}^{B \times T \times H \times d}$：

$$
K_{\text{expanded}}[:, :, h, :] = K[:, :, h // g, :]
$$

其中 $g = H / H_{\text{KV}}$（组大小）。广播**不实际复制数据**，只是逻辑上重复。

---

## 9. 逐元素操作 (Hadamard Product)

$$
(A \odot B)_{ij} = A_{ij} \cdot B_{ij}
$$

**在 Transformer 中**：
- SwiGLU：$\text{SiLU}(\text{gate}) \odot \text{up}$
- 门控注意力（GLA）：$\alpha_t \odot S_{t-1}$
- RoPE：复数旋转等价于实数的逐元素操作

**常见激活函数**：

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)
$$

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

---

## 面试一句话

> "Transformer 的计算核心是矩阵乘法（FLOPs = $2mkn$）和逐元素操作（激活/归一化）。SVD 解释了 LoRA 的有效性（Eckart-Young），核范数连接了隐式偏置与低秩，梯度裁剪防止训练爆炸。"
