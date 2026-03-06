# LoRA 与参数高效微调 (PEFT) 数学详解

> **核心定位**：从低秩分解的数学本质出发，严格推导 LoRA 的前向传播、初始化策略、缩放因子的设计动机，深入剖析 QLoRA / DoRA / AdaLoRA / LoRA+ 等变体的数学改进，并从内在维度（Intrinsic Dimensionality）理论解释低秩有效性的根源。

---

## 1. LoRA 核心公式推导

### 1.1 基本形式

对于原始线性层 $Y = XW$，$W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$。LoRA 在冻结 $W$ 的前提下，添加一个低秩增量 $\Delta W$：

$$
Y = X(W + \Delta W) = XW + X \underbrace{BA}_{\Delta W}
$$

其中 $B \in \mathbb{R}^{d_{\text{in}} \times r}$，$A \in \mathbb{R}^{r \times d_{\text{out}}}$，$r \ll \min(d_{\text{in}}, d_{\text{out}})$。

**带缩放因子的完整公式**：

$$
\boxed{Y = XW + \frac{\alpha}{r} \cdot X B A}
$$

### 1.2 参数量分析

| 量 | 公式 | 7B 模型 ($d = 4096, r = 16$) |
|----|------|:---------------------------:|
| 原始参数 | $d_{\text{in}} \times d_{\text{out}}$ | $16{,}777{,}216$ (单层) |
| LoRA 参数 | $r \times (d_{\text{in}} + d_{\text{out}})$ | $131{,}072$ (单层) |
| 压缩比 | $\frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}} \times d_{\text{out}}}$ | $\approx 0.78\%$ |

### 1.3 初始化策略

$$
A \sim \mathcal{N}(0, \sigma^2), \quad B = \mathbf{0}
$$

**设计动机**：训练开始时 $\Delta W = BA = \mathbf{0} \cdot A = \mathbf{0}$。模型的初始行为与预训练模型**完全一致**，不会因为微调开始就突然改变输出。

### 1.4 缩放因子 $\alpha / r$ 的数学作用

设学习率为 $\eta$。LoRA 增量对输出的影响尺度为：

$$
\|XBA\| \propto r \quad (\text{因为 } A \text{ 有 } r \text{ 行，每行独立贡献})
$$

如果不加 $\alpha / r$ 缩放，改变 $r$ 就需要同步调整 $\eta$。加入 $\alpha / r$ 后：

$$
\left\|\frac{\alpha}{r} XBA\right\| \propto \alpha \quad (\text{与 } r \text{ 无关})
$$

这使得调节 $r$ 时**不需要同步调节学习率**，极大简化了超参搜索。

---

## 2. 推理时的零开销合并

LoRA 的杀手级特性：推理时可以将增量**合并回原权重**：

$$
W_{\text{merged}} = W + \frac{\alpha}{r} BA
$$

合并后的模型与原始模型结构**完全相同**，推理时**零额外开销**。

---

## 3. 低秩有效性的理论根源

### 3.1 内在维度（Intrinsic Dimensionality）

> **出处**：Aghajanyan et al., "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning", 2020

即使模型有数十亿参数，微调时**实际需要改变的参数方向**远少于参数总数。定义内在维度 $d_{\text{int}}$：

$$
d_{\text{int}} = \min \left\{ d : \text{在 } d \text{ 维随机子空间中微调，能达到全量微调 } 90\% \text{ 的性能} \right\}
$$

实验发现：对于 GPT-2 (175M) 到 GPT-3 (175B)，$d_{\text{int}}$ 通常在 $O(10^2)$ 到 $O(10^3)$ 量级——远小于参数空间的 $O(10^9)$–$O(10^{11})$。

### 3.2 SVD 视角

权重更新矩阵的 SVD 分解：
$$
\Delta W = U \Sigma V^\top
$$

实验观察到 $\Sigma$ 的奇异值快速衰减。前 $r$ 个奇异值就捕获了 $\Delta W$ 的绝大部分信息：

$$
\Delta W \approx U_r \Sigma_r V_r^\top
$$

LoRA 等价于**直接学习这个低秩分解**，而不需要先全量微调再做 SVD。

### 3.3 隐式偏置的联系

SGD / Adam 的隐式偏置天然倾向于低秩解（见 Gunasekar 2017, Arora 2019 的理论证明）。LoRA 不是在"强迫"模型低秩，而是在**顺应**优化器的本能。

---

## 4. LoRA 变体详解

### 4.1 QLoRA

> **出处**：Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models", 2023

| 组件 | 精度 | 说明 |
|------|------|------|
| 基座模型 $W$ | **NF4** (4-bit) | 使用正态浮点量化 |
| LoRA 适配器 $B, A$ | BF16 | 保持高精度更新 |
| 量化参数 | **Double Quantization** | 对 scale 本身再做一次量化 |

$$
Y = X \cdot \text{Dequant}(W_{\text{NF4}}) + \frac{\alpha}{r} X B A
$$

**效果**：7B 模型 QLoRA 微调仅需 **单张 24 GB GPU**（原始 BF16 全量微调需要 $> 100$ GB）。

### 4.2 DoRA (Weight-Decomposed Low-Rank Adaptation)

> **出处**：Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", 2024

将权重分解为**方向**和**大小**两个独立部分：

$$
W' = m \cdot \frac{W + BA}{\|W + BA\|_c}
$$

其中 $m \in \mathbb{R}^{d_{\text{out}}}$ 是可学习的列级缩放向量，$\|\cdot\|_c$ 表示按列求范数。

**动机**：全量微调同时改变权重的方向和大小；标准 LoRA 将两者耦合在一起。DoRA 解耦方向（由 $BA$ 控制）和大小（由 $m$ 控制），更接近全量微调的行为。

### 4.3 AdaLoRA (Adaptive Budget Allocation)

> **出处**：Zhang et al., "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning", 2023

**核心思想**：不同层、不同矩阵需要的秩 $r$ 不同。AdaLoRA 使用 SVD 参数化 $\Delta W = P \Lambda Q$（$P, Q$ 正交，$\Lambda$ 对角），训练中**动态剪枝** $\Lambda$ 中不重要的奇异值。

重要性评分：
$$
\text{Importance}(i) = s_i \cdot \left\|\nabla_{s_i} \mathcal{L}\right\|
$$

其中 $s_i = \Lambda_{ii}$（第 $i$ 个奇异值）。低重要性的奇异值被置零，等效于降低该层的秩。

### 4.4 LoRA+

> **出处**：Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models", 2024

发现 $A$ 和 $B$ 应该使用**不同的学习率**：

$$
\eta_B = \eta, \quad \eta_A = \lambda \cdot \eta \quad (\lambda \gg 1, \text{通常 } \lambda = 16)
$$

**理论基础**：$B$ 的梯度尺度与 $A$ 不同（因为 $B$ 靠近输入端，$A$ 靠近输出端），用相同学习率会导致更新不平衡。

---

## 5. 应用位置选择

| 策略 | 应用位置 | LoRA 参数量 | 效果 |
|------|---------|:-----------:|------|
| Q + V | $W_Q, W_V$ | 最少 | 原始 LoRA 推荐，性价比最高 |
| Q + K + V + O | 全部 Attention 投影 | 中等 | 多数场景最优 |
| 全量 LoRA | Attention + FFN 所有线性层 | 最多 | 效果最好但参数量增加 $3$–$4\times$ |

---

## 6. 多 LoRA 服务 (Multi-LoRA Serving)

在推理服务中，同一基座模型可挂载多个 LoRA adapter：

$$
Y_i = XW + \frac{\alpha_i}{r_i} X B_i A_i \quad (\text{第 } i \text{ 个 adapter})
$$

| 方面 | 细节 |
|------|------|
| 基座权重 | 所有 adapter **共享**，显存只存一份 |
| Adapter 显存 | 每个 $\sim$ 几十 MB（$r = 16$, 应用于全部线性层约 50 MB） |
| 切换方式 | 按请求路由到对应 adapter |
| 与 KV Cache 的交互 | 不同 adapter 的 KV 不可共享（投影不同） |

---

## 面试一句话

> "LoRA 利用微调增量的低秩特性（内在维度 $\ll$ 参数维度），用 $< 1\%$ 的参数逼近全量微调效果。$B$ 零初始化保证起点不变，$\alpha/r$ 解耦秩和学习率，推理时合并回原权重零开销。QLoRA 再压基座到 4-bit，让 7B 模型在 24 GB GPU 上微调。"
