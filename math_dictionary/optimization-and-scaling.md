# 优化器、Scaling Law 与训练数学详解

> **核心定位**：从交叉熵损失出发，严格推导 Adam / AdamW 的更新规则与设计动机，深入 Chinchilla Scaling Law 的幂律模型与最优配比，涵盖学习率调度、梯度累积、训练 FLOPs 估算、MFU，以及推理 Scaling 的新范式。

---

## 1. 交叉熵损失（Next-Token Prediction）

$$
\boxed{\mathcal{L} = -\frac{1}{N}\sum_{t=1}^{N} \log p_\theta(x_t \mid x_{<t})}
$$

- $N$：总 token 数。
- $p_\theta(x_t \mid x_{<t})$：模型在位置 $t$ 对正确 token $x_t$ 的预测概率。
- **训练目标**：最大化正确 token 的概率 $\Leftrightarrow$ 最小化负对数似然。

---

## 2. Adam 优化器严格推导

### 2.1 一阶矩（动量）

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$m_t$ 是梯度 $g_t$ 的**指数加权移动平均**（EMA），用于平滑梯度噪声，提供动量。

### 2.2 二阶矩（自适应学习率）

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$v_t$ 是梯度平方的 EMA，估计每个参数维度的梯度方差。**逐元素**除以 $\sqrt{v_t}$ 实现自适应学习率。

### 2.3 偏差校正

由于 $m_0 = 0, v_0 = 0$，初始步骤的估计偏向零。校正公式：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

### 2.4 参数更新

$$
\boxed{\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}
$$

**常见超参**（LLM 训练）：$\beta_1 = 0.9$，$\beta_2 = 0.95$，$\epsilon = 10^{-8}$，$\eta \sim 10^{-4}$。

### 2.5 Adam 的致命盲区

Adam 是**逐元素（Element-wise）**的优化器。一个 $1000 \times 1000$ 的权重矩阵，在 Adam 眼里是 $10^6$ 个独立标量。它完全忽视了这些元素共同构成一个"几何空间变换矩阵"的事实，这加剧了权重向低秩塌陷的趋势。

---

## 3. AdamW（解耦权重衰减）

$$
\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
$$

**与 L2 正则化的区别**：
- **L2 正则**：将 $\lambda \theta$ 加到梯度 $g$ 中，然后经过 Adam 的自适应缩放。
- **AdamW**：将 $\lambda \theta$ 直接加到参数更新中，**不经过**自适应缩放。

后者在实验中表现更好，因为权重衰减不应被 $\sqrt{v_t}$ 调整。

---

## 4. 学习率调度

### 4.1 Warmup + Cosine Decay

$$
\eta(t) = \begin{cases}
\eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t \le T_{\text{warmup}} \\[6pt]
\eta_{\min} + \frac{\eta_{\max} - \eta_{\min}}{2}\left(1 + \cos\!\left(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\right)\right) & t > T_{\text{warmup}}
\end{cases}
$$

| 超参 | 典型值 | 说明 |
|------|--------|------|
| $T_{\text{warmup}}$ | 总步数的 $1\%$–$5\%$ | 防止初始大梯度导致不稳定 |
| $\eta_{\min}$ | $\eta_{\max} \times 0.1$ | 最终学习率 |

### 4.2 WSD (Warmup-Stable-Decay)

MiniCPM 等提出的三阶段调度：Warmup → 恒定 $\eta_{\max}$ → 快速衰减。更适合固定算力预算的训练。

---

## 5. 梯度累积

当显存不足以放下目标 Batch Size 时：

$$
\text{Effective Batch Size} = B_{\text{micro}} \times G_{\text{accum}} \times D_{\text{parallel}}
$$

每 $G_{\text{accum}}$ 步做一次参数更新。数学上等价于大 Batch（忽略 BN 等），但 wall-clock 时间不变。

---

## 6. Scaling Law

### 6.1 Chinchilla 幂律模型

$$
\boxed{L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_{\text{irr}}}
$$

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $N$ | 模型参数量 | — |
| $D$ | 训练 Token 数 | — |
| $\alpha$ | 参数量指数 | $\approx 0.34$ |
| $\beta$ | 数据量指数 | $\approx 0.28$ |
| $L_{\text{irr}}$ | 不可约损失（数据本身的熵下界） | $\approx 1.69$ |

### 6.2 Chinchilla 最优配比

给定计算预算 $C \approx 6ND$（总训练 FLOPs），最小化 $L(N, D)$ 的最优条件为：

$$
\frac{\partial L}{\partial N} \cdot \frac{\partial C}{\partial D} = \frac{\partial L}{\partial D} \cdot \frac{\partial C}{\partial N}
$$

求解得到最优比例：

$$
\boxed{D^* \approx 20 N^*}
$$

即**训练 Token 数应约为参数量的 20 倍**。

| 模型 | 参数量 | Chinchilla 最优 Token 数 |
|------|--------|:-------------------------:|
| 7B | $7 \times 10^9$ | $\sim 140$B |
| 70B | $7 \times 10^{10}$ | $\sim 1.4$T |
| 405B | $4.05 \times 10^{11}$ | $\sim 8$T |

**注**：实际训练（如 Llama 3）常显著超过 Chinchilla 最优（"过训练"），因为推理成本与 $N$ 成正比——小模型多训点更划算。

---

## 7. 训练 FLOPs 估算

### 7.1 单 Token 的 FLOPs

| 阶段 | FLOPs per Token |
|------|:---------------:|
| 前向传播 | $\approx 2N$ |
| 反向传播 | $\approx 4N$（约前向的 2 倍） |
| **总计** | $\approx 6N$ |

### 7.2 总训练 FLOPs

$$
\boxed{C = 6 N D}
$$

**代入示例**：

| 模型 | $N$ | $D$ | $C$ | A100 GPU-Days (MFU=50%) |
|------|-----|-----|-----|:----------------------:|
| 7B | $7 \times 10^9$ | $1 \times 10^{12}$ | $4.2 \times 10^{22}$ | $\sim 3{,}120$ |
| 70B | $7 \times 10^{10}$ | $1.4 \times 10^{12}$ | $5.9 \times 10^{23}$ | $\sim 43{,}500$ |

GPU-Days 计算：
$$
\text{GPU-Days} = \frac{C}{\text{Peak FLOPS} \times \text{MFU} \times 86400}
$$

---

## 8. MFU (Model FLOPs Utilization)

$$
\boxed{\text{MFU} = \frac{6ND / T_{\text{wall}}}{\text{Num GPUs} \times \text{Peak FLOPS per GPU}}}
$$

| MFU 范围 | 评价 |
|:--------:|------|
| $> 50\%$ | 优秀 |
| $30\%$–$50\%$ | 良好 |
| $< 30\%$ | 需要优化（通信瓶颈、加载瓶颈等） |

---

## 9. 推理 Scaling (Test-Time Compute Scaling)

### 9.1 核心思路

在推理阶段通过更多计算提升质量：

| 方法 | 推理 FLOPs 倍增 | 说明 |
|------|:--------------:|------|
| **Best-of-N** | $N\times$ | 生成 $N$ 个回答，用 Verifier 选最好的 |
| **Chain-of-Thought** | $L_{\text{CoT}} / L_{\text{direct}}$ | 更长的推理链 → 更多 Token |
| **Tree Search / MCTS** | 指数级 | 搜索更大的候选空间 |

### 9.2 Scaling 行为

推理 FLOPs 与质量也呈现类 Scaling Law 的 log-linear 关系：

$$
\text{Quality} \propto a \cdot \log(\text{Inference FLOPs}) + b
$$

---

## 面试一句话

> "Chinchilla 告诉我们参数和数据要同步 scale（$D^* \approx 20N^*$）；$C = 6ND$ 一步算出训练总预算；AdamW 解耦权重衰减比 L2 更优；推理 Scaling 则打开了'测试时算力换质量'的新范式。"