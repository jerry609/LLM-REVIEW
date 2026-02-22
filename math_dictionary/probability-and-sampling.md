# 概率与采样数学详解

> **核心定位**：从概率论基础出发，严格推导 LLM 推理中所有采样策略的数学定义——温度缩放、Top-k / Top-p / Min-p、Beam Search 的评分函数，以及熵、KL 散度、困惑度之间的精确数学关系。每个概念都直接关联到实际的推理参数调优。

---

## 1. Softmax 与温度缩放

### 1.1 带温度的 Softmax

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

温度 $\tau$ 控制分布的"尖锐程度"：

| $\tau$ | 效果 | 极限 |
|:------:|------|------|
| $\tau \to 0^+$ | 分布退化为 $\arg\max$（贪婪解码） | $p_{\arg\max} \to 1$，其余 $\to 0$ |
| $\tau = 1$ | 原始 logit 分布 | 标准 Softmax |
| $\tau \to \infty$ | 分布趋向均匀 | $p_i \to 1/V$（$V$ 为词表大小） |

### 1.2 温度的数学本质

令 $z_i' = z_i / \tau$，则温度缩放等价于将 logit 的**动态范围**压缩 $\tau$ 倍：

$$
\max(z') - \min(z') = \frac{\max(z) - \min(z)}{\tau}
$$

---

## 2. 采样策略

### 2.1 Top-k 采样

$$
S_k = \{i : p_i \text{ 属于概率最高的 } k \text{ 个 token}\}
$$
$$
p_i^{(k)} = \begin{cases} p_i / \sum_{j \in S_k} p_j & i \in S_k \\ 0 & i \notin S_k \end{cases}
$$

### 2.2 Top-p (Nucleus) 采样

$$
S_p = \arg\min_{|S'|} \left\{ S' : \sum_{i \in S'} p_i \ge p \right\}
$$

取概率从大到小排序后，累积概率恰好达到阈值 $p$ 的**最小集合**，然后在 $S_p$ 中重归一化后采样。

**与 Top-k 的区别**：Top-p 的候选集大小是**动态的**——在模型很确定的位置可能只有 $1$–$2$ 个候选，不确定时可能有数百个。

### 2.3 Min-p 采样

$$
S_{\text{min-p}} = \{i : p_i \ge \text{min\_p} \times p_{\max}\}
$$

其中 $p_{\max} = \max_i p_i$。

**直觉**：只保留概率不低于最大概率的 $\text{min\_p}$ 比例的 token。

**优势**：在高确定性（$p_{\max}$ 大）和低确定性（$p_{\max}$ 小）时自适应调整候选集大小。

### 2.4 联合使用

$$
S_{\text{final}} = S_k \cap S_p \cap S_{\text{min-p}}
$$

先 Top-k 过滤 → 再 Top-p 截断 → 再 Min-p 进一步精筛 → 重归一化后采样。

---

## 3. 贪婪解码 vs Beam Search

### 3.1 贪婪解码

$$
y_t = \arg\max_i p_\theta(i \mid x, y_{<t})
$$

确定性，不需要采样。但可能陷入局部最优（重复 loop）。

### 3.2 Beam Search

维护 $B_{\text{beam}}$ 个候选序列，每步扩展所有可能的下一个 token 后保留得分最高的 $B_{\text{beam}}$ 个。

序列得分：
$$
\text{Score}(y) = \sum_{t=1}^{|y|} \log p_\theta(y_t \mid y_{<t})
$$

**长度惩罚**：防止偏好短序列：
$$
\text{Score}_{\text{norm}}(y) = \frac{\text{Score}(y)}{|y|^\alpha}
$$

$\alpha = 0$：不惩罚；$\alpha = 1$：按长度归一化；$\alpha = 0.6$–$0.8$ 为常用值。

---

## 4. 熵与不确定性

### 4.1 Shannon 熵

$$
H(p) = -\sum_i p_i \log p_i
$$

| 分布 | 熵 |
|------|:--:|
| 确定性（只有一个 $p_i = 1$） | $0$ |
| 均匀分布（$p_i = 1/V$） | $\log V$（最大熵） |
| 自然语言的典型位置 | $3$–$8$ bits |

### 4.2 动态温度调节

可以用熵作为信号动态调节 $\tau$：当 $H$ 过低（过于确定）时适当增大 $\tau$ 以增加多样性。

---

## 5. KL 散度

### 5.1 定义

$$
\text{KL}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i} \ge 0
$$

当且仅当 $p = q$ 时取等。

### 5.2 非对称性

$$
\text{KL}(p \| q) \ne \text{KL}(q \| p)
$$

- $\text{KL}(p \| q)$（前向 KL）：惩罚 $q$ 在 $p$ 有概率的地方给出低概率 → 倾向 $q$ 覆盖 $p$ 的所有模式（**模式覆盖**）。
- $\text{KL}(q \| p)$（反向 KL）：惩罚 $q$ 在 $p$ 无概率的地方给出高概率 → 倾向 $q$ 聚焦在 $p$ 的高概率区域（**模式追踪**）。

### 5.3 在 RLHF 中的角色

$$
J(\theta) = \mathbb{E}_{\pi_\theta}[r(x,y)] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

KL 项防止策略过度偏离参考模型，避免 Reward Hacking。

---

## 6. 交叉熵与困惑度

### 6.1 交叉熵

$$
\text{CE} = -\frac{1}{N}\sum_{t=1}^{N} \log p_\theta(x_t \mid x_{<t})
$$

### 6.2 困惑度

$$
\boxed{\text{PPL} = \exp(\text{CE}) = \exp\!\left(-\frac{1}{N}\sum_{t=1}^{N} \log p_\theta(x_t \mid x_{<t})\right)}
$$

**直觉**：PPL = $k$ 意味着模型在每个位置的平均不确定性等价于在 $k$ 个等概率选项中选择。

- PPL = $1$：完美预测。
- PPL = $10$：平均每步相当于从 $10$ 个选项中选。
- PPL = $V$（词表大小）：等于随机猜。

### 6.3 与 KL 散度 / 熵的关系

$$
\text{CE}(p_{\text{data}}, p_\theta) = H(p_{\text{data}}) + \text{KL}(p_{\text{data}} \| p_\theta)
$$

由于数据分布的熵 $H(p_{\text{data}})$ 是常数，**最小化交叉熵 $\Leftrightarrow$ 最小化 KL 散度**。

---

## 7. 复读惩罚

### 7.1 频率惩罚 (Frequency Penalty)

$$
z_i' = z_i - \alpha_{\text{freq}} \cdot \text{count}(i)
$$

出现次数越多，logit 降得越多。

### 7.2 存在惩罚 (Presence Penalty)

$$
z_i' = z_i - \beta_{\text{pres}} \cdot \mathbb{1}[\text{count}(i) > 0]
$$

只要出现过就统一扣分（不管出现几次）。

| 参数 | 控制行为 |
|------|---------|
| 频率惩罚 | 减少高频重复 |
| 存在惩罚 | 鼓励引入新 token（多样性） |

---

## 8. 投机解码中的接受概率

Draft Model 提出 $x \sim q$，Target Model 验证：

$$
P_{\text{accept}}(x) = \min\!\left(1, \frac{p(x)}{q(x)}\right)
$$

被拒绝时从修正分布重采样：

$$
p_{\text{resample}}(x) \propto \max(0, \; p(x) - q(x))
$$

**保证最终分布 = Target Model 分布**（数学无损）。

期望加速比（理想情况）：
$$
\text{Speedup} \propto \frac{1}{1 - \alpha}, \quad \alpha = \sum_x \min(p(x), q(x))
$$

---

## 面试一句话

> "采样参数控制的是输出分布的形状（$\tau$ 控制尖锐度，Top-p 控制候选集大小），不改变模型能力本身。PPL = $\exp(\text{CE})$ 直接关联训练损失，KL 散度保证对齐不走偏。实际调参时温度 + Top-p 联合使用效果最好。"