# RLHF 与对齐 (Alignment) 数学详解

> **核心定位**：严格推导 RLHF 三阶段（SFT → RM → PPO）的数学目标函数，证明 DPO 如何将 RL 目标转化为闭式解的监督学习问题，深入剖析 KL 约束的对齐角色、KTO 的前景理论基础，以及 Reward Hacking 的形成机制。

---

## 1. RLHF 三阶段完整流程

### 阶段 1：SFT (Supervised Fine-Tuning)

$$
\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{human}}} \left[\sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})\right]
$$

标准的 Next-Token Prediction，在人类标注的高质量数据上微调。

### 阶段 2：RM (Reward Modeling)

### 阶段 3：PPO (Policy Optimization)

---

## 2. 奖励模型 (Reward Model) 推导

### 2.1 Bradley-Terry 偏好模型

给定 Prompt $x$，人类标注两个回答 $y_w$（Winner）和 $y_l$（Loser）的偏好。假设偏好概率遵循 Bradley-Terry 模型：

$$
P(y_w \succ y_l \mid x) = \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

其中 $\sigma(\cdot)$ 是 Sigmoid 函数，$r_\phi$ 是带参数 $\phi$ 的奖励模型。

### 2.2 训练损失

$$
\boxed{\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]}
$$

**直觉**：最大化"好回答的奖励 $>$ 坏回答的奖励"这件事的概率。

### 2.3 奖励模型的关键性质

- 奖励的**绝对值无意义**，只有**差值**有意义（因为 Bradley-Terry 模型只依赖差值）。
- 通常在 SFT 模型基础上，将最后一层的 LM Head 替换为标量输出 Head。

---

## 3. PPO 目标函数推导

### 3.1 完整目标

$$
\boxed{J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \; y \sim \pi_\theta(\cdot|x)} \left[r_\phi(x, y)\right] - \beta \cdot \text{KL}\!\left(\pi_\theta \,\parallel\, \pi_{\text{ref}}\right)}
$$

| 项 | 含义 |
|----|------|
| $\mathbb{E}[r_\phi(x, y)]$ | 最大化奖励（让模型生成高分回答） |
| $\beta \cdot \text{KL}(\pi_\theta \parallel \pi_{\text{ref}})$ | KL 惩罚项（防止策略偏离参考模型太远） |
| $\pi_{\text{ref}}$ | 参考策略（通常是 SFT 模型） |

### 3.2 KL 散度展开

$$
\text{KL}(\pi_\theta \parallel \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta} \left[\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right]
$$

### 3.3 $\beta$ 的权衡

| $\beta$ | 效果 |
|---------|------|
| 太小 | **Reward Hacking**：策略找到奖励模型的漏洞，生成高分但低质量的回答 |
| 太大 | 学不到新东西，策略退化为 $\pi_{\text{ref}}$（过度保守） |
| 典型值 | $0.01$–$0.2$ |

### 3.4 最优策略的闭式解

给定目标函数 $J(\theta)$，可以证明最优策略为：

$$
\boxed{\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \cdot \exp\!\left(\frac{r_\phi(x, y)}{\beta}\right)}
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r_\phi(x,y)/\beta)$ 是归一化常数。

**证明**：对 $J(\theta)$ 关于 $\pi_\theta$ 取变分导数，令其为零，利用 KL 散度的凸性得到上述闭式解。$\blacksquare$

---

## 4. DPO (Direct Preference Optimization)

> **出处**：Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023

### 4.1 核心洞察

从最优策略的闭式解反解出奖励函数：

$$
r_\phi(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

将其代入 Bradley-Terry 偏好概率（归一化常数 $Z(x)$ 在差值中消掉）：

$$
P(y_w \succ y_l | x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

### 4.2 DPO 损失函数

用策略 $\pi_\theta$ 替代 $\pi^*$，直接优化：

$$
\boxed{\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\!\left(\beta \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\right]}
$$

### 4.3 DPO 的革命性意义

| 方面 | PPO | DPO |
|------|-----|-----|
| 需要训练 RM | ✅ | ❌ |
| 需要 RL（采样 + 策略梯度） | ✅ | ❌（纯监督学习） |
| 训练稳定性 | 较差（高方差梯度） | 好（标准交叉熵） |
| 隐式 RM | — | $r(x,y) = \beta \log(\pi_\theta / \pi_{\text{ref}}) + \text{const}$ |

---

## 5. KTO (Kahneman-Tversky Optimization)

> **出处**：Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization", 2024

### 5.1 动机

DPO 需要**成对偏好数据**（$y_w, y_l$ 对同一 $x$），但实际中更容易获得**单独的 good/bad 标签**。

### 5.2 基于前景理论的损失

$$
\mathcal{L}_{\text{KTO}} = \mathbb{E}_{(x,y) \sim \text{Good}} \left[-\log \sigma(r_\theta)\right] + \lambda \cdot \mathbb{E}_{(x,y) \sim \text{Bad}} \left[-\log \sigma(-r_\theta)\right]
$$

其中 $r_\theta = \beta \log(\pi_\theta(y|x) / \pi_{\text{ref}}(y|x))$。

$\lambda > 1$（典型值 $\lambda \approx 1.5$–$2.0$）体现了**损失厌恶（Loss Aversion）**：模型对坏回答的惩罚力度大于对好回答的奖励力度。

---

## 6. Reward Hacking 的数学分析

### 6.1 Goodhart 定律的形式化

$$
\text{True Reward} = r_{\text{true}}(x, y)
$$
$$
\text{Proxy Reward} = r_\phi(x, y) \approx r_{\text{true}}(x, y) + \epsilon(x, y)
$$

当 $\pi_\theta$ 过度优化 $r_\phi$ 时：

$$
\mathbb{E}_{\pi_\theta}[r_\phi] \uparrow \quad \text{but} \quad \mathbb{E}_{\pi_\theta}[r_{\text{true}}] \downarrow
$$

策略学会了利用 $\epsilon$ 中的系统性误差（代理模型的漏洞），而非真正提升回答质量。

### 6.2 KL 约束的保护作用

KL 惩罚限制了策略的探索范围，使其不能偏离 $\pi_{\text{ref}}$ 太远。在 $\pi_{\text{ref}}$ 附近，$r_\phi \approx r_{\text{true}}$（代理模型的训练数据主要来自 $\pi_{\text{ref}}$ 的分布），因此 KL 约束间接保证了奖励的可靠性。

---

## 7. 方法对比总表

| 方法 | 数据需求 | 需要 RM | 需要 RL | 训练稳定性 | 核心数学 |
|------|---------|:------:|:------:|:---------:|---------|
| **RLHF (PPO)** | 偏好对 | ✅ | ✅ | 较不稳定 | 策略梯度 + KL 正则 |
| **DPO** | 偏好对 | ❌ | ❌ | 稳定 | 闭式解消去 RM |
| **KTO** | Good/Bad 标签 | ❌ | ❌ | 稳定 | 前景理论 + 损失厌恶 |
| **RLAIF** | AI 生成偏好 | ✅ (AI标注) | ✅ | 中等 | 与 RLHF 相同 |
| **SimPO** | 偏好对 | ❌ | ❌ | 稳定 | 长度归一化的隐式奖励 |

---

## 面试一句话

> "RLHF 的核心权衡是 reward 最大化 vs KL 约束：太贪心会 reward hack，太保守学不到新能力。DPO 的数学贡献是证明了最优策略的闭式解可以消掉 RM，将 RL 问题转化为标准监督损失，本质上 $\pi_\theta$ 自身就是一个隐式的 Reward Model。"
