# 投机解码（Speculative Decoding）数学详解

> **核心定位**：从接受-拒绝采样的概率论基础出发，严格证明投机解码的无损性质，推导期望加速比的闭式表达，并深入剖析 Draft Model 选型、树形投机（Tree Speculation）与最优 $K$ 选择的数学框架。

---

## 1. 核心思想

标准自回归解码中，每生成一个 token 都需要一次完整的 Target Model（大模型）前向传播。Decode 阶段是 Memory-bound 的，GPU 算力严重闲置。

**投机解码的核心洞察**：用一个小而快的 Draft Model 连续猜测 $K$ 个 token，然后用 Target Model **一次前向**并行验证。验证通过的直接输出，不通过的重新采样。

---

## 2. 接受-拒绝采样的数学推导

### 2.1 单 Token 接受概率

设 Draft Model 在当前上下文下的分布为 $q(x)$，Target Model 的分布为 $p(x)$。对 Draft Model 提出的 token $x$：

$$
\boxed{P_{\text{accept}}(x) = \min\!\left(1, \frac{p(x)}{q(x)}\right)}
$$

### 2.2 拒绝后的修正分布

若 token $x$ 被拒绝，从以下修正分布中重新采样：

$$
p_{\text{resample}}(x) = \frac{\max(0, \; p(x) - q(x))}{Z}
$$

其中归一化常数：
$$
Z = \sum_x \max(0, \; p(x) - q(x)) = 1 - \sum_x \min(p(x), q(x))
$$

### 2.3 无损性证明

**定理**：最终输出的 token 分布严格等于 Target Model 的分布 $p(x)$。

**证明**：对于任意 token $x$，其最终被选中的概率为：

$$
\Pr[X = x] = \underbrace{q(x) \cdot \min\!\left(1, \frac{p(x)}{q(x)}\right)}_{\text{被 Draft 提出且被接受}} + \underbrace{(1 - \alpha) \cdot \frac{\max(0, p(x) - q(x))}{Z}}_{\text{被拒绝后从修正分布采样}}
$$

其中 $\alpha = \sum_x q(x) \cdot \min(1, p(x)/q(x)) = \sum_x \min(p(x), q(x))$ 是整体接受概率。

**Case 1**：$p(x) \ge q(x)$
$$
= q(x) \cdot 1 + (1 - \alpha) \cdot \frac{p(x) - q(x)}{1 - \alpha} = q(x) + p(x) - q(x) = p(x) \quad \checkmark
$$

**Case 2**：$p(x) < q(x)$
$$
= q(x) \cdot \frac{p(x)}{q(x)} + (1 - \alpha) \cdot 0 = p(x) \quad \checkmark
$$

因此 $\Pr[X = x] = p(x)$ 对所有 $x$ 成立。$\blacksquare$

---

## 3. 期望加速比分析

### 3.1 期望接受长度

假设每步的接受概率为 $\alpha$（对 $x \sim q$ 取期望）。$K$ 个连续候选 token 中，期望被接受的数量为：

$$
\mathbb{E}[\text{accepted}] = \sum_{i=0}^{K-1} \alpha^{i+1} \cdot \prod_{j=0}^{i-1} \alpha = \sum_{i=1}^{K} \alpha^i = \frac{\alpha(1 - \alpha^K)}{1 - \alpha}
$$

当 $K \to \infty$ 时：
$$
\mathbb{E}[\text{accepted}] \to \frac{\alpha}{1 - \alpha}
$$

加上拒绝后的重采样 token（总是 1 个），每轮实际输出 $\mathbb{E}[\text{accepted}] + 1$ 个 token。

### 3.2 加速比公式

无投机时，每 token 需要 1 次 Target Model 前向（延迟 $T_p$）。有投机时：

$$
\text{Latency per round} = K \cdot T_q + T_p
$$
$$
\text{Tokens per round} = \mathbb{E}[\text{accepted}] + 1
$$

$$
\boxed{\text{Speedup} = \frac{T_p}{\displaystyle \frac{K \cdot T_q + T_p}{\mathbb{E}[\text{accepted}] + 1}} = \frac{T_p \cdot \left(\frac{\alpha(1-\alpha^K)}{1-\alpha} + 1\right)}{K \cdot T_q + T_p}}
$$

**典型数值**：$\alpha = 0.8$，$K = 5$，$T_q / T_p = 0.1$：
$$
\text{Tokens per round} = \frac{0.8 \times (1 - 0.8^5)}{0.2} + 1 = 0.8 \times 3.36 / 0.2 + 1 \approx 3.69
$$
$$
\text{Speedup} \approx \frac{3.69}{0.5 + 1} = 2.46\times
$$

---

## 4. 最优 $K$ 的选择

$K$ 的选择是**边际效益递减**的：

- $K$ 太小：验证频率高，Target Model 前向开销无法充分摊销。
- $K$ 太大：后续 token 的接受概率 $\alpha^K$ 衰减到接近 $0$，Draft Model 的计算浪费。

最优 $K^*$ 满足**边际接受概率 = 边际成本比**：

$$
\frac{\partial}{\partial K}\left[\frac{\text{Tokens per round}}{\text{Latency per round}}\right] = 0
$$

近似解：
$$
K^* \approx \frac{-\ln(T_q / T_p)}{\ln(1/\alpha)}
$$

实践中还可以**动态调整**：跟踪最近 $N$ 轮的实际接受率，自适应地增减 $K$。

---

## 5. Draft Model 选型

| 类型 | 代表方案 | 优势 | 劣势 |
|------|---------|------|------|
| **独立小模型** | 1B draft → 70B target | 灵活部署 | 分布差异大，$\alpha$ 低 |
| **Self-draft (Medusa)** | 在 Target 上加 $K$ 个轻量 Head | 共享 backbone，$\alpha$ 高 | 需要额外训练 Head |
| **Self-draft (EAGLE)** | 利用 Target 的中间层特征预测 | 无需额外训练 | 实现复杂 |
| **N-gram 匹配** | 从 Prompt 中匹配重复模式 | 零额外模型 | 仅适用于重复性内容 |

---

## 6. 树形投机 (Tree Speculation)

传统投机是一条**链**（单路径），每步只有一个候选。树形投机每步生成多个候选分支：

$$
\text{Tree Acceptance} = 1 - \prod_{\text{branch}} (1 - \alpha_{\text{branch}})
$$

用 **Tree Attention Mask**（特殊的因果 Mask）使 Target Model 一次前向验证整棵树。

**优势**：单次验证期望接受的 token 数更多（但 Draft 阶段计算量也更大）。

**代表方案**：SpecInfer (Miao et al., 2023)——维护一个候选 Token Tree，每轮用 Target Model 做树形验证。

---

## 7. 与 KV Cache 的交互

投机解码对 KV Cache 管理有特殊要求：

1. **Draft Model KV Cache**：独立小模型需要自己的 KV Cache（额外显存开销 = $\text{bytes\_per\_token}_{\text{draft}} \times T$）。
2. **Rollback 机制**：验证失败的 token 对应的 KV 必须从 Target Model 的 Cache 中**回滚删除**。
3. **PagedAttention 兼容**：vLLM 的 Block Fork / Rollback 操作天然支持投机解码的分叉与回退。

---

## 8. 面试实战追问

**Q1：投机解码为什么是"无损"的？**
> 因为接受-拒绝采样保证了最终输出分布严格等于 Target Model 分布。被拒绝时从修正分布 $\max(0, p - q) / Z$ 重新采样，数学上可以证明两种路径（接受 + 拒绝后重采样）的总概率恰好等于 $p(x)$。

**Q2：什么时候投机解码不划算？**
> 当 Draft Model 与 Target Model 的分布差异大（$\alpha$ 低，如 $< 0.5$）、或 Draft Model 本身不够快（$T_q / T_p > 0.3$）时，加速比可能低于 $1.5\times$，此时引入的系统复杂性（双模型管理、KV 回滚）可能不值得。