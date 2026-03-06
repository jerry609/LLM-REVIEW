# MoE (Mixture of Experts) 路由与负载数学详解

> **核心定位**：系统性推导 MoE 架构的路由机制、负载均衡损失、容量限制、FLOPs/显存分析，以及 Expert Parallelism (EP) 的 All-to-All 通信代价。覆盖 Mixtral、DeepSeek-MoE 等实际模型的设计选择。

---

## 1. MoE 架构概述

标准 Transformer 中每层的 FFN（前馈网络）替换为 $E$ 个并行的 Expert FFN，每个 token 只激活 $E_{\text{active}}$ 个 Expert。

$$
\text{Output} = \sum_{i \in \text{TopK}(p, E_{\text{active}})} p_i \cdot \text{Expert}_i(x)
$$

| 模型 | $E$ | $E_{\text{active}}$ | 总参数 | 活跃参数/token |
|------|:---:|:-------------------:|:------:|:-------------:|
| Mixtral 8×7B | $8$ | $2$ | $\sim 47$B | $\sim 13$B |
| DeepSeek-MoE (16B) | $64$ | $6$ | $\sim 16$B | $\sim 2.8$B |
| DBRX | $16$ | $4$ | $\sim 132$B | $\sim 36$B |

---

## 2. 门控函数 (Router) 数学推导

### 2.1 标准 Top-K 路由

对于 token 的隐藏状态 $x \in \mathbb{R}^{d_{\text{model}}}$：

$$
g = x W_{\text{gate}} \quad \text{(门控 logit)}
$$
$$
p = \text{softmax}(g) \in \mathbb{R}^E \quad \text{(路由概率)}
$$
$$
S = \text{TopK}(p, E_{\text{active}}) \quad \text{(选择前 } E_{\text{active}} \text{ 个 Expert)}
$$
$$
\tilde{p}_i = \frac{p_i}{\sum_{j \in S} p_j} \quad \text{(重归一化)}
$$
$$
\text{Output} = \sum_{i \in S} \tilde{p}_i \cdot \text{Expert}_i(x)
$$

其中 $W_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times E}$ 是路由器的权重矩阵（参数量很小，$\sim d_{\text{model}} \times E$）。

### 2.2 DeepSeek-MoE 的共享 Expert 设计

DeepSeek-MoE 引入 $E_{\text{shared}}$ 个**共享 Expert**（所有 token 都会通过）+ $E - E_{\text{shared}}$ 个路由 Expert：

$$
\text{Output} = \sum_{i=1}^{E_{\text{shared}}} \text{Expert}_i^{\text{shared}}(x) + \sum_{j \in S_{\text{routed}}} \tilde{p}_j \cdot \text{Expert}_j^{\text{routed}}(x)
$$

**动机**：将通用知识放在共享 Expert 中，让路由 Expert 专注于差异化的专业知识。

---

## 3. 负载均衡损失 (Auxiliary Loss)

### 3.1 问题：Expert 坍缩

如果不加约束，路由器可能将所有 token 路由到少数几个 Expert（"Expert 坍缩"）。其他 Expert 永远得不到训练，等于浪费。

### 3.2 辅助损失公式

$$
\boxed{\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i}
$$

| 符号 | 定义 | 含义 |
|------|------|------|
| $f_i$ | $\frac{1}{N} \sum_{\text{tokens}} \mathbb{1}[\text{token 被路由到 Expert } i]$ | 实际选择频率 |
| $P_i$ | $\frac{1}{N} \sum_{\text{tokens}} p_i(\text{token})$ | 平均路由概率 |

### 3.3 最优解分析

当所有 Expert 被均匀选中时：$f_i = E_{\text{active}} / E$，$P_i = 1 / E$。

$$
\mathcal{L}_{\text{balance}}^{\min} = E \cdot E \cdot \frac{E_{\text{active}}}{E} \cdot \frac{1}{E} = E_{\text{active}}
$$

**总损失**：
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha \cdot \mathcal{L}_{\text{balance}}, \quad \alpha \sim 0.01
$$

### 3.4 为什么用 $f_i \cdot P_i$ 而不是直接用 $f_i$？

$f_i$ 涉及 $\text{argmax}$（TopK）操作，不可微。$P_i$ 是 Softmax 输出的均值，是可微的。通过将两者相乘，梯度可以通过 $P_i$ 反传到路由器权重。

---

## 4. Expert Capacity（容量限制）

### 4.1 动机

在分布式训练中，每个 Expert 的 Batch 大小必须固定（GPU 不支持动态形状的高效计算）。

### 4.2 容量公式

$$
\text{Capacity} = \text{CF} \times \frac{B \times T}{E}
$$

- $\text{CF}$（Capacity Factor）：通常 $1.0$–$1.5$。
- $B \times T / E$：均匀分配时每个 Expert 期望处理的 token 数。

### 4.3 溢出处理

超出容量的 token 被**丢弃（Dropped）**或路由到备选 Expert：

$$
\text{Drop Rate} = \frac{N_{\text{dropped}}}{N_{\text{total}}}
$$

| CF | Drop Rate | 质量 | 显存 |
|:--:|:---------:|:----:|:----:|
| $1.0$ | 高（$5\%$–$15\%$） | 较差 | 低 |
| $1.25$ | 中（$1\%$–$5\%$） | 较好 | 中 |
| $1.5$ | 低（$< 1\%$） | 好 | 高 |

---

## 5. FLOPs 分析

### 5.1 单层 MoE 的 FLOPs

$$
\text{FLOPs}_{\text{MoE\_layer}} = \underbrace{\text{FLOPs}_{\text{Attention}}}_{\text{不变（共享）}} + \underbrace{E_{\text{active}} \times \text{FLOPs}_{\text{single\_FFN}}}_{\text{只算活跃 Expert}}
$$

### 5.2 与 Dense 模型的对比

对于 Mixtral 8×7B（$E = 8, E_{\text{active}} = 2$）：
- 每个 Expert 的 FFN 参数 ≈ $5.6$B
- 每 token 活跃 FFN 参数 = $2 \times 5.6$B $= 11.2$B
- 加上共享的 Attention 参数 $\sim 1.2$B/层

**推理速度接近 13B Dense 模型，但质量接近更大的 Dense 模型**（因为总参数 $47$B 提供了更大的知识容量）。

---

## 6. 显存分析（推理）

### 6.1 权重显存

所有 Expert 的权重都需要加载（即使每 token 只用部分）：

$$
M_{\text{weights}} = \underbrace{E \times \text{FFN}_{\text{params}} \times s}_{\text{所有 Expert}} + \underbrace{\text{Shared}_{\text{params}} \times s}_{\text{Attention + Embedding}}
$$

### 6.2 KV Cache

KV Cache **不受 MoE 影响**（Attention 层是共享的，非 MoE）。

### 6.3 Expert Parallelism (EP) 下的显存

每张卡只存 $E / P_{\text{EP}}$ 个 Expert：

$$
M_{\text{weights\_per\_card}} = \frac{E}{P_{\text{EP}}} \times \text{FFN}_{\text{params}} \times s + \text{Shared}_{\text{params}} \times s
$$

---

## 7. Expert Parallelism (EP) 通信详解

### 7.1 All-to-All 通信流程

1. **Dispatch**：每张卡将被路由到其他卡上 Expert 的 token 发送出去。
2. **Compute**：每张卡上的 Expert 处理收到的 token。
3. **Combine**：每张卡将处理结果返回给原始卡。

### 7.2 通信量

每次 All-to-All 的最坏情况数据量：

$$
n_{\text{A2A}} = B \times T \times d_{\text{model}} \times s
$$

双向（Dispatch + Combine）：

$$
\text{Comm}_{\text{EP}} = 2 \times n_{\text{A2A}} = 2 B T d_{\text{model}} s
$$

### 7.3 负载不均衡对通信的影响

如果路由不均匀，某些卡收到的 token 远多于其他卡。All-to-All 的完成时间取决于**最慢的通信路径**：

$$
T_{\text{A2A}} = \max_{i \to j} \left(\alpha + \beta \times n_{\text{bytes}}(i \to j)\right)
$$

---

## 8. TP + EP 混合部署

在 Expert 本身很大时，需要在 Expert 内部再做 Tensor Parallel：

| 层级 | 并行策略 | 通信模式 |
|------|---------|---------|
| 节点内 | TP = $8$（切分 Expert 内部权重） | All-Reduce (NVLink) |
| 节点间 | EP = $N_{\text{nodes}}$（切分 Expert） | All-to-All (IB/RoCE) |
| Pipeline | PP（层间切分） | Point-to-Point |

---

## 面试一句话

> "MoE 用稀疏激活换算力效率：总参数大但每 token 只用 $E_{\text{active}} / E$ 的 FFN。关键挑战是负载均衡（辅助损失 $E \sum f_i P_i$）和 All-to-All 通信（$\propto BTd$）。Capacity Factor 控制精度-效率权衡，Drop Rate 高于 $5\%$ 则需要调参。"

---

## 对应源码与阅读顺序

- 先读 [../notes/distributed/moe-formula-to-code-walkthrough.md](../notes/distributed/moe-formula-to-code-walkthrough.md)，把 softmax、top-k、capacity、drop rate、All-to-All 串成完整链路。
- 再对照 [../src/simulators/moe_routing.py](../src/simulators/moe_routing.py) 的 `topk_route()`、`load_balancing_loss()`、`expert_capacity()`、`dispatch_to_experts()`、`all_to_all_bytes()`。
- 如果你想从单机路由继续扩展到系统部署，再读 [../notes/distributed/moe-ep.md](../notes/distributed/moe-ep.md) 和 [../notes/distributed/moe-inference-deep.md](../notes/distributed/moe-inference-deep.md)。
- 最后跑 `python -m pytest tests/test_moe_routing.py -v`，确认路由、容量和通信量的最小实现是自洽的。
