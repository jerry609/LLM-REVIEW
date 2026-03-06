# KV 驱逐策略数学建模详解

> **核心定位**：系统性梳理 KV Cache 驱逐（Eviction）策略的数学框架——从经典 LRU/LFU 基线，到注意力感知的 H2O / StreamingLLM / SnapKV / Ada-KV 等方法，再到多租户场景下的公平性约束与命中率优化理论。每种策略均给出精确的重要性评分公式和驱逐判定规则。

---

## 1. 问题定义

KV Cache 的总容量为 $C$（以 token 数为单位），但序列在不断增长。当已存储的 token 数 $T > C$ 时，系统必须选择**驱逐**一部分 token 的 KV 表示，以腾出空间。

**驱逐目标的三元优化**：
$$
\min_{\text{eviction policy}} \left( \underbrace{\text{Miss Rate}}_{\text{命中率}} + \lambda_1 \cdot \underbrace{\text{Recompute Cost}}_{\text{重算成本}} + \lambda_2 \cdot \underbrace{\text{Unfairness}}_{\text{多租户公平性}} \right)
$$

---

## 2. 经典缓存策略基线

### 2.1 LRU (Least Recently Used)

$$
\text{Score}_{\text{LRU}}(i) = t_{\text{current}} - t_{\text{last\_access}}(i)
$$

驱逐分数最高（最久未访问）的 token。
- **时间复杂度**：$\mathcal{O}(1)$（双向链表 + 哈希表）。
- **缺陷**：完全不考虑 token 的内容重要性。

### 2.2 LFU (Least Frequently Used)

$$
\text{Score}_{\text{LFU}}(i) = \sum_{t=1}^{T_{\text{current}}} \mathbb{1}[\text{token } i \text{ 被访问于 step } t]
$$

驱逐历史访问频次最低的 token。
- **缺陷**：对突发模式响应慢；历史高频但已失去价值的 token 难以驱逐。

---

## 3. 注意力感知驱逐策略

### 3.1 H2O (Heavy-Hitter Oracle)

> **出处**：Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", 2023

**核心思想**：累积注意力分数最高的 token 是"Heavy Hitter"，最值得保留。

$$
\text{Score}_{\text{H2O}}(i) = \sum_{t=1}^{T_{\text{current}}} \sum_{h=1}^{H} a_{t,h}(i)
$$

其中 $a_{t,h}(i)$ 是第 $t$ 步、第 $h$ 个头对位置 $i$ 的注意力权重。

**保留策略**：Top-K 高分 token + 最近 $w$ 个 token（Sliding Window），总预算 $C = K + w$。

### 3.2 StreamingLLM (Attention Sink)

> **出处**：Xiao et al., "Efficient Streaming Language Models with Attention Sinks", 2023

**关键发现**：序列开头的几个 token（即使内容无关）总是获得极高的注意力分数，称为**注意力汇聚点（Attention Sink）**。丢弃它们会导致模型输出严重退化。

**保留策略**（极简但有效）：
$$
\text{Keep} = \underbrace{\{1, 2, \dots, s\}}_{\text{Sink tokens}} \cup \underbrace{\{T-w+1, \dots, T\}}_{\text{最近 } w \text{ 个}}
$$

总缓存大小固定为 $s + w$，与序列长度 $T$ 完全无关 → 支持**无限长度**的流式推理。

### 3.3 SnapKV

> **出处**：Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation", 2024

**核心思想**：利用 Prefill 末尾的一小段"观察窗口"（Observation Window）的注意力模式，**一次性**决定整个历史 KV Cache 中哪些 token 值得保留。

**算法步骤**：
1. 取 Prompt 最后 $w_{\text{obs}}$ 个 token 的注意力权重矩阵 $A \in \mathbb{R}^{w_{\text{obs}} \times T}$。
2. 对每个历史位置 $i$ 计算"投票分"：

$$
\text{Score}_{\text{SnapKV}}(i) = \sum_{t=T-w_{\text{obs}}+1}^{T} \sum_{h=1}^{H} a_{t,h}(i)
$$

3. 按分数选 Top-$K$ 保留，丢弃其余。

**优势**：仅在 Prefill 结束时做一次决策，Decode 过程中无需反复重算分数。

### 3.4 Ada-KV / PyramidKV

> **出处**：Feng et al., "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference", 2024

**核心思想**：不同层、不同头需要的 KV 预算应该**不同**。注意力越分散（Entropy 越高）的头需要保留更多 token。

层/头 $l, h$ 的注意力熵：
$$
\mathcal{H}_{l,h} = -\sum_{i=1}^{T} \bar{a}_{l,h}(i) \log \bar{a}_{l,h}(i)
$$

其中 $\bar{a}_{l,h}(i)$ 是在观察窗口内平均的注意力权重。

**自适应预算分配**：
$$
C_{l,h} = C_{\text{total}} \times \frac{\mathcal{H}_{l,h}}{\sum_{l',h'} \mathcal{H}_{l',h'}}
$$

熵高的头（注意力分散，需要看更多 token）分配更多预算；熵低的头（只关注少数位置）分配更少预算。

---

## 4. 结构化保护策略 (ACTA)

> **出处**：Patel et al., "ACTA: Advanced Cache Token Allocation", 2024

**核心思想**：某些 token（System Prompt、标签、用户 Query 的关键词）无论注意力分数如何，都必须被**无条件保护**。

**两阶段策略**：

$$
\text{Phase 1: } \text{Protected} = \text{System} \cup \text{Label} \cup \text{Query\_Keywords}
$$
$$
\text{Phase 2: } \text{Score-based fill} = \text{Top-}(C - |\text{Protected}|) \text{ from remaining by attention score}
$$

**评估指标**：Label Token Survival Rate (LTSR)
$$
\text{LTSR} = \frac{|\text{Label tokens surviving in cache}|}{|\text{Total label tokens}|}
$$

理想情况 $\text{LTSR} = 1$（所有标签 token 都被保留）。

---

## 5. 统一价值打分框架

将上述策略统一为一个**加权打分模型**：

$$
\text{Value}(i) = \alpha \cdot \text{AttnScore}(i) + \beta \cdot \text{RecomputeCost}(i) + \gamma \cdot \text{Recency}(i) + \delta \cdot \text{StructuralPriority}(i)
$$

| 分量 | 含义 | 典型来源 |
|------|------|---------|
| $\text{AttnScore}$ | 累积注意力权重 | H2O / SnapKV |
| $\text{RecomputeCost}$ | 重算该 token 的代价 | $\propto$ 该 token 所在 prefix 长度 |
| $\text{Recency}$ | 最近访问时间 | LRU 信号 |
| $\text{StructuralPriority}$ | 结构重要性 | ACTA 的保护列表 |

驱逐规则：**优先驱逐 $\text{Value}(i)$ 最小的 token**。

---

## 6. 命中率理论

### 6.1 理论最优 (Bélády's Algorithm)

Bélády 算法驱逐**未来最晚被再次访问**的块。它给出了命中率的**理论上界**，但无法在线实现（需要知道未来的访问序列）。

### 6.2 有效容量

$$
\text{Effective Capacity} = C \times \text{Hit Rate}
$$

若命中率为 $90\%$，则 $C = 1000$ 的缓存实际只等价于 $900$ 的有效容量。

### 6.3 监控指标

- **Refill Rate**（回填率）= 被驱逐后又被重算的 token 比例。Refill Rate 高说明驱逐策略存在严重误判。
- **驱逐开销**：$\text{overhead} = f_{\text{evict}} \times c_{\text{per\_evict}}$（驱逐频率 × 单次驱逐成本）。

---

## 7. 驱逐粒度选择

| 粒度 | 优势 | 劣势 | 代表系统 |
|------|------|------|---------|
| **Token 级** | 最精细，最高命中率 | 管理开销大 | H2O, SnapKV |
| **Block 级** | 与 PagedAttention 天然兼容 | 整块驱逐可能含有重要 token | vLLM |
| **Layer 级** | 某些层 KV 可牺牲 | 影响粒度粗 | PyramidKV |

---

## 8. 面试实战追问

**Q1：H2O 和 SnapKV 的核心区别是什么？**
> H2O 在每一步 Decode 时**动态**更新所有 token 的累积注意力分数，计算开销更大但更精确。SnapKV 只在 Prefill 结束时做**一次性**决策，运行时零开销，但无法适应 Decode 过程中注意力模式的变化。

**Q2：为什么 StreamingLLM 要保留开头的 Sink Token？**
> Softmax 的归一化特性要求注意力权重之和为 1。当模型对某些位置"不知道该关注哪里"时，它会把注意力"倒"到序列开头的 token 上（即使内容无关）。如果删除这些 token，归一化分母发生剧变，导致所有注意力分布紊乱，输出严重退化。

---

## 9. 对应源码与阅读顺序

- 先读 [../notes/kv-eviction/formula-to-code-walkthrough.md](../notes/kv-eviction/formula-to-code-walkthrough.md)，把 LRU、LFU、Fair quota 三类驱逐策略放到同一个“价值函数 / 预算约束”框架里理解。
- 再看 [../src/kv_cache/core.py](../src/kv_cache/core.py)，重点关注 `last_access_step`、`use_count`、`num_blocks()` 这些元数据是如何随着访问过程被维护的。
- 接着看 [../src/kv_cache/eviction/policies.py](../src/kv_cache/eviction/policies.py)，对照三种 `select_victim()` 的排序键，理解为什么不同策略会选出不同牺牲者。
- 最后跑 `python -m pytest tests/test_kv_cache.py -v`，检查缓存生命周期、访问统计和驱逐结果是否与策略预期一致。
