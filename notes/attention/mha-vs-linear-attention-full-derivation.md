# 传统 MHA vs 线性注意力：全流程数学与矩阵推导对比

> **摘要**：本文以严格的数学形式，完整推导 Multi-Head Attention (MHA) 与 Linear Attention 在 Prefill 和 Decode 两个阶段的每一步计算。线性注意力的核心洞察在于：用特征映射函数 $\phi(\cdot)$ 替代 Softmax 的非线性枷锁后，利用**矩阵乘法结合律**，将 $\text{Softmax}(QK^\top)V$ 的 $O(L^2)$ 计算重新排列为 $\phi(Q)(\phi(K)^\top V)$ 的 $O(L \cdot d_h^2)$ 计算——彻底消除 $L \times L$ 的注意力矩阵。在 Decode 阶段，线性注意力进一步退化为一个**恒定大小的状态机**（RNN），实现 $O(1)$ 的单步推理。

---

## 核心思想对比

| | MHA（全局稠密注意力） | 线性注意力（Linear Attention） |
|--|---------------------|-------------------------------|
| **核心公式** | $\text{Softmax}(QK^\top)V$ | $\phi(Q)\bigl(\phi(K)^\top V\bigr)$ |
| **瓶颈** | Softmax 锁死 $Q$ 和 $K$，必须生成 $L \times L$ 矩阵 | 结合律打破锁定，中间矩阵仅 $d_h \times d_h$ |
| **Prefill 复杂度** | $O(L^2 \cdot d_h)$ | $O(L \cdot d_h^2)$ |
| **Decode 复杂度** | $O(L \cdot d_h)$（每步，随 $L$ 线性增长） | $O(d_h^2)$（每步，恒定！） |
| **KV Cache** | 随序列长度无限增长 | 固定大小的状态矩阵 $S_t \in \mathbb{R}^{d_h \times d_h}$ |
| **类比** | 侦探每次都要翻遍**所有笔记本** | 侦探只维护一份**不断更新的摘要** |

---

## 符号约定

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $L$ | 序列长度 | 128K+ |
| $d$ | 模型隐藏维度 | 4096 |
| $d_h$ | 每头维度 | 128 |
| $n_h$ | 注意力头数 | 32 |
| $\phi(\cdot)$ | 核特征映射函数 | $\text{elu}(x)+1$、$\text{ReLU}(x)$、$1+\text{elu}(x)$ |
| $S_t$ | 线性注意力的隐状态矩阵 | $\in \mathbb{R}^{d_h \times d_h}$ |
| $Z_t$ | 线性注意力的归一化因子向量 | $\in \mathbb{R}^{d_h \times 1}$ |

---

## 一、输入定义

两种机制共享完全相同的输入：

$$X \in \mathbb{R}^{L \times d}$$

---

## 二、训练 / Prefill 阶段

### 2.1 Q、K、V 投影（两者完全相同）

对某个注意力头：

$$Q = X W^Q \in \mathbb{R}^{L \times d_h}, \qquad K = X W^K \in \mathbb{R}^{L \times d_h}, \qquad V = X W^V \in \mathbb{R}^{L \times d_h}$$

### 2.2 核函数映射（线性注意力独有）

#### MHA

**无此步骤。** 直接使用原始 $Q, K$。

#### 线性注意力

将 $Q$ 和 $K$ 通过特征映射函数 $\phi(\cdot)$ 映射到**正值域**：

$$\tilde{Q} = \phi(Q) \in \mathbb{R}^{L \times d_h}, \qquad \tilde{K} = \phi(K) \in \mathbb{R}^{L \times d_h}$$

> **为什么要保证正值？** Softmax 输出的注意力权重天然是非负的（$\exp(\cdot) > 0$），并且和为 1。线性注意力用 $\phi(\cdot)$ 替代后，需要手动确保非负性，否则"注意力权重为负"在物理上没有意义。常见的 $\phi$ 选择：
>
> | $\phi(x)$ | 优点 | 缺点 |
> |-----------|------|------|
> | $\text{elu}(x) + 1$ | 平滑，可导 | 计算略重 |
> | $\text{ReLU}(x)$ | 极简，天然稀疏 | 零值多，信息丢失 |
> | $1 + \text{elu}(x)$ | 保证下界为 1，数值稳定 | — |
> | $\exp(x)$ | 最接近 Softmax 行为 | 数值溢出风险 |

### 2.3 注意力矩阵计算（⭐ 核心差异）

#### MHA：必须先算 $L \times L$ 矩阵

$$A = Q K^\top \in \mathbb{R}^{L \times L}$$

$$S = \text{Softmax}(A / \sqrt{d_h}) \in \mathbb{R}^{L \times L}$$

> **问题所在**：无论后续怎么优化，Softmax 要求**先看到所有 $L$ 个 Key 的分数**才能归一化。这把 $Q$ 和 $K^\top$ 锁死在一起，无法拆分。

#### 线性注意力：利用结合律，先算 $d_h \times d_h$ 矩阵

**数学推导——结合律的魔法：**

传统注意力（忽略 Softmax 归一化）的本质是：

$$O = \underbrace{(\tilde{Q} \, \tilde{K}^\top)}_{L \times L} \; V$$

矩阵乘法满足**结合律**，因此可以改变计算顺序：

$$O = \tilde{Q} \; \underbrace{(\tilde{K}^\top \, V)}_{d_h \times d_h}$$

定义**全局隐状态矩阵**：

$$\boxed{S_{\text{global}} = \tilde{K}^\top V \in \mathbb{R}^{d_h \times d_h}}$$

> **关键洞察**：$S_{\text{global}}$ 的尺寸完全独立于序列长度 $L$！无论文本有多长，它始终是一个 $d_h \times d_h = 128 \times 128$ 的固定矩阵。

### 2.4 加权输出与归一化

#### MHA

Softmax 自带归一化（分母 $\sum_j \exp(a_{ij})$）：

$$O = S \, V \in \mathbb{R}^{L \times d_h}$$

#### 线性注意力

需要手动归一化，模拟 Softmax 的"权重和为 1"约束：

**分子：**

$$\text{Numerator} = \tilde{Q} \, S_{\text{global}} = \tilde{Q} \, (\tilde{K}^\top V) \in \mathbb{R}^{L \times d_h}$$

**分母（归一化因子）：**

$$Z_{\text{global}} = \tilde{K}^\top \mathbf{1}_L \in \mathbb{R}^{d_h \times 1}$$

$$\text{Denominator} = \tilde{Q} \, Z_{\text{global}} \in \mathbb{R}^{L \times 1}$$

**最终输出（逐行除法）：**

$$\boxed{O_i = \frac{\tilde{q}_i \, S_{\text{global}}}{\tilde{q}_i \, Z_{\text{global}}}, \qquad i = 1, \ldots, L}$$

### 2.5 Prefill 复杂度对比

| | MHA | 线性注意力 |
|--|-----|-----------|
| **时间复杂度** | $O(L^2 \cdot d_h)$ | $O(L \cdot d_h^2)$ |
| **空间复杂度** | $O(L^2)$（注意力矩阵） | $O(L \cdot d_h)$（无 $L^2$ 矩阵） |
| **当 $L > d_h$ 时** | $L^2$ 主导，极度昂贵 | $L \cdot d_h^2$ 更优 |
| **交叉点** | — | 当 $L = d_h$（如 $L = 128$）时两者等价 |

> **直觉**：$d_h$ 通常是 128，当 $L = 4096$ 时，MHA 需要计算 $4096^2 = 16M$ 个注意力分数；线性注意力只需要维护一个 $128^2 = 16K$ 的状态矩阵——差了 **1000 倍**。当 $L = 128\text{K}$ 时差距更是天文数字。

---

## 三、KV Cache 阶段

这是线性注意力最具革命性的差异——**Cache 不再增长**。

### MHA：线性增长的 Cache

存储**每一个 Token** 产生的 K、V：

$$\text{Cache} = \{K_1, V_1, K_2, V_2, \ldots, K_t, V_t\}$$

$$\text{Cache}_K \in \mathbb{R}^{t \times d_h}, \qquad \text{Cache}_V \in \mathbb{R}^{t \times d_h}$$

> **致命问题**：Cache 大小随对话长度 $t$ **无限线性增长**。

### 线性注意力：恒定大小的"隐状态"

像 RNN 一样，仅存储**累积后的隐状态矩阵**和**归一化向量**：

$$\text{Cache} = \{S_t, Z_t\}$$

$$S_t \in \mathbb{R}^{d_h \times d_h}, \qquad Z_t \in \mathbb{R}^{d_h \times 1}$$

> **革命性优势**：无论对话了 1 千字还是 100 万字，Cache **始终只有** $d_h^2 + d_h$ 个浮点数！

### Cache 大小对比

| 序列长度 $L$ | MHA KV Cache（每层每头） | 线性注意力 Cache（每层每头） | MHA / Linear |
|-------------|------------------------|---------------------------|-------------|
| 4K | $2 \times 4096 \times 128 = 1$ M 维 | $128^2 + 128 = 16.5$ K 维 | **63×** |
| 32K | $2 \times 32768 \times 128 = 8.4$ M 维 | $16.5$ K 维（不变） | **509×** |
| 128K | $2 \times 131072 \times 128 = 33.6$ M 维 | $16.5$ K 维（不变） | **2,036×** |
| 1M | $2 \times 10^6 \times 128 = 256$ M 维 | $16.5$ K 维（不变） | **15,515×** |

---

## 四、推理 / Decode 阶段

### 4.1 计算新向量

#### MHA

$$q_t = x_t W^Q, \qquad k_t = x_t W^K, \qquad v_t = x_t W^V$$

#### 线性注意力

$$\tilde{q}_t = \phi(x_t W^Q), \qquad \tilde{k}_t = \phi(x_t W^K), \qquad v_t = x_t W^V$$

### 4.2 Cache / 状态更新（⭐ 核心差异）

#### MHA：追加（Append）

把新的 $k_t, v_t$ 拼接到 Cache 后面，Cache **变长**：

$$\text{Cache}_K \leftarrow [\text{Cache}_K;\; k_t] \qquad (\text{长度从 } t{-}1 \text{ 变为 } t)$$

$$\text{Cache}_V \leftarrow [\text{Cache}_V;\; v_t]$$

#### 线性注意力：累加（Accumulate）— 完美的 $O(1)$ 状态机

当前状态 = 历史状态 + 新 Token 的**外积**：

$$\boxed{S_t = S_{t-1} + \tilde{k}_t^\top \, v_t}$$

$$\boxed{Z_t = Z_{t-1} + \tilde{k}_t^\top}$$

> **数学解读**：$\tilde{k}_t^\top v_t$ 是一个 $d_h \times d_h$ 的外积矩阵（Outer Product），代表"新 Token 的 Key-Value 关联信息"。将它**累加**到 $S_t$ 中，就像把一滴墨水融入水池——新 Token 的信息被永久"融化"进了全局状态。

### 4.3 Attention 输出

#### MHA：必须和所有历史重新计算

$$\text{Score} = q_t \cdot (\text{Cache}_K)^\top \in \mathbb{R}^{1 \times t}$$

$$o_t = \text{softmax}(\text{Score}) \cdot \text{Cache}_V \in \mathbb{R}^{1 \times d_h}$$

#### 线性注意力：直接查询状态矩阵

$$\boxed{o_t = \frac{\tilde{q}_t \, S_t}{\tilde{q}_t \, Z_t}}$$

> **直觉**：$\tilde{q}_t \, S_t$ 就是用当前的 Query 去"查询"凝缩了所有历史知识的状态矩阵。$\tilde{q}_t \, Z_t$ 是归一化因子，确保输出的尺度合理。

### 4.4 Decode 复杂度对比

| | MHA | 线性注意力 |
|--|-----|-----------|
| **状态更新** | $O(d_h)$（拼接） | $O(d_h^2)$（外积累加） |
| **Attention 计算** | $O(t \cdot d_h)$（随 $t$ 线性增长！） | $O(d_h^2)$（恒定！） |
| **Cache 读取** | $O(t \cdot d_h)$（读取全量 Cache） | $O(d_h^2)$（读取固定状态） |
| **总计 / 每步** | $O(t \cdot d_h)$ | $O(d_h^2)$ |

> **关键区别**：MHA 的 Decode 耗时随对话长度 $t$ **线性增长**——第 10 万个词的生成比第 1 个词慢 10 万倍。而线性注意力的每一步**完全恒定**——无论是第 1 个词还是第 100 万个词，生成耗时一模一样。

---

## 五、线性注意力 = Transformer 化身 RNN

### 5.1 形式等价

将线性注意力的 Decode 过程写成递推形式：

$$S_t = S_{t-1} + \tilde{k}_t^\top v_t$$

$$Z_t = Z_{t-1} + \tilde{k}_t^\top$$

$$o_t = \frac{\tilde{q}_t \, S_t}{\tilde{q}_t \, Z_t}$$

对比经典 RNN 的递推：

$$h_t = f(h_{t-1}, x_t)$$

$$o_t = g(h_t)$$

> **完全对应**：$S_t$ 就是 RNN 的隐状态 $h_t$，更新规则从"非线性变换"变成了"外积累加"。线性注意力本质上是一个**矩阵值的 RNN**。

### 5.2 Transformer ↔ RNN 统一视角

| 特性 | 传统 MHA | 线性注意力 | 经典 RNN (LSTM) |
|------|---------|-----------|----------------|
| **历史访问** | 全量访问所有历史 Token | 通过状态矩阵隐式访问 | 通过隐状态隐式访问 |
| **Decode 复杂度** | $O(t)$ | $O(1)$ | $O(1)$ |
| **并行训练** | ✓（全序列一次算完） | ✓（结合律批量计算） | ✗（必须逐步递推） |
| **状态大小** | $O(t)$ 增长 | $O(d_h^2)$ 恒定 | $O(d_h)$ 恒定 |
| **精确检索** | ✓（精确匹配每个 Token） | △（信息被"融化"，部分丢失） | ✗（信息遗忘严重） |
| **训练效率** | $O(L^2)$ | $O(L)$ | $O(L)$（但不可并行） |

> **线性注意力的独特优势**：它兼具了 Transformer 的**并行训练能力**和 RNN 的**恒定推理开销**——而传统 RNN 无法并行训练，传统 Transformer 推理开销随序列增长。

---

## 六、精度惩罚：为什么主流大模型不全用线性注意力？

### 6.1 Softmax 的"赢者通吃"特性

Softmax 的指数函数天然产生**尖锐分布**——少数高分 Token 获得绝大部分注意力权重：

$$\text{Softmax}(z_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

当 $z_i$ 远大于其他分数时，$\text{Softmax}(z_i) \approx 1$，其余趋近于 0。

> **精确检索能力**：这使得 MHA 能从 128K 个历史 Token 中精确定位到那**唯一正确**的 Token（如大海捞针实验）。

### 6.2 线性注意力的"特征平均化"问题

$\phi(\cdot)$ 的映射相对平滑，缺乏 Softmax 的指数放大效应：

$$\text{Linear Attention Weight} \propto \phi(q)^\top \phi(k) \quad \text{（无指数放大，分布更平坦）}$$

所有历史 Token 的信息被"融化"进同一个 $S_t$ 矩阵，相当于做了一种**加权平均**。当需要精确回忆某个具体细节时，被平均化的信息可能**无法被准确提取**。

> **"大海捞针"困境**：如果在 10 万词的文档中插入一句关键信息，MHA 可以通过尖锐的注意力权重精准定位；线性注意力的平滑分布可能会让这条关键信息"淹没"在海量上下文中。

### 6.3 状态压缩的信息瓶颈

$S_t \in \mathbb{R}^{d_h \times d_h}$ 是一个固定大小的矩阵。当序列足够长时：

$$\text{信息量（bits）} = O(L \cdot d_h) \gg \text{状态容量} = O(d_h^2)$$

> 以 $d_h = 128, L = 100\text{K}$ 为例：信息量 $\propto 12.8\text{M}$，状态容量 $\propto 16\text{K}$——**差 800 倍**。这意味着绝大部分历史信息在压缩过程中**不可避免地丢失**。

---

## 七、前沿发展：线性注意力的变体与演进

学术界目前最活跃的研究方向之一，就是在**线性注意力的效率**和**MHA 的精度**之间寻找最佳平衡点：

| 模型 / 方法 | 核心创新 | 状态更新规则 |
|-------------|---------|-------------|
| **Mamba (S4/S6)** | 选择性状态空间模型，输入依赖的门控 | $h_t = A_t h_{t-1} + B_t x_t$ |
| **RetNet** | 引入衰减因子 $\gamma$，自动遗忘远古历史 | $S_t = \gamma \, S_{t-1} + k_t^\top v_t$ |
| **RWKV** | 时间混合（Time-Mixing）+ 通道混合 | $wkv_t = \alpha \, wkv_{t-1} + e^{u+k_t} v_t$ |
| **GLA** | 门控线性注意力，学习的衰减矩阵 | $S_t = G_t \odot S_{t-1} + k_t^\top v_t$ |
| **DeltaNet** | Delta Rule 更新，选择性覆写 | $S_t = S_{t-1} + k_t^\top (v_t - S_{t-1}^\top k_t)$ |
| **Based** | 短卷积 + 线性注意力混合 | 混合局部与全局注意力 |

### 共同趋势

1. **引入衰减/遗忘**：给 $S_t$ 加上衰减因子 $\gamma < 1$，让远古信息自然淡出（模拟人类遗忘曲线）
2. **输入依赖的门控**：让更新规则依赖当前输入，而非简单累加（选择性记忆）
3. **混合架构**：在部分层使用线性注意力（处理长距离），部分层保留标准注意力（精确检索）
4. **硬件适配**：设计 IO 高效的 chunk-wise 并行训练算法（如 Mamba 的 selective scan）

---

## 八、全局因果链

```text
传统 MHA:

    x_t ──→ [Q, K, V 投影] ──→ [读取全量 Cache: O(t)] ──→ [Q·K^T: O(t)] ──→ [Softmax·V] ──→ output
                                        ↑
                                  Cache 无限增长
                                 Memory Bandwidth Bound

线性注意力:

    x_t ──→ [Q, K, V 投影] ──→ [φ(·) 特征映射]
                                      │
                                      ▼
              [外积累加: S_t += k̃^T v_t]    ← O(d_h²) 恒定更新
                        │
                        ▼
              [查询状态: o_t = q̃ · S_t / q̃ · Z_t]    ← O(d_h²) 恒定计算
                        │
                        ▼
                      output

              状态大小恒定：128 × 128 = 16K 浮点数
              无论序列多长，每步耗时完全相同
```

---

## 九、注意力机制演进全景

$$\underbrace{\text{MHA}}_{\substack{\text{全量 KV} \\ \text{全量序列} \\ O(L^2)}} \xrightarrow{\text{KV 头裁撤}} \underbrace{\text{GQA}}_{\substack{\text{分组 KV} \\ \text{全量序列} \\ O(L^2)}} \xrightarrow{\text{特征压缩}} \underbrace{\text{MLA}}_{\substack{\text{潜在 KV} \\ \text{全量序列} \\ O(L^2)}} \xrightarrow{\text{序列压缩}} \underbrace{\text{DSA}}_{\substack{\text{潜在 KV} \\ \text{稀疏序列} \\ O(L \cdot k)}} \xrightarrow{\text{结合律}} \underbrace{\text{Linear}}_{\substack{\text{固定状态} \\ \text{无序列维度} \\ O(d_h^2)}}$$

| 方案 | Decode 复杂度 | Cache 增长 | 精确检索 | 代表模型 |
|------|-------------|-----------|---------|---------|
| **MHA** | $O(L)$ | 线性增长 | ★★★★★ | GPT-3/4 |
| **GQA** | $O(L)$ | 线性（减缓） | ★★★★☆ | LLaMA-3 |
| **MLA** | $O(L)$ | 线性（大幅减缓） | ★★★★☆ | DeepSeek-V2/V3 |
| **DSA** | $O(k)$ 恒定 | 线性（含索引） | ★★★★☆ | DeepSeek 下一代 |
| **Linear** | $O(1)$ 恒定 | **恒定** | ★★★☆☆ | Mamba, RWKV, RetNet |

> **注意**：这不是简单的"越新越好"。MHA 和 GQA 在精确检索上仍然无可替代；线性注意力在超长文本的流式生成场景（如持续对话、代码补全）中优势巨大。现代最优架构（如 Jamba, Zamba）倾向于**混合使用**两者。

---

## 参考文献

1. Katharopoulos, A. et al. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020.
2. Gu, A. & Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752, 2023.
3. Sun, Y. et al. *Retentive Network: A Successor to Transformer for Large Language Models.* arXiv:2307.08621, 2023.
4. Peng, B. et al. *RWKV: Reinventing RNNs for the Transformer Era.* EMNLP 2023.
5. Yang, S. et al. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML 2024.
6. Schlag, I. et al. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021.
7. Choromanski, K. et al. *Rethinking Attention with Performers.* ICLR 2021.
8. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
