# 传统 MHA vs GQA：全流程数学与矩阵推导对比

> **摘要**：本文以严格的数学形式，逐步对比推导 Multi-Head Attention (MHA) 与 Grouped-Query Attention (GQA) 在 **训练/Prefill** 和 **推理/Decode** 两个阶段的每一步计算。GQA 通过"分组共享 KV 头"实现了 KV Cache 的物理维度裁撪，在几乎不损失模型质量的前提下将 KV Cache 显存压缩至 MHA 的 $n_g / n_h$（典型值为 1/4）。文末深入对比 GQA 与 MLA 截然不同的压缩哲学。

---

## 核心思想对比

| | MHA (Multi-Head Attention) | GQA (Grouped-Query Attention) |
|--|---------------------------|-------------------------------|
| **Q:KV 关系** | 1 对 1——每个 Query 头都有专属的 Key 和 Value 头 | 多对 1——同组内多个 Query 头共享一个 KV 头 |
| **类比** | 32 个侦探，每人有专属的资料库 | 32 个侦探分 8 组，每组共用一个资料库 |
| **代表模型** | GPT-3, LLaMA-1, OPT, Bloom | **LLaMA-2/3, Mistral, Qwen-1.5/2, Gemma**（当前行业绝对主流） |

---

## 符号约定

| 符号 | 含义 | 典型值 (LLaMA-3 8B) |
|------|------|---------------------|
| $L$ | 序列长度 (Sequence Length) | 可变 |
| $d$ | 模型隐藏维度 (d_model) | 4096 |
| $n_h$ | Query 的注意力头数 | 32 |
| $d_h$ | 每头维度 $= d / n_h$ | 128 |
| $n_g$ | KV 的组数 / 头数 | 8 |
| $G$ | 每个 KV 头服务的 Q 头数 $= n_h / n_g$ | 4 |

**索引约定**：
- $i \in \{1, \ldots, n_h\}$ 表示第 $i$ 个 Query 头
- $g \in \{1, \ldots, n_g\}$ 表示第 $g$ 个 KV 组
- $g(i) = \lfloor (i-1) / G \rfloor + 1$ 表示第 $i$ 个 Q 头所属的 KV 组索引

> **举例**（LLaMA-3 8B, $G = 4$）：Q 头 1, 2, 3, 4 → 共享 KV 组 1；Q 头 5, 6, 7, 8 → 共享 KV 组 2；……Q 头 29, 30, 31, 32 → 共享 KV 组 8。

---

## 一、输入定义

两种机制共享完全相同的输入：

$$X \in \mathbb{R}^{L \times d}$$

其中 $X$ 是经过 Embedding + 前序 Transformer Block 计算后、进入当前注意力层的隐状态矩阵。

---

## 二、训练 / Prefill 阶段

### 2.1 Query 投影（完全相同）

两种方式在 Query 投影上没有任何区别——都对 $n_h$ 个头独立投影：

#### MHA

$$Q_i = X \, W_i^Q \quad \in \mathbb{R}^{L \times d_h}, \qquad W_i^Q \in \mathbb{R}^{d \times d_h}, \quad i = 1, \ldots, n_h$$

#### GQA

$$Q_i = X \, W_i^Q \quad \in \mathbb{R}^{L \times d_h}, \qquad W_i^Q \in \mathbb{R}^{d \times d_h}, \quad i = 1, \ldots, n_h$$

> **理解要点**：GQA 不减少 Query 头数。模型的"观察视角"数量不变——它依然用 32 个不同的"眼睛"去审视输入。GQA 减少的是"资料库"的数量。

### 2.2 Key、Value 投影（⭐ 核心差异点）

#### MHA：各自为政——生成 $n_h$ 份 KV

$$K_i = X \, W_i^K \quad \in \mathbb{R}^{L \times d_h}, \qquad V_i = X \, W_i^V \quad \in \mathbb{R}^{L \times d_h}$$

其中 $W_i^K, W_i^V \in \mathbb{R}^{d \times d_h}$，$i = 1, \ldots, n_h$。

**投影矩阵总参数量**：$2 \times n_h \times d \times d_h = 2 \times 32 \times 4096 \times 128 \approx 33.6M$

#### GQA：按组共享——只生成 $n_g$ 份 KV

$$K_g = X \, W_g^K \quad \in \mathbb{R}^{L \times d_h}, \qquad V_g = X \, W_g^V \quad \in \mathbb{R}^{L \times d_h}$$

其中 $W_g^K, W_g^V \in \mathbb{R}^{d \times d_h}$，$g = 1, \ldots, n_g$。

**投影矩阵总参数量**：$2 \times n_g \times d \times d_h = 2 \times 8 \times 4096 \times 128 \approx 8.4M$

> **参数量压缩比**：$n_h / n_g = 32 / 8 = 4\times$。KV 投影矩阵的参数量直接缩减为 MHA 的 $1/4$。

### 2.3 位置编码（RoPE）

#### MHA

$$\hat{Q}_i = \text{RoPE}(Q_i), \qquad \hat{K}_i = \text{RoPE}(K_i)$$

对全部 $n_h$ 个 Q 和 $n_h$ 个 K 分别施加 RoPE。

#### GQA

$$\hat{Q}_i = \text{RoPE}(Q_i), \qquad \hat{K}_g = \text{RoPE}(K_g)$$

Q 依然对全部 $n_h$ 个头施加 RoPE。但 K 只对 $n_g$ 个组施加 RoPE——因为本来就只有 $n_g$ 份 K。

> **RoPE 计算量压缩**：K 的 RoPE 计算从 $n_h \times L \times d_h$ 降至 $n_g \times L \times d_h$，压缩 $4\times$。

### 2.4 Attention 计算

#### MHA

$$S_i = \text{softmax}\!\left(\frac{\hat{Q}_i \, \hat{K}_i^\top}{\sqrt{d_h}} + M\right) \quad \in \mathbb{R}^{L \times L}$$

$$O_i = S_i \, V_i \quad \in \mathbb{R}^{L \times d_h}$$

每个 Q 头 $i$ 与自己专属的 K 头 $i$ 做点积，加权求和自己专属的 V 头 $i$。

#### GQA

$$S_i = \text{softmax}\!\left(\frac{\hat{Q}_i \, \hat{K}_{g(i)}^\top}{\sqrt{d_h}} + M\right) \quad \in \mathbb{R}^{L \times L}$$

$$O_i = S_i \, V_{g(i)} \quad \in \mathbb{R}^{L \times d_h}$$

同组内的多个 $Q_i$（例如 $Q_1, Q_2, Q_3, Q_4$）都与**同一个** $K_{g(1)}$ 做点积，加权求和**同一个** $V_{g(1)}$。

> **关键洞察**：虽然同组的 4 个 Q 头共享同一份 KV，但由于它们各自的 $W_i^Q$ 不同，它们会产生**完全不同的注意力分数 $S_i$ 和输出 $O_i$**。信息多样性来自 Q 的多样性，而非 KV 的多样性。

#### 数学展开——为什么共享 KV 不会损失太多信息

考虑 MHA 中同组的 4 个头（假设 $i = 1, 2, 3, 4$，它们在 GQA 中共享组 $g = 1$）：

**MHA 的 4 组注意力分数**：

$$S_1^{\text{MHA}} = f(\hat{Q}_1, \hat{K}_1), \quad S_2^{\text{MHA}} = f(\hat{Q}_2, \hat{K}_2), \quad S_3^{\text{MHA}} = f(\hat{Q}_3, \hat{K}_3), \quad S_4^{\text{MHA}} = f(\hat{Q}_4, \hat{K}_4)$$

**GQA 的 4 组注意力分数**：

$$S_1^{\text{GQA}} = f(\hat{Q}_1, \hat{K}_1), \quad S_2^{\text{GQA}} = f(\hat{Q}_2, \hat{K}_1), \quad S_3^{\text{GQA}} = f(\hat{Q}_3, \hat{K}_1), \quad S_4^{\text{GQA}} = f(\hat{Q}_4, \hat{K}_1)$$

GQA 的核心假设是：在训练充分后，$\hat{K}_1, \hat{K}_2, \hat{K}_3, \hat{K}_4$ 之间的差异其实不大（高度冗余），因此用一个 $\hat{K}_1$ 替代它们，对最终输出的影响可控。

**Ainslie et al. (2023)** 在 GQA 论文中验证了这一假设——从 MHA 到 GQA 的 "uptrained" 模型在各项 benchmark 上的质量损失在 **0.1–0.5%** 以内。

### 2.5 Output 投影（完全相同）

$$O = \text{Concat}(O_1, O_2, \ldots, O_{n_h}) \, W^O \quad \in \mathbb{R}^{L \times d}$$

两种方式都拼接全部 $n_h$ 个头的输出，然后通过 $W^O \in \mathbb{R}^{(n_h \cdot d_h) \times d}$ 投影回模型维度。

---

## 三、KV Cache 阶段

### 3.1 物理存储内容

#### MHA

缓存所有 $n_h$ 个头的完整 K 和 V：

$$\text{Cache}_{\text{MHA}} = \left\{\; \hat{K}_i,\; V_i \;\right\}_{i=1}^{n_h}$$

#### GQA

仅缓存 $n_g$ 个组的 K 和 V：

$$\text{Cache}_{\text{GQA}} = \left\{\; \hat{K}_g,\; V_g \;\right\}_{g=1}^{n_g}$$

### 3.2 显存占用对比（每层每 token）

#### MHA

$$\text{Mem}_{\text{MHA}} = 2 \times n_h \times d_h$$

以 LLaMA-3 8B 为例：$2 \times 32 \times 128 = 8192$ 维 $\xrightarrow{\text{FP16}}$ $16{,}384$ bytes = $16$ KB/token/layer

#### GQA

$$\text{Mem}_{\text{GQA}} = 2 \times n_g \times d_h$$

$2 \times 8 \times 128 = 2048$ 维 $\xrightarrow{\text{FP16}}$ $4{,}096$ bytes = $4$ KB/token/layer

**压缩比：$n_h / n_g = 32 / 8 = 4\times$**

### 3.3 全模型 KV Cache 对比

假设 32 层，序列长度 32K，Batch Size 64，FP16：

| 指标 | MHA | GQA | 压缩比 |
|------|-----|-----|-------|
| 每 token 每层 | 16 KB | 4 KB | 4× |
| 单序列 (32K tokens, 32 layers) | 16.4 GB | 4.1 GB | 4× |
| **Batch=64 总计** | **1,048 GB** | **262 GB** | 4× |

> **实际影响**：在 A100 80GB 上，MHA 的 KV Cache 在 Batch=64、32K 上下文时直接爆掉（需要 1TB+），而 GQA 压缩 4× 后虽然仍然很大（262 GB），但配合 PagedAttention 和量化，可以在多卡上勉强运行。更重要的是——减少 4× 的 Cache 意味着 Decode 时需要从 HBM 读取的数据量也减少 4×，直接缓解了带宽瓶颈。

---

## 四、推理 / Decode 阶段

Decode 阶段是自回归生成：每一步仅输入一个新 token $x_t \in \mathbb{R}^{1 \times d}$，需要与所有历史 token 的 KV Cache 交互。

### 4.1 计算新 Query（完全相同）

#### MHA / GQA

$$q_{t,i} = \text{RoPE}(x_t \, W_i^Q) \quad \in \mathbb{R}^{1 \times d_h}, \qquad i = 1, \ldots, n_h$$

### 4.2 读取 KV Cache（⭐ 带宽差异的关键）

#### MHA

对每个头 $i$，独立从 HBM 读取：

$$\hat{K}_i^{\text{cached}} \in \mathbb{R}^{L \times d_h}, \qquad V_i^{\text{cached}} \in \mathbb{R}^{L \times d_h}$$

**总内存读取量**：$2 \times n_h \times d_h \times L = 2 \times 32 \times 128 \times L = 8192L$ 个参数。

#### GQA

对每个组 $g$，从 HBM 读取一次：

$$\hat{K}_g^{\text{cached}} \in \mathbb{R}^{L \times d_h}, \qquad V_g^{\text{cached}} \in \mathbb{R}^{L \times d_h}$$

然后同组的 $G$ 个 Q 头**在 GPU 的 SRAM 中复用**这份已经拉到片上的 KV Cache。

**总内存读取量**：$2 \times n_g \times d_h \times L = 2 \times 8 \times 128 \times L = 2048L$ 个参数。

> **带宽压缩 $4\times$**：这是 GQA 在 Decode 阶段最核心的工程优势。Decode 的瓶颈是 Memory Bandwidth Bound——GPU 算力用不满，全在等数据从 HBM 搬到 SRAM。GQA 将搬运量减少 4 倍，Decode 吞吐量理论上提升 4 倍。

### 4.3 Attention 点积

#### MHA

$$\text{Score}_{t,i} = q_{t,i} \cdot (\hat{K}_i^{\text{cached}})^\top \quad \in \mathbb{R}^{1 \times L}$$

每个头用自己专属的 K Cache。

#### GQA

$$\text{Score}_{t,i} = q_{t,i} \cdot (\hat{K}_{g(i)}^{\text{cached}})^\top \quad \in \mathbb{R}^{1 \times L}$$

同组的 $G$ 个 Q 头都与同一份 $\hat{K}_{g(i)}^{\text{cached}}$ 做点积。

> **SRAM 复用原理**：在 FlashAttention / FlashDecoding 的实现中，一个 KV Block 被加载到 SRAM 后，同组的 $G$ 个 Q 头依次（或并行）与它做点积。这意味着 1 次 HBM→SRAM 的搬运服务了 $G$ 次点积计算，内存 IO 效率提升 $G$ 倍。

### 4.4 V 的加权和

#### MHA

$$o_{t,i} = \text{softmax}(\text{Score}_{t,i}) \cdot V_i^{\text{cached}} \quad \in \mathbb{R}^{1 \times d_h}$$

#### GQA

$$o_{t,i} = \text{softmax}(\text{Score}_{t,i}) \cdot V_{g(i)}^{\text{cached}} \quad \in \mathbb{R}^{1 \times d_h}$$

同理，同组的 $G$ 个头复用同一份 V Cache。

### 4.5 最终投影（完全相同）

$$o_t = \text{Concat}(o_{t,1}, \ldots, o_{t,n_h}) \cdot W^O = \sum_{i=1}^{n_h} o_{t,i} \, W_i^O \quad \in \mathbb{R}^{1 \times d}$$

---

## 五、GQA 的工程实现核心：`repeat_interleave`

在 PyTorch 实现中，GQA 的核心技巧是在计算注意力前，将 $n_g$ 个 KV 头通过 `repeat_interleave` 广播到 $n_h$ 个：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) — 当前行业主流"""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads          # Q 头数 (32)
        self.n_kv_heads = n_kv_heads    # KV 头数 (8)
        self.n_groups = n_heads // n_kv_heads  # 每组 Q 头数 G (4)
        self.d_head = d_model // n_heads       # 头维度 (128)

        # Q: 32 个头的投影
        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        # K, V: 只有 8 个头的投影 ← 核心差异！
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # Q: [B, N, 32, 128] → [B, 32, N, 128]
        q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # K, V: [B, N, 8, 128] → [B, 8, N, 128]
        k = self.W_k(x).view(B, N, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_kv_heads, self.d_head).transpose(1, 2)

        # ★ 核心：将 KV 头 repeat 到与 Q 头数量对齐 (8 → 32)
        # 每个 KV 头被复制 G=4 次，分别供给 4 个 Q 头使用
        k = k.repeat_interleave(self.n_groups, dim=1)  # [B, 32, N, 128]
        v = v.repeat_interleave(self.n_groups, dim=1)  # [B, 32, N, 128]

        # 后续与标准 MHA 完全一致
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.triu(
            torch.full((N, N), float('-inf'), device=x.device), diagonal=1
        )
        scores = scores + causal_mask
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)
```

> **💡 关键洞察**：`repeat_interleave` 是零成本的逻辑广播——在 FlashAttention 的 CUDA 实现中，它实际上通过指针偏移实现，不会真的在显存中复制 KV 数据。计算时每个 Q 头都"看到了完整信息"，存储时只付 $1/G$ 的代价。

---

## 六、GQA 的特殊情况：MHA 和 MQA

GQA 是一个统一框架，通过调整 $n_g$ 可以退化为两个极端情况：

| 方案 | $n_g$ | $G = n_h / n_g$ | KV Cache 大小 | 特点 |
|------|-------|----------------|--------------|------|
| **MHA** | $n_g = n_h = 32$ | $G = 1$ | $2 \times 32 \times d_h$ | 每个 Q 头有专属 KV，质量最好，显存最大 |
| **GQA** | $n_g = 8$ | $G = 4$ | $2 \times 8 \times d_h$ | 4 个 Q 头共享 1 个 KV，**甜点均衡** |
| **MQA** | $n_g = 1$ | $G = 32$ | $2 \times 1 \times d_h$ | 所有 Q 头共享 1 个 KV，显存最小，质量损失最大 |

> GQA 论文的核心贡献是发现了 $n_g = 8$ 这个"甜点"：KV Cache 压缩 4 倍，质量损失 < 0.5%。

$$\text{MQA} \xleftarrow{n_g = 1} \text{GQA} \xrightarrow{n_g = n_h} \text{MHA}$$

---

## 七、Decode 阶段性能对比

### 7.1 单步 Decode 内存读取量

| 操作 | MHA | GQA | 压缩比 |
|------|-----|-----|-------|
| 读 K Cache | $n_h \times d_h \times L$ | $n_g \times d_h \times L$ | $n_h / n_g$ |
| 读 V Cache | $n_h \times d_h \times L$ | $n_g \times d_h \times L$ | $n_h / n_g$ |
| **总读取** | $2 n_h d_h L$ | $2 n_g d_h L$ | $n_h / n_g = 4\times$ |

### 7.2 Roofline 模型分析

Decode 是 Memory Bandwidth Bound 计算。以 A100 80GB（HBM 带宽 2.0 TB/s）为例：

**MHA Decode 延迟**（单层，$L = 32K$，FP16）：

$$\text{Bytes} = 2 \times 32 \times 128 \times 32768 \times 2 = 536{,}870{,}912 \;\text{B} \approx 512 \;\text{MB}$$

$$t_{\text{MHA}} = \frac{512 \;\text{MB}}{2.0 \;\text{TB/s}} = 0.25 \;\text{ms}$$

**GQA Decode 延迟**（同条件）：

$$\text{Bytes} = 2 \times 8 \times 128 \times 32768 \times 2 = 134{,}217{,}728 \;\text{B} \approx 128 \;\text{MB}$$

$$t_{\text{GQA}} = \frac{128 \;\text{MB}}{2.0 \;\text{TB/s}} = 0.064 \;\text{ms}$$

**单层 Decode 加速比**：$t_{\text{MHA}} / t_{\text{GQA}} = 0.25 / 0.064 \approx 3.9\times$

> 接近理论极限的 $4\times$。剩余的 0.1× 差距来自于 Q 投影、RoPE 计算、输出投影等固定开销。

### 7.3 Batch 吞吐量提升

由于 GQA 的 KV Cache 更小，同一块 GPU 显存可以容纳更多的并发请求：

$$\text{Max Batch}_{\text{GQA}} = G \times \text{Max Batch}_{\text{MHA}} = 4 \times \text{Max Batch}_{\text{MHA}}$$

在推理服务场景中，Batch Size 越大，GPU 利用率越高，每请求成本越低。GQA 的显存节省直接转化为 **4 倍的吞吐量提升**。

---

## 八、GQA vs MLA：两种截然不同的压缩哲学

| 维度 | GQA | MLA |
|------|-----|-----|
| **压缩方式** | **物理维度的裁撤**：直接减少 KV 头数 | **化学层面的提纯**：将所有 KV 信息压缩到低维潜在空间 |
| **类比** | 公司砍掉 24 个资料库，只保留 8 个，4 个部门共用一个 | 公司把 32 个资料库的内容提纯压缩成一个 ZIP 文件，各部门用时实时解压 |
| **逻辑头数** | KV 头数物理减少（$n_h \to n_g$） | 所有 $n_h$ 个逻辑头保留，底层共享一个压缩向量 |
| **KV Cache 大小** | $2 n_g d_h$（如 $2 \times 8 \times 128 = 2048$） | $d_c + d_r$（如 $512 + 64 = 576$） |
| **压缩比** (vs MHA) | $n_h / n_g = 4\times$ | $(2 n_h d_h) / (d_c + d_r) \approx 57\times$ |
| **Decode 优化** | 简单的分组复用（SRAM 内广播） | 矩阵吸收——将解压矩阵融合到 Q 和 O 中，永不解压 Cache |
| **实现难度** | 极低（仅需 `repeat_interleave`） | 高（需要重写 CUDA 算子、解耦 RoPE、预计算融合权重） |
| **框架兼容** | 几乎所有推理框架原生支持 | 早期 vLLM 等框架不支持，需要专用实现 |
| **质量损失** | $< 0.5\%$（已被大量实验验证） | 理论上无损（端到端训练自适应） |
| **压缩极限** | $n_g = 1$（MQA）时质量明显下降 | 理论上只要 $d_c \geq$ 有效秩就不丢信息 |

> **一句话总结**：GQA 是"少买几个硬盘"（简单粗暴，效果够好）；MLA 是"发明了一种新的压缩算法"（技术精妙，压缩比更高）。在当前的工业实践中，**GQA 是绝对的主流**（几乎所有开源大模型都在用），而 **MLA 是下一代技术的方向**（DeepSeek 系列已验证其效果）。

---

## 九、各方案 KV Cache 显存对比总表

以 $d = 4096, n_h = 32, d_h = 128, n_g = 8$ 为例，FP16 精度（2 bytes/param），单层单 token：

| 方案 | KV Cache 维度 | 每层每 token 字节 | 相对 MHA | 代表模型 |
|------|-------------|-----------------|---------|---------|
| **MHA** | $2 \times n_h \times d_h = 8192$ | $16{,}384$ B | 1.0× | GPT-3, LLaMA-1 |
| **GQA** | $2 \times n_g \times d_h = 2048$ | $4{,}096$ B | **1/4** | LLaMA-2/3, Mistral |
| **MQA** | $2 \times 1 \times d_h = 256$ | $512$ B | **1/32** | PaLM, Gemini |
| **MLA** | $d_c + d_r = 576$ | $1{,}152$ B | **~1/14** | DeepSeek-V2/V3 |

---

## 参考文献

1. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
2. Shazeer, N. *Multi-Query Attention Is All You Need.* arXiv:1911.02150, 2019.
3. Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023.
4. Touvron, H. et al. *LLaMA 2: Open Foundation and Fine-Tuned Chat Models.* arXiv:2307.09288, 2023.
5. DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* arXiv:2405.04434, 2024.
6. Su, J. et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing, 2024.

---

## 附：仓库最小实现对照

本文在仓库中的最小实现对应 [../../src/attention/mha_gqa.py](../../src/attention/mha_gqa.py)。如果你想把本页的推导和实际代码逐行对齐，建议再配合 [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md) 一起看。

### 1. 投影与分头

```python
q = x @ w_q
k = x @ w_k
v = x @ w_v

qh = _split_heads(q, num_heads)
kh = _split_heads(k, num_kv_heads)
vh = _split_heads(v, num_kv_heads)
```

这对应前文的：

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

以及从 `[B, T, D]` 到 `[B, H, T, d_h]` 的 reshape / transpose。

### 2. GQA 的分组共享

```python
group_size = num_heads // num_kv_heads
if group_size > 1:
    kh = np.repeat(kh, repeats=group_size, axis=1)
    vh = np.repeat(vh, repeats=group_size, axis=1)
```

它正对应：

$$
G = \frac{n_h}{n_g},
\qquad
K'_h = K_{\lceil h / G \rceil},
\qquad
V'_h = V_{\lceil h / G \rceil}
$$

也就是“逻辑上广播 KV，物理上只存更少的 KV 头”。

### 3. 计算复杂度不变，但带宽压力变小

```python
out = _scaled_dot_product_attention(qh, kh, vh, mask=mask)
out = _merge_heads(out)
return out @ w_o
```

GQA 没有改变 attention 的数学定义，改变的是 KV Cache 的物理尺寸，因此训练 / prefill 的主公式基本不变，decode 阶段的 HBM 读取量会按 $n_g / n_h$ 缩小。这也是为什么工程收益主要出现在长上下文 decode。
