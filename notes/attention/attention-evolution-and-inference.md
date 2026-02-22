# 大模型注意力机制演进与现代端到端推理流程变革

大语言模型（LLM）的算力和内存瓶颈主要集中在**注意力机制（Attention）**和**自回归解码（Auto-regressive Decode）**两个方面。本文将梳理各家大模型注意力机制的发展脉络（**含完整公式推导与 PyTorch 代码**），并详细解析现代工业级推理引擎（如 vLLM, TensorRT-LLM, SGLang 等）对标准端到端推理流程所做的革命性改造。

---

## 第一部分：各家大模型注意力机制的发展脉络

注意力机制的演进主线，是一部**"与显存和显存带宽（Memory Bandwidth）作斗争"**的历史。在解码阶段，模型每次计算都需要读取历史上所有的 KV Cache，这导致推理速度被显存读取速度死死卡住（Memory Bound）。

> **统一符号表**（以 LLaMA-3 8B 为例）：
> | 符号 | 含义 | LLaMA-3 8B 取值 |
> |------|------|----------------|
> | $d$ | 模型隐藏维度 | 4096 |
> | $H$ | Query 头数 | 32 |
> | $H_{kv}$ | KV 头数（GQA 时 $< H$） | 8 |
> | $d_h$ | 每头维度 $= d / H$ | 128 |
> | $G$ | 每组 Q 头数 $= H / H_{kv}$ | 4 |
> | $N$ | 序列长度 | — |
> | $L$ | 层数 | 32 |

---

### 1. 经典原点：MHA (Multi-Head Attention)

*   **代表模型**：Transformer 原作, GPT-3, LLaMA-1, OPT, Bloom
*   **机制原理**：每个 Query 头都有自己独立的、对应的 Key 和 Value 头。$H_{kv} = H$。

#### 1.1 MHA 公式推导

**第一步：线性投影**——将输入 $X \in \mathbb{R}^{N \times d}$ 投影到 $H$ 个独立的 QKV 子空间：

$$Q_h = X \, W^Q_h, \quad K_h = X \, W^K_h, \quad V_h = X \, W^V_h \qquad h = 1, \ldots, H$$

其中 $W^Q_h, W^K_h, W^V_h \in \mathbb{R}^{d \times d_h}$。注意：**每个头都有独立的 K 和 V 投影矩阵**。

**第二步：Scaled Dot-Product Attention**：

$$A_h = \text{softmax}\!\left(\frac{Q_h \, K_h^\top}{\sqrt{d_h}} + M \right) \quad \in \mathbb{R}^{N \times N}$$

其中 $M$ 是因果掩码矩阵（$M_{ij} = 0$ if $j \le i$, else $-\infty$）。

**第三步：加权聚合 + 拼接输出**：

$$\text{head}_h = A_h \, V_h \quad \in \mathbb{R}^{N \times d_h}$$

$$\text{MHA}(X) = [\text{head}_1 \mid \text{head}_2 \mid \cdots \mid \text{head}_H] \, W^O \quad \in \mathbb{R}^{N \times d}$$

**KV Cache 大小（每层每 token）**：

$$\text{KV}_{\text{MHA}} = 2 \times H \times d_h = 2d \quad \text{(K 和 V 各一份)}$$

#### 1.2 MHA PyTorch 核心代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """标准 Multi-Head Attention (MHA)"""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 每个头有独立的 Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # [d, H * d_h]
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # [d, H * d_h]
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # [d, H * d_h]
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # ① 线性投影 → [B, N, H, d_h] → [B, H, N, d_h]
        q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # ② Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 因果掩码
        causal_mask = torch.triu(
            torch.full((N, N), float('-inf'), device=x.device), diagonal=1
        )
        scores = scores + causal_mask

        attn = F.softmax(scores, dim=-1)  # [B, H, N, N]

        # ③ 加权聚合 + 拼接
        out = torch.matmul(attn, v)  # [B, H, N, d_h]
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.W_o(out)
```

---

### 2. 极端压缩：MQA (Multi-Query Attention)

*   **代表模型**：PaLM, Falcon, StarCoder
*   **机制原理**：$H$ 个 Query 头**共享唯一 1 个** Key 头和 1 个 Value 头。即 $H_{kv} = 1$。

#### 2.1 MQA 公式推导

**与 MHA 唯一的区别**：K 和 V 的投影矩阵<strong>只有一份</strong>，不再按头区分：

$$Q_h = X \, W^Q_h \quad (h = 1,\ldots,H), \qquad K = X \, W^K, \qquad V = X \, W^V$$

其中 $W^K, W^V \in \mathbb{R}^{d \times d_h}$——注意没有下标 $h$。

**注意力计算**（所有头共用同一个 K 和 V）：

$$A_h = \text{softmax}\!\left(\frac{Q_h \, K^\top}{\sqrt{d_h}} + M\right), \qquad \text{head}_h = A_h \, V$$

**KV Cache 大小（每层每 token）**：

$$\text{KV}_{\text{MQA}} = 2 \times 1 \times d_h = 2d_h = \frac{2d}{H}$$

对比 MHA 压缩了 $H$ 倍（LLaMA-3 的话就是 **32 倍**）。

#### 2.2 MQA PyTorch 核心代码

```python
class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA) — 所有 Q 头共享 1 个 KV 头"""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)     # H 个 Q 头
        self.W_k = nn.Linear(d_model, self.d_head, bias=False)  # ← 只有 1 个 K 头！
        self.W_v = nn.Linear(d_model, self.d_head, bias=False)  # ← 只有 1 个 V 头！
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # k, v: [B, N, 1, d_h] → 广播到 [B, H, N, d_h]
        k = self.W_k(x).unsqueeze(1)  # [B, 1, N, d_h]
        v = self.W_v(x).unsqueeze(1)  # [B, 1, N, d_h]
        # PyTorch 广播: [B, H, N, d_h] × [B, 1, d_h, N] 自动扩展

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.triu(
            torch.full((N, N), float('-inf'), device=x.device), diagonal=1
        )
        scores = scores + causal_mask
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # 广播: [B, H, N, N] × [B, 1, N, d_h]
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.W_o(out)
```

> **💡 关键区别只有一处**：`W_k` 和 `W_v` 的输出维度从 `d_model`（= $H \times d_h$）变成了 `d_head`（= $1 \times d_h$）。KV Cache 体积直接缩小到原来的 $1/H$。

---

### 3. 甜点均衡：GQA (Grouped-Query Attention)

*   **代表模型**：LLaMA-2/3, Mistral, Qwen-1.5/2, Gemma （**目前行业绝对主流**）
*   **机制原理**：将 $H$ 个 Query 头分成 $H_{kv}$ 组，每组 $G = H / H_{kv}$ 个 Q 头共享一个 KV 头。

#### 3.1 GQA 公式推导

**投影**：

$$Q_h = X \, W^Q_h \quad (h = 1,\ldots,H)$$
$$K_g = X \, W^K_g, \qquad V_g = X \, W^V_g \quad (g = 1,\ldots,H_{kv})$$

其中 $W^K_g, W^V_g \in \mathbb{R}^{d \times d_h}$。

**分组共享规则**（LLaMA-3 8B：$H=32, H_{kv}=8, G=4$）：

$$\text{对于第 } h \text{ 个 Q 头，它使用第 } g = \lfloor h / G \rfloor \text{ 组的 KV}$$

即 Q 头 0,1,2,3 共享 KV 组 0；Q 头 4,5,6,7 共享 KV 组 1；依此类推。

**注意力**：

$$A_h = \text{softmax}\!\left(\frac{Q_h \, K_{g(h)}^\top}{\sqrt{d_h}} + M\right), \qquad \text{head}_h = A_h \, V_{g(h)}$$

**KV Cache 大小（每层每 token）**：

$$\text{KV}_{\text{GQA}} = 2 \times H_{kv} \times d_h$$

LLaMA-3 8B：$2 \times 8 \times 128 = 2048$ 个参数 → 对比 MHA 的 $2 \times 32 \times 128 = 8192$，压缩了 **4 倍**。

#### 3.2 GQA PyTorch 核心代码

```python
class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) — 当前行业主流"""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads        # Q 头数 (32)
        self.n_kv_heads = n_kv_heads  # KV 头数 (8)
        self.n_groups = n_heads // n_kv_heads  # 每组 Q 头数 (4)
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)  # ← 只有 8 个 KV 头
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
        # 每个 KV 头被复制 4 次，分别供给 4 个 Q 头使用
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

> **💡 关键洞察**：GQA 在**计算时**把 8 个 KV 头 `repeat_interleave` 扩展成 32 个（零成本广播），但在**存储 KV Cache 时**只需要保存 8 份。这就是它既快又不丢精度的秘密——计算时每个 Q 头都"看到了完整信息"，存储时只付 1/4 的代价。

---

### 4. 潜空间压缩：MLA (Multi-head Latent Attention)

*   **代表模型**：DeepSeek-V2, DeepSeek-V3, DeepSeek-R1
*   **机制原理**：不缓存高维 K/V，而是将历史信息压缩到一个极低维的**隐向量（Latent）**中。推理时从隐向量实时还原出 K/V。

#### 4.1 MLA 公式推导

MLA 的核心想法是：与其直接存 K 和 V（维度很高），不如先"压缩"再"解压"。

**第一步：下投影（Compress）—— 将输入压缩成低维隐向量**

$$c_t = X_t \, W^{DKV} \quad \in \mathbb{R}^{d_c}$$

其中 $W^{DKV} \in \mathbb{R}^{d \times d_c}$，$d_c \ll d$。
例如 DeepSeek-V2 中 $d = 5120$, $d_c = 512$，压缩比 10:1。

> **这个 $c_t$ 就是唯一需要缓存的东西！** KV Cache 从存 K + V（$2 \times d$）变为只存 $c_t$（$d_c$）。

**第二步：上投影（Decompress）—— 从隐向量还原出 K 和 V**

$$K_h = c_t \, W^{UK}_h, \qquad V_h = c_t \, W^{UV}_h$$

其中 $W^{UK}_h, W^{UV}_h \in \mathbb{R}^{d_c \times d_h}$。这一步在**每次需要计算注意力时实时完成**（不缓存）。

**第三步：RoPE 位置编码的解耦**

传统注意力中，RoPE 直接加在 K 上：$K_{\text{rope}} = \text{RoPE}(K)$。

但 MLA 的 K 是从隐向量 $c_t$ 还原出来的。如果把 RoPE 直接加到 $c_t$ 上，位置信息就会"污染"隐向量，导致无法在不同位置之间正确共享。

**DeepSeek 的天才解法——解耦 RoPE**：

$$K_h^{\text{final}} = [K_h^{\text{nope}} \mid K_h^{\text{rope}}]$$

- $K_h^{\text{nope}} = c_t \, W^{UK}_h$：内容信息（不加位置编码），从隐向量还原
- $K_h^{\text{rope}} = \text{RoPE}(X_t \, W^{KR})$：位置信息（加 RoPE），从原始输入直接投影

两部分 concat 后一起参与注意力计算。

> **💡 理解要点**：$K^{\text{rope}}$ 很小（比如 64 维），也要缓存。但总的 KV Cache = $d_c + d_r$ 依然远小于 MHA 的 $2d$。

**KV Cache 大小（每层每 token）**：

$$\text{KV}_{\text{MLA}} = d_c + d_r \quad \text{（隐向量 + RoPE Key）}$$

DeepSeek-V2 实际值：$512 + 64 = 576$，对比 MHA 的 $2 \times 5120 = 10240$，压缩了约 **18 倍**。

#### 4.2 MLA PyTorch 核心代码

```python
class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) — DeepSeek-V2/V3 的核心
    KV Cache 只需存 [compressed_kv, rope_key]，极致压缩
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_compress: int,   # 隐向量维度 d_c (如 512)
        d_rope: int,       # RoPE key 维度 d_r (如 64)
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_compress = d_compress
        self.d_rope = d_rope

        # Q 投影 (标准)
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        # ★ 下投影：x → 低维隐向量 c (这个 c 就是 KV Cache 的主体)
        self.W_dkv = nn.Linear(d_model, d_compress, bias=False)

        # ★ 上投影：c → 高维 K, V (推理时实时计算，不缓存)
        self.W_uk = nn.Linear(d_compress, d_model, bias=False)
        self.W_uv = nn.Linear(d_compress, d_model, bias=False)

        # ★ 解耦 RoPE：单独投影出一个小维度的 key (需要额外缓存)
        self.W_kr = nn.Linear(d_model, d_rope, bias=False)

        # 输出
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """简化版 RoPE（实际实现用 cos/sin 旋转）"""
        # 此处省略旋转细节，实际用 freqs_cis 做复数旋转
        return x  # placeholder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # ──────────── Q 路径（标准） ────────────
        q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # ──────────── KV 路径（MLA 核心） ────────────
        # Step 1: 下投影 → 低维隐向量 (★ 这是 KV Cache 要存的东西)
        c = self.W_dkv(x)  # [B, N, d_compress]

        # Step 2: 上投影 → 高维 K, V (实时还原，不缓存)
        k_nope = self.W_uk(c).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_uv(c).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # Step 3: 解耦 RoPE → 单独投影出小维度 key 并加旋转位置编码
        k_rope = self.W_kr(x)         # [B, N, d_rope]
        k_rope = self.apply_rope(k_rope)
        # 广播到所有头: [B, N, d_rope] → [B, H, N, d_rope]
        k_rope = k_rope.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        # Step 4: 拼接 → 完整 K = [K_nope | K_rope]
        # 注意：Q 也要相应地把末尾 d_rope 维加上 RoPE
        # (此处简化，实际 Q 也做了同样的 nope/rope 拆分)
        k = torch.cat([k_nope, k_rope], dim=-1)  # [B, H, N, d_head + d_rope]

        # ──────────── Attention 计算 ────────────
        # (此处简化，实际 Q 也需要 pad 到相同维度)
        scores = torch.matmul(q, k_nope.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.triu(
            torch.full((N, N), float('-inf'), device=x.device), diagonal=1
        )
        scores = scores + causal_mask
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.W_o(out)

    def get_kv_cache(self, x: torch.Tensor):
        """推理时：只返回需要缓存的内容"""
        c = self.W_dkv(x)              # [B, N, d_compress] ← 主体
        k_rope = self.apply_rope(self.W_kr(x))  # [B, N, d_rope]   ← 位置
        return c, k_rope  # ← 这就是全部的 KV Cache！
```

---

### 5. 架构革命：线性注意力与无 KV Cache 架构

*   **代表模型**：Mamba (SSM), RWKV (RNN), Jamba, GLA
*   **机制原理**：传统 Attention 的复杂度是 $O(N^2)$，且 KV Cache 随长度增长。这类架构通过状态空间方程或门控线性注意力，将所有历史信息压缩为一个**固定大小的隐状态（Hidden State）**。
*   **现状**：推理阶段**彻底消灭了 KV Cache**，显存占用恒定。但目前在"大海捞针"和 In-Context Learning 等任务上，还未能完全超越标准 Transformer。

---

### 各方案 KV Cache 显存对比表

> 以 $d = 4096, H = 32, d_h = 128, H_{kv} = 8$ 为例，FP16 精度（2 bytes/param），单层单 token：

| 方案 | KV Cache 维度 | 每层每 token 字节 | 相对 MHA | 代表模型 |
|------|-------------|-----------------|---------|---------|
| **MHA** | $2 \times H \times d_h = 2d$ | $2 \times 4096 \times 2 = 16384$ B | 1.0× | GPT-3 |
| **MQA** | $2 \times 1 \times d_h$ | $2 \times 128 \times 2 = 512$ B | **1/32** | PaLM |
| **GQA** | $2 \times H_{kv} \times d_h$ | $2 \times 8 \times 128 \times 2 = 4096$ B | **1/4** | LLaMA-3 |
| **MLA** | $d_c + d_r$ | $(512 + 64) \times 2 = 1152$ B | **~1/14** | DeepSeek-V2 |
| **SSM** | 固定 state | **恒定**（与序列长度无关） | — | Mamba |

> **💡 关键洞察**：32K 上下文、32 层、Batch = 64 的场景下（LLaMA-3 8B GQA）：
> KV Cache 总占用 = $64 \times 32000 \times 32 \times 4096 \times 2 \approx \mathbf{16.8 \text{ GB}}$
> 这几乎占满了一张 A100 40GB 的一半显存！而如果用 MHA，则需要 $\mathbf{67.1 \text{ GB}}$——直接爆掉。

---

## 第二部分：现代端到端推理流程（流程发生了什么变化？）

你列出的标准流程在**逻辑上**依然是成立的，这是所有文本生成任务的基石：
1. **Prompt** -> 2. **Tokenizer** -> 3. **Prefill (算全段 KV)** -> 4. **Decode (自回归)** -> 5. **Stop** -> 6. **Detokenize**

然而，在现代真实的工业生产环境（如 vLLM, TensorRT-LLM, SGLang 等推理引擎）中，**物理执行层面已经发生了翻天覆地的变化**。为了极致压榨 GPU 性能，流程中的每一步都被高度魔改了。

### 变革 1：内存管理的革命 —— PagedAttention
*   **标准流程的痛点**：预先为每个句子的 KV Cache 分配一块连续的显存（不管它最终生成多长）。这会导致大量的显存碎片和预留浪费，显存利用率往往不到 30%。
*   **现代流程的变化**：引入 **PagedAttention**。推理引擎像操作系统管理虚拟内存一样，将 KV Cache 切分为固定大小的 Block（如每块存 16 个 token）。在 Decode 阶段，生成一个 token 就按需分配一块显存，物理内存不再连续。**这使得系统能同时处理的 Batch Size 翻了数倍，彻底解决了 OOM（内存溢出）问题。**

### 变革 2：调度机制的革命 —— Continuous Batching (In-flight Batching)
*   **标准流程的痛点**：静态批处理（Static Batching）。比如把 4 个请求打包成一个 Batch，必须等这 4 个请求全部走到 Stop 阶段，才能接下一批。如果其中 3 个很短，1 个很长，GPU 会为了等那个长的而闲置大半算力。
*   **现代流程的变化**：**连续批处理（Continuous Batching）**。引擎以单步 Iteration 为单位进行调度。如果池子里有一个请求触发了 EOS（Stop），它立刻被踢出池子，调度器**瞬间**将队列中等待的新请求的 Prefill 塞入当前 Batch。在 GPU 里，**有的序列在做 Prefill，有的序列在做 Decode，它们在同一个 Batch 里混合计算。**

### 变革 3：Prefill 阶段的革命 —— Chunked Prefill & Prefix Caching
*   **标准流程的痛点**：如果用户输入了一个 100K token 的超长 Prompt，Prefill 阶段会瞬间进行庞大的矩阵乘法，导致极高的延迟尖峰，甚至直接把显存干爆；同时，这个庞大的计算会卡住其他正在 Decode 的用户。
*   **现代流程的变化**：
    1.  **Chunked Prefill（分块预填充）**：引擎把 100K 的 Prompt 强行切碎，每次只算 4K，分多次塞进不同的 Batch 里算完。这平滑了系统的算力开销。
    2.  **Prefix Caching（前缀缓存/基数树）**：像 SGLang 这样的引擎会在显存里维护一棵前缀树。如果用户多次对话都带着相同的 System Prompt（或历史上下文），引擎会直接从内存中"命中"并复用对应的 KV Cache，**直接跳过 Prefill 阶段**，首字延迟（TTFT）降至 0。

### 变革 4：Decode 阶段的革命 —— Speculative Decoding (投机解码)
*   **标准流程的痛点**：Decode 是严格的**自回归（串行）**计算：吐出第 1 个字，拿它去查 KV Cache；吐出第 2 个字，再去查。因为每次只计算一个 token，GPU 强大的并行算力（矩阵乘法）根本用不满，沦为了"内存搬运工"。
*   **现代流程的变化**：**投机解码**打破了自回归的步长限制。
    *   引擎会额外挂载一个极其轻量级的"草稿模型"（或者用大模型的浅层）。
    *   草稿模型飞速跑 4 步，"猜"出 4 个候选 token：`[A, B, C, D]`。
    *   大模型（目标模型）**把这 4 个 token 当作一句话（就像 Prefill 一样）**，一次性并行验证它们是否符合大模型的概率分布。
    *   如果验证通过，大模型相当于**只做了一次矩阵运算，就往前走了 4 步**，解码速度提升 2~3 倍。

### 变革 5：输出结构的革命 —— Structured Decoding (结构化解码)
*   **标准流程的痛点**：模型自由生成文本，如果是调用 API（如 JSON 输出），很容易因为模型幻觉漏掉一个括号导致解析失败。
*   **现代流程的变化**：在 Decode 产生 Logits 到采样输出这个环节之间，插入了**状态机约束（如 XGrammar / Outlines）**。引擎会维护一个合法的语法树，当模型生成的 Logits 中，那些会导致 JSON 语法错误的 token，其概率会被引擎**强行置为 $-\infty$（负无穷）**。这确保了模型生成的输出 100% 符合 JSON Schema，将 Decode 从"自由创作"变成了"戴着镣铐跳舞"。

---

## 全局因果链总结图

现代大模型之所以能支撑起每天几千万的调用量，正是因为整个流水线被极致地"流水线化"和"异步化"了：

```text
用户请求输入
  │
  ▼ [Prefix Caching 检查]
  ├── 命中历史缓存 ──跳过──┐
  └── 未命中缓存         │
                         ▼
             【Chunked Prefill阶段】
             不再是一次性算完，而是切分成小块，
             见缝插针地塞进 GPU 的算力空隙中
                         │
                         ▼
             【PagedAttention 内存池】
             像操作系统分配内存一样，申请几个 Block，
             把算好的 KV Cache 碎片化存储
                         │
                         ▼
             【Speculative Decode 阶段】 (连续批处理中)
             草稿模型飞速狂奔猜答案，
             大模型并行批改作业，每次向前走 N 步。
             期间还要受到【结构化解码】的语法树约束
                         │
                         ▼
               触发 EOS 或长度限制，资源瞬间释放
```

你的 6 步标准流程是算法工程师眼中的 Transformer，而现在的实际流程，是**系统工程师（Systems for ML）**眼中一台极致精密的**异步流水线机器**。
