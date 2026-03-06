# 传统 MHA vs DSA：全流程数学与矩阵推导对比

> **摘要**：本文以严格的数学形式，完整推导 Multi-Head Attention (MHA) 与 DeepSeek Dynamic/Native Sparse Attention (DSA) 在 Prefill 和 Decode 两个阶段的每一步计算。DSA 的核心创新在于引入一个极轻量的**闪电索引器（Lightning Indexer）**，在执行主注意力之前先以 $O(L \cdot d_{\text{idx}})$ 的超低代价全局扫描历史 Token，选出 Top-$k$ 个最相关的 Token，然后仅在这 $k$ 个 Token 上执行底层为 MLA 的注意力计算。这实现了**"序列压缩（DSA）"× "特征压缩（MLA）"**的乘法级显存节省。

---

## 核心思想对比

| | MHA (全局稠密注意力) | DSA (闪电索引 + 稀疏 MLA) |
|--|---------------------|--------------------------|
| **计算范围** | 每个 Query 必须与**全部 $L$ 个** Key/Value 做点积 | 先粗筛 Top-$k$ 个 Token，再仅在 $k$ 个上计算 |
| **复杂度** | $O(L^2 \cdot d)$ | $O(L \cdot d_{\text{idx}}) + O(k \cdot d)$ |
| **类比** | 侦探翻遍整栋图书馆的每一本书 | 先用电子目录搜索出 2048 条线索，再精读这 2048 本 |
| **底层 KV 架构** | 标准 MHA（高维 K/V） | MLA（低维潜在向量 $C^{KV}$） |
| **压缩维度** | 无压缩 | **序列维度 × 特征维度**双重压缩 |

---

## 符号约定

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $L$ | 历史序列总长度 | 128K+ |
| $d$ | 模型隐藏维度 | 5120 |
| $n_h$ | 注意力头数 | 128 |
| $d_h$ | 每头维度 $= d / n_h$ | 128 |
| $k$ | 稀疏挑选出的 Top-$k$ Token 数 | 2048（$k \ll L$） |
| $S_t$ | 当前 Token $t$ 的 Top-$k$ 历史索引集合 | $\lvert S_t \rvert = k$ |
| $n_{\text{idx}}$ | 闪电索引器的头数 | 极小（如 4） |
| $d_{\text{idx}}$ | 闪电索引器的特征维度 | 极小（如 64） |
| $I_{t,s}$ | 索引器算出的 Query $t$ 与 Key $s$ 的相关性分数 | 标量 |
| $d_c$ | MLA 潜在空间维度 | 512 |
| $d_r$ | 解耦 RoPE 维度 | 64 |
| $C^{KV}$ | MLA 的 KV 潜在向量 | $\in \mathbb{R}^{L \times d_c}$ |

---

## 一、输入定义

两种机制共享完全相同的输入：

$$X \in \mathbb{R}^{L \times d}$$

---

## 二、闪电索引器（Lightning Indexer）— DSA 的核心独创阶段

### MHA

**无此阶段。** MHA 直接对全量 $L$ 个 Token 计算注意力。

### DSA

引入一个独立的、极其轻量的索引器网络，负责快速判断"哪些历史 Token 与当前 Query 最相关"。

#### 2.1 生成低维索引 Query 和 Key

对当前 Token $t$：

$$q_t^{\text{idx}} = x_t \, W^{Q_{\text{idx}}} \quad \in \mathbb{R}^{n_{\text{idx}} \times d_{\text{idx}}}$$

对所有历史 Token $s \in \{1, \ldots, L\}$：

$$k_s^{\text{idx}} = x_s \, W^{K_{\text{idx}}} \quad \in \mathbb{R}^{n_{\text{idx}} \times d_{\text{idx}}}$$

其中 $W^{Q_{\text{idx}}} \in \mathbb{R}^{d \times (n_{\text{idx}} \cdot d_{\text{idx}})}$，$W^{K_{\text{idx}}}$ 同理。

> **关键设计**：$n_{\text{idx}}$ 和 $d_{\text{idx}}$ 都极小（如 $4 \times 64 = 256$ 维），且使用 **FP8 精度**计算——这使得索引器的参数量和计算量仅占主注意力的 $< 1\%$。

#### 2.2 计算索引分数（⭐ ReLU 替代 Softmax）

$$\boxed{I_{t,s} = \sum_{j=1}^{n_{\text{idx}}} \text{ReLU}\!\left(q_{t,j}^{\text{idx}} \cdot (k_{s,j}^{\text{idx}})^\top\right) \cdot w_{t,j}}$$

其中：
- $q_{t,j}^{\text{idx}} \in \mathbb{R}^{1 \times d_{\text{idx}}}$ 是第 $j$ 个索引头的 Query
- $k_{s,j}^{\text{idx}} \in \mathbb{R}^{1 \times d_{\text{idx}}}$ 是第 $j$ 个索引头的 Key
- $w_{t,j} \in \mathbb{R}$ 是可学习的头权重（动态 Gating）

> **为什么用 ReLU 而不用 Softmax？**
>
> | 对比 | Softmax | ReLU |
> |------|---------|------|
> | **依赖性** | 需要全局分母 $\sum_s \exp(I_{t,s})$，必须看完所有 $L$ 个 Token 才能归一化 | 每对 $(t, s)$ 独立计算，无全局依赖 |
> | **并行性** | 难以流水线化，全局归一化形成同步点 | **完美适配 GPU 并行**——每个 CUDA 线程独立处理一对 |
> | **排序友好** | 归一化后仍需排序选 Top-$k$ | 直接比较 ReLU 输出值，支持 **Partial Sort**（只需 $O(L \log k)$ 而非 $O(L \log L)$） |
> | **稀疏性** | 所有分数 $> 0$（指数函数恒正） | ReLU 天然产生 $0$ 值，**自动过滤无关 Token** |

#### 2.3 索引器的计算复杂度

对单个 Query Token $t$，索引器需要与全部 $L$ 个历史 Key 做点积：

$$\text{Indexer FLOPs} = n_{\text{idx}} \times d_{\text{idx}} \times L = 4 \times 64 \times L = 256L$$

对比主注意力的 FLOPs：

$$\text{MHA FLOPs} = n_h \times d_h \times L = 128 \times 128 \times L = 16384L$$

**索引器计算量占主注意力的比例**：$256L / 16384L = 1.56\%$

> 再加上 FP8 精度（相比 FP16 吞吐翻倍），索引器的实际耗时 $< 1\%$，几乎"免费"。

---

## 三、Token 稀疏路由

### MHA

**无此阶段。** 强制与全量 $L$ 个历史 Token 交互——没有选择权。

### DSA

根据索引分数 $I_{t,s}$，选出得分最高的 $k$ 个 Token：

$$\boxed{S_t = \left\{ s \;\middle|\; I_{t,s} \in \text{Top-}k\!\left(\{I_{t,1}, I_{t,2}, \ldots, I_{t,L}\}\right) \right\}, \qquad |S_t| = k}$$

> **动态性**：对于每一个不同的 Query Token $t$，$S_t$ 是不同的——这就是"Dynamic"的含义。不同的问题关注不同的上下文！

#### 3.1 路由的工程实现

在 GPU 上，Top-$k$ 选择可以通过 **Partial Sort**（如 `torch.topk`）高效实现：

```python
# indexer_scores: [B, L]  — 每个历史 Token 的相关性分数
# k: int = 2048
top_k_scores, top_k_indices = torch.topk(indexer_scores, k=k, dim=-1)
# top_k_indices: [B, k]  — 被选中的 Token 位置索引
```

复杂度：$O(L \log k)$（而非全排序的 $O(L \log L)$）。

---

## 四、主注意力 K、V 投影

### MHA：每个头独立生成完整 K、V

$$K_i = X \, W_i^K \quad \in \mathbb{R}^{L \times d_h}, \qquad V_i = X \, W_i^V \quad \in \mathbb{R}^{L \times d_h}$$

其中 $W_i^K, W_i^V \in \mathbb{R}^{d \times d_h}$，$i = 1, \ldots, n_h$。

### DSA：继承 MLA 的极致特征压缩

$$\boxed{C^{KV} = X \, W^{DKV} \quad \in \mathbb{R}^{L \times d_c}}$$

其中 $W^{DKV} \in \mathbb{R}^{d \times d_c}$。这一步与标准 MLA 完全相同——所有 $n_h$ 个头的 KV 信息被联合压缩到 $d_c = 512$ 维的共享潜在向量中。

> **DSA 的底层不是 MHA，而是 MLA。** 这是"双剑合璧"的第一剑——**特征维度压缩**。

---

## 五、Attention 计算（Prefill 阶段）

### MHA：全量稠密计算 $O(L^2)$

$$S_{t,s} = \text{softmax}\!\left(\frac{Q_t \, K_s^\top}{\sqrt{d_h}}\right), \qquad s = 1, \ldots, L$$

$$O_t = \sum_{s=1}^{L} S_{t,s} \, V_s$$

**单个 Token 的 Attention FLOPs**：$O(L \cdot d_h \cdot n_h) = O(L \cdot d)$

**整个序列**：$O(L^2 \cdot d)$

### DSA：仅在 Top-$k$ 集合上执行 MLA 计算

首先，从 $C^{KV}$ 中解压出被选中 Token 的 K 和 V（或通过矩阵吸收直接在潜在空间计算）：

$$S_{t,s} = \text{softmax}\!\left(\frac{\hat{Q}_t \, \hat{K}_s^\top}{\sqrt{d_h + d_r}}\right), \qquad \color{red}{s \in S_t \text{ only}}$$

$$O_t = \sum_{\color{red}{s \in S_t}} S_{t,s} \, V_s^C$$

**单个 Token 的 Attention FLOPs**：$O(k \cdot d_h \cdot n_h) = O(k \cdot d)$

**加上索引器**：$O(L \cdot d_{\text{idx}} \cdot n_{\text{idx}}) + O(k \cdot d)$

**整个序列**：$O(L \cdot k \cdot d)$，其中 $k \ll L$

### 5.1 复杂度对比

| | MHA | DSA | 加速比 |
|--|-----|-----|--------|
| 单 Token Attention | $O(L \cdot d)$ | $O(L \cdot d_{\text{idx}}) + O(k \cdot d)$ | — |
| 整个序列 Prefill | $O(L^2 \cdot d)$ | $O(L^2 \cdot d_{\text{idx}}) + O(L \cdot k \cdot d)$ | — |
| $L = 128\text{K}, k = 2\text{K}$ | $\propto L^2 = 1.64 \times 10^{10}$ | $\propto L \cdot k = 2.62 \times 10^8$ | **~62×** |

> **注意**：DSA 在 Prefill 阶段的索引器项 $O(L^2 \cdot d_{\text{idx}})$ 看似依然是 $O(L^2)$，但由于 $d_{\text{idx}} \ll d$（如 $64 \ll 5120$，差 80 倍），且用 FP8 精度计算，实际耗时远小于主注意力。真正的"重活"（$O(L \cdot k \cdot d)$）被压缩了 $L/k = 64$ 倍。

---

## 六、KV Cache 存储阶段

### MHA

缓存所有头的完整 K 和 V：

$$\text{Cache}_{\text{MHA}} = \left\{\; K_i,\; V_i \;\right\}_{i=1}^{n_h}$$

**每层每 token 显存**：$2 \times n_h \times d_h = 2d$（如 $2 \times 128 \times 128 = 32768$ 维）

### DSA

缓存 MLA 潜在向量 + 解耦 RoPE Key + 索引器 Key：

$$\boxed{\text{Cache}_{\text{DSA}} = \left\{\; \underbrace{C^{KV}}_{\text{MLA 潜在向量}},\; \underbrace{K^{\text{RoPE}}}_{\text{解耦位置 Key}},\; \underbrace{k^{\text{idx}}}_{\text{索引器 Key}} \;\right\}}$$

**每层每 token 显存**：

$$d_c + d_r + n_{\text{idx}} \times d_{\text{idx}}$$

$$= 512 + 64 + 4 \times 64 = 832 \;\text{维}$$

> **对比 MHA 的 32768 维**：压缩比 $32768 / 832 \approx 39.4\times$。其中索引器 Key 仅占 $256 / 832 = 30.8\%$ 的额外开销。

### 6.1 Cache 显存对比表

| 组件 | MHA | DSA |
|------|-----|-----|
| K Cache | $n_h \times d_h \times L$ | — |
| V Cache | $n_h \times d_h \times L$ | — |
| $C^{KV}$ | — | $d_c \times L$ |
| $K^{\text{RoPE}}$ | — | $d_r \times L$ |
| $k^{\text{idx}}$ | — | $n_{\text{idx}} \times d_{\text{idx}} \times L$ |
| **总计（每 token）** | **32768 维** | **832 维** |
| **FP16 字节** | **65,536 B** | **1,664 B** |

---

## 七、推理 / Decode 阶段

Decode 是 DSA 解决超长文本推理的核心环节。每一步仅输入一个新 token $x_t \in \mathbb{R}^{1 \times d}$。

### 7.1 索引器全局粗筛

#### MHA

**跳过**——直接进入全量 Attention 计算。

#### DSA

使用 FP8 精度，让当前新词的 $q_t^{\text{idx}}$ 迅速扫过 Cache 中的**全量** $k^{\text{idx}}$：

$$I_{t,s} = \sum_{j=1}^{n_{\text{idx}}} \text{ReLU}\!\left(q_{t,j}^{\text{idx}} \cdot (k_{s,j}^{\text{idx}})^\top\right) \cdot w_{t,j}, \qquad s = 1, \ldots, L$$

$$S_t = \text{Top-}k\!\left(\{I_{t,1}, \ldots, I_{t,L}\}\right)$$

**粗筛 FLOPs**：$n_{\text{idx}} \times d_{\text{idx}} \times L = 256L$

**粗筛读取量**（FP8）：$n_{\text{idx}} \times d_{\text{idx}} \times L \times 1\;\text{byte} = 256L\;\text{bytes}$

> 以 $L = 128\text{K}$ 为例：$256 \times 131072 = 33.5\;\text{MB}$（FP8），而 MHA 读取 KV Cache 需要 $32768 \times 131072 \times 2 = 8\;\text{GB}$（FP16）——**粗筛仅用 MHA Cache 读取量的 0.4%！**

### 7.2 Cache 精确拉取（Gather）— ⭐ 双剑合璧的关键

#### MHA

从显存全量读取庞大的历史 KV：

$$\hat{K}_i^{\text{cached}} \in \mathbb{R}^{L \times d_h}, \qquad V_i^{\text{cached}} \in \mathbb{R}^{L \times d_h}$$

**总读取量**：$2 \times n_h \times d_h \times L \times 2\;\text{bytes} = 2 \times 32768 \times L$

#### DSA

仅从显存中拉取**被选中的 Top-$k$ 个 Token** 的潜在向量 Cache：

$$C^{KV}[S_t] \in \mathbb{R}^{k \times d_c}$$

$$K^{\text{RoPE}}[S_t] \in \mathbb{R}^{k \times d_r}$$

**总读取量**：$(d_c + d_r) \times k \times 2\;\text{bytes} = 576 \times k \times 2$

### 7.3 双剑合璧：序列压缩 × 特征压缩

| 维度 | MHA | DSA (MLA + Sparse) | 压缩比 |
|------|-----|---------------------|-------|
| **序列维度** | $L$（全量） | $k$（稀疏选择） | $L / k$ |
| **特征维度** | $2 \times n_h \times d_h = 32768$ | $d_c + d_r = 576$ | $56.9\times$ |
| **总读取量** | $32768 \times L$ | $576 \times k$ | $\frac{32768 \times L}{576 \times k}$ |

以 $L = 128\text{K}, k = 2048$ 为例：

$$\text{MHA 读取量} = 32768 \times 131072 \times 2 \;\text{B} \approx 8.0 \;\text{GB}$$

$$\text{DSA 读取量} = 576 \times 2048 \times 2 \;\text{B} \approx 2.3 \;\text{MB}$$

$$\boxed{\text{压缩比} = \frac{8.0 \;\text{GB}}{2.3 \;\text{MB}} \approx 3{,}640\times}$$

> **这就是"乘法级别的指数级下降"**：序列维度压缩 $64\times$（$128K / 2K$）$\times$ 特征维度压缩 $57\times$（MLA）$= 3640\times$。一次 Decode 步骤的 Cache 读取量从 **8 GB 骤降至 2.3 MB**。

### 7.4 矩阵吸收与稀疏 Attention

#### MHA

在全量长度 $L$ 上计算：

$$\text{Score}_{t} = q_t \cdot (\hat{K}^{\text{cached}})^\top \quad \in \mathbb{R}^{1 \times L}$$

$$o_t = \text{softmax}(\text{Score}_t) \cdot V^{\text{cached}} \quad \in \mathbb{R}^{1 \times d_h}$$

#### DSA

继承 MLA 的矩阵吸收技术，且仅在长度 $k$ 上计算：

**Q 吸收**（将 $W^{UK}$ 融合进 Query）：

$$\tilde{q}_t^C = q_t^C \cdot (W_i^{UK})^\top \quad \in \mathbb{R}^{1 \times d_c}$$

**在潜在空间做稀疏点积**：

$$\text{Score}_t = \tilde{q}_t^C \cdot (C^{KV}[S_t])^\top + q_t^R \cdot (K^{\text{RoPE}}[S_t])^\top \quad \in \mathbb{R}^{1 \times k}$$

**V 吸收**：

$$u_t = \text{softmax}(\text{Score}_t) \cdot C^{KV}[S_t] \quad \in \mathbb{R}^{1 \times d_c}$$

**O 吸收**：

$$o_t = u_t \cdot W_i^{UV\_O} \quad \in \mathbb{R}^{1 \times d}$$

> 整个 Decode 过程中，模型只需要读取 $C^{KV}[S_t] \in \mathbb{R}^{k \times d_c}$——这是一个 $2048 \times 512$ 的小矩阵，约 2 MB。

---

## 八、单步 Decode 复杂度对比

| 阶段 | MHA | DSA |
|------|-----|-----|
| 索引器粗筛 | — | $O(L \cdot d_{\text{idx}} \cdot n_{\text{idx}})$（FP8，极小） |
| Top-$k$ 选择 | — | $O(L \log k)$ |
| Cache 读取 | $O(L \cdot n_h \cdot d_h)$ | $O(k \cdot d_c)$ |
| Attention 计算 | $O(L \cdot d)$ | $O(k \cdot d)$ |
| **总计** | $O(L \cdot d)$ | $O(L \cdot d_{\text{idx}}) + O(k \cdot d)$ |
| **瓶颈类型** | Memory Bandwidth Bound | **索引器 Bound**（极小）+ Compute Bound |

### 8.1 随序列长度的扩展性

| $L$ | MHA Decode 延迟 | DSA Decode 延迟 | 加速比 |
|-----|----------------|-----------------|--------|
| 4K | 基准 | ~基准（$k$ 可能接近 $L$） | ~1× |
| 32K | 8× 基准 | 索引器 8×，主注意力不变 | ~4× |
| 128K | 32× 基准 | 索引器 32×，主注意力不变 | **~16×** |
| 1M | 256× 基准 | 索引器 256×，主注意力不变 | **~100×** |

> **关键洞察**：MHA 的 Decode 延迟**随序列长度线性增长**（因为必须读取全量 Cache）。而 DSA 的主注意力计算**恒定**（$k$ 是常数），只有索引器的 FP8 粗筛随 $L$ 线性增长——但这部分的代价极小。因此 DSA 在**超长文本（128K+）场景下呈现断崖式加速**。

---

## 九、DSA 的三大工程奇迹

### 9.1 分离"寻找"与"计算"（Two-Stage Pipeline）

传统注意力是"边计算边寻找"——Softmax 在全量 $L$ 上既做归一化又做加权求和，算力与内存 IO 高度耦合。

DSA 将其拆解为两个松耦合阶段：

```text
Stage 1: Lightning Indexer  ──→  WHERE (在哪里)
          极轻量 / FP8 / ReLU / 独立并行

Stage 2: MLA Attention      ──→  WHAT (算什么)
          厚重精确 / FP16 / Softmax / 仅在 Top-k 上
```

> 这种分离的好处：Stage 1 可以在一个 GPU Stream 上异步执行，与 Stage 2 形成流水线，进一步隐藏延迟。

### 9.2 ReLU 替代 Softmax 带来的极致吞吐

在索引器中使用 ReLU 而非 Softmax 的三大优势：

1. **无全局依赖**：每对 $(t, s)$ 的分数独立计算，不需要等全局分母
2. **天然稀疏**：负值直接归零，自动过滤大量无关 Token
3. **Partial Sort 友好**：只需找出最大的 $k$ 个值，无需全局排序

数学上：

$$\text{Softmax}: \quad p_{t,s} = \frac{\exp(z_{t,s})}{\sum_{s'=1}^{L} \exp(z_{t,s'})} \quad \text{← 全局归一化，$O(L)$ 依赖}$$

$$\text{ReLU}: \quad I_{t,s} = \max(0, z_{t,s}) \quad \text{← 逐元素，完全独立}$$

### 9.3 "特征压缩" × "序列压缩"的双剑合璧

| 压缩维度 | 技术 | 压缩比 | 数学表达 |
|----------|------|--------|---------|
| **特征维度** | MLA（潜在空间） | ~57× | $C^{KV} \in \mathbb{R}^{d_c}$ vs $K, V \in \mathbb{R}^{n_h d_h}$ |
| **序列维度** | DSA（Top-$k$ 路由） | $L / k$（如 64×） | $S_t$ 仅选 $k$ 个 Token |
| **总计** | DSA + MLA | **~3640×** | $576k$ vs $32768L$ |

> **类比**：MLA 是"把每本书从 500 页压缩成 8 页摘要"；DSA 是"从 10 万本书中挑出 2000 本最相关的"。双剑合璧后，你从 "阅读 10 万本 × 500 页" 变成了 "阅读 2000 本 × 8 页"——信息获取效率提升了几个数量级。

---

## 十、DSA Decode 完整计算流（伪代码）

```python
# ─── 模型加载时（预计算 MLA 融合权重） ───
for each head i:
    W_QK_absorbed[i] = W_UQ[i].T @ W_UK[i].T   # Q 吸收 K 的解压矩阵
    W_UVO[i]         = W_UV[i] @ W_O[i]         # V+O 融合权重

# ─── 每步 Decode ───
def dsa_decode_step(x_t, cache_C_KV, cache_K_rope, cache_k_idx):
    """
    x_t:          [1, d]           当前 token 隐状态
    cache_C_KV:   [L, d_c]         MLA 潜在向量 Cache (512 维)
    cache_K_rope: [L, d_r]         解耦 RoPE Key Cache (64 维)
    cache_k_idx:  [L, n_idx*d_idx] 索引器 Key Cache (256 维, FP8)
    """
    # ══════ Stage 1: Lightning Indexer (FP8, 极快) ══════
    q_idx = fp8(x_t @ W_Q_idx)                    # [1, n_idx * d_idx]
    
    # 全局粗筛：与所有 L 个历史 Token 的索引 Key 做点积
    scores_idx = relu_weighted_dot(q_idx, cache_k_idx)  # [1, L]
    
    # Top-k 选择
    _, S_t = torch.topk(scores_idx, k=k, dim=-1)  # [1, k] 索引集合

    # ══════ Stage 2: Sparse MLA Attention (FP16, 精确) ══════
    # Gather: 仅拉取被选中 Token 的 Cache (极小 IO)
    C_selected = cache_C_KV[S_t]                   # [k, d_c]   ← 2048 × 512
    K_rope_sel = cache_K_rope[S_t]                 # [k, d_r]   ← 2048 × 64

    # Query 路径
    c_q = x_t @ W_DQ                              # [1, d_c']
    q_rope = RoPE(x_t @ W_QR)                     # [1, d_r]

    for each head i:
        # Q 吸收
        q_semantic = c_q @ W_QK_absorbed[i]        # [1, d_c]
        
        # 稀疏 Attention Score (仅在 k 个 Token 上)
        score_sem = q_semantic @ C_selected.T      # [1, k]  ← 不是 [1, L]!
        score_pos = q_rope @ K_rope_sel.T          # [1, k]
        score = (score_sem + score_pos) / sqrt(d_h + d_r)

        # V 吸收
        alpha = softmax(score)                     # [1, k]
        u_i = alpha @ C_selected                   # [1, d_c]

        # V+O 吸收
        output_i = u_i @ W_UVO[i]                  # [1, d]

    output = sum(output_i for all heads)
    return output
```

---

## 十一、全局因果链总结

```text
传统 MHA:

    x_t ──→ [Q投影] ──→ [全量 L 个 KV Cache 读取] ──→ [O(L) Attention] ──→ output
                              ↑
                        8 GB HBM IO (128K)
                       Memory Bandwidth Bound

DSA (闪电索引 + 稀疏 MLA):

    x_t ──→ [Q_idx 投影] ──→ [全量粗筛: FP8, 33 MB] ──→ [Top-k 选择]
                                                              │
                                ┌─────────────────────────────┘
                                ▼
              [Gather: 仅拉取 Top-k 的 C^{KV}, 2.3 MB]
                                │
                                ▼
              [Q 吸收] ──→ [O(k) 稀疏 Attention] ──→ [V+O 吸收] ──→ output
                                │
                         仅在 2048 个 Token × 512 维上计算
                              恒定开销，不随 L 增长
```

> **一句话总结**：DSA 让 LLM 在百万级上下文中，像人类一样"先翻目录，再精读"——而不是从第一页读到最后一页。

---

## 十二、DSA 在注意力机制演进中的位置

$$\underbrace{\text{MHA}}_{\text{全量 KV, 全量序列}} \xrightarrow{\text{KV 头裁撤}} \underbrace{\text{GQA}}_{\text{分组 KV, 全量序列}} \xrightarrow{\text{特征压缩}} \underbrace{\text{MLA}}_{\text{潜在 KV, 全量序列}} \xrightarrow{\text{序列压缩}} \underbrace{\text{DSA}}_{\text{潜在 KV, 稀疏序列}}$$

| 方案 | 特征压缩 | 序列压缩 | Cache 大小 (128K) | 代表模型 |
|------|---------|---------|-------------------|---------|
| **MHA** | ✗ | ✗ | ~8 GB/层 | GPT-3 |
| **GQA** | 4× | ✗ | ~2 GB/层 | LLaMA-3 |
| **MLA** | 57× | ✗ | ~140 MB/层 | DeepSeek-V2/V3 |
| **DSA** | 57× (MLA) | 64× ($L/k$) | ~140 MB/层 + 索引开销 | DeepSeek 下一代 |

> **注意**：DSA 的 Cache 总量与 MLA 相似（因为索引器 Key 需要全量存储），其核心优势体现在 **Decode 时的读取量和计算量**——只需读取并计算 Top-$k$ 个 Token 的 Cache。

---

## 参考文献

1. DeepSeek-AI. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* arXiv:2502.11089, 2025.
2. DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* arXiv:2405.04434, 2024.
3. DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024.
4. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
5. Kitaev, N. et al. *Reformer: The Efficient Transformer.* ICLR 2020.
6. Beltagy, I. et al. *Longformer: The Long-Document Transformer.* arXiv:2004.05150, 2020.
7. Child, R. et al. *Generating Long Sequences with Sparse Transformers.* arXiv:1904.10509, 2019.
