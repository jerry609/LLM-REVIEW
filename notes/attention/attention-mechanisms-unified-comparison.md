# 五大注意力机制全流程统一对比表

> **MHA → GQA → MLA → DSA → Linear Attention**：从训练到推理，每一步的数学公式与矩阵维度对比。

---

## 符号约定

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $L$ | 序列长度 | 128K |
| $d$ | 模型隐藏维度 | 4096 / 5120 |
| $n_h$ | Query 头数 | 32 / 128 |
| $d_h$ | 每头维度 $= d / n_h$ | 128 |
| $n_g$ | GQA 的 KV 组数 | 8 |
| $G$ | 每组共享的 Q 头数 $= n_h / n_g$ | 4 |
| $g(i)$ | 第 $i$ 个 Q 头所属的 KV 组索引 | — |
| $d_c$ | MLA 潜在空间维度 | 512 |
| $d_r$ | 解耦 RoPE 维度 | 64 |
| $k$ | DSA 稀疏选择的 Top-$k$ Token 数 | 2048 |
| $n_{\text{idx}}, d_{\text{idx}}$ | 闪电索引器头数/维度 | 4, 64 |
| $\phi(\cdot)$ | 线性注意力核特征映射 | $\text{elu}(x)+1$ |

---

## 一、训练 / Prefill 阶段

### 1.0 输入定义

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| $X \in \mathbb{R}^{L \times d}$ | $X \in \mathbb{R}^{L \times d}$ | $X \in \mathbb{R}^{L \times d}$ | $X \in \mathbb{R}^{L \times d}$ | $X \in \mathbb{R}^{L \times d}$ |

### 1.1 Query 投影

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| $Q_i = X W_i^Q$ | $Q_i = X W_i^Q$ | $c^Q = X W^{DQ}$ | 同 MLA | $Q = X W^Q$ |
| $\in \mathbb{R}^{L \times d_h}$ | $\in \mathbb{R}^{L \times d_h}$ | $Q_i^C = c^Q W_i^{UQ}$ | $Q_i^C = c^Q W_i^{UQ}$ | $\in \mathbb{R}^{L \times d_h}$ |
| $i = 1, \ldots, n_h$ | $i = 1, \ldots, n_h$ | $\in \mathbb{R}^{L \times d_h}$ | $\in \mathbb{R}^{L \times d_h}$ | |

### 1.2 Key, Value 投影（⭐ 核心差异起点）

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| 每个头独立投影 | 按**组**投影 | 统一压缩到**潜在空间** | 同 MLA | 同 MHA |
| $K_i = X W_i^K$ | $K_g = X W_g^K$ | $C^{KV} = X W^{DKV}$ | $C^{KV} = X W^{DKV}$ | $K = X W^K$ |
| $V_i = X W_i^V$ | $V_g = X W_g^V$ | $\in \mathbb{R}^{L \times d_c}$ | $\in \mathbb{R}^{L \times d_c}$ | $V = X W^V$ |
| $\in \mathbb{R}^{L \times d_h}$ | $\in \mathbb{R}^{L \times d_h}$ | | | $\in \mathbb{R}^{L \times d_h}$ |
| $i = 1, \ldots, n_h$ | $g = 1, \ldots, n_g$ | 所有头共享 | 所有头共享 | |
| **生成 $n_h$ 份** | **生成 $n_g$ 份** | **生成 1 份** | **生成 1 份** | **生成 1 份** |

### 1.3 K, V 解压（MLA / DSA 独有）

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| — | — | $K_i^C = C^{KV} W_i^{UK}$ | 同 MLA | — |
| | | $V_i^C = C^{KV} W_i^{UV}$ | | |
| | | $\in \mathbb{R}^{L \times d_h}$ | $\in \mathbb{R}^{L \times d_h}$ | |

### 1.4 闪电索引器（DSA 独有）

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| — | — | — | $q_t^{\text{idx}} = x_t W^{Q_{\text{idx}}}$ | — |
| | | | $k_s^{\text{idx}} = x_s W^{K_{\text{idx}}}$ | |
| | | | $I_{t,s} = \sum_j \text{ReLU}(q_{t,j}^{\text{idx}} \cdot (k_{s,j}^{\text{idx}})^\top) \cdot w_{t,j}$ | |
| | | | $S_t = \text{Top-}k(\{I_{t,:}\})$ | |

### 1.5 核特征映射（Linear 独有）

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| — | — | — | — | $\tilde{Q} = \phi(Q)$ |
| | | | | $\tilde{K} = \phi(K)$ |
| | | | | $\in \mathbb{R}^{L \times d_h}$ |

### 1.6 位置编码 (RoPE)

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| 直接应用 | 直接应用 | **解耦 RoPE** | 同 MLA | 通常不用 RoPE |
| $\hat{Q}_i = \text{RoPE}(Q_i)$ | $\hat{Q}_i = \text{RoPE}(Q_i)$ | $Q^R = X W^{QR}$ | 同 MLA | （使用相对位置或无） |
| $\hat{K}_i = \text{RoPE}(K_i)$ | $\hat{K}_g = \text{RoPE}(K_g)$ | $K^R = X W^{KR}$ | | |
| | | $\hat{Q}_i = [Q_i^C, \text{RoPE}(Q^R)]$ | | |
| | | $\hat{K}_i = [K_i^C, \text{RoPE}(K^R)]$ | | |

### 1.7 Attention 矩阵计算

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| **全量 $L \times L$** | **全量 $L \times L$** | **全量 $L \times L$** | **稀疏 $L \times k$** | **结合律重排** |
| $A_i = \hat{Q}_i \hat{K}_i^\top$ | $A_i = \hat{Q}_i \hat{K}_{g(i)}^\top$ | $A_i = \hat{Q}_i \hat{K}_i^\top$ | $A_i = \hat{Q}_i \hat{K}_i^\top$ | $S_{\text{global}} = \tilde{K}^\top V$ |
| $\in \mathbb{R}^{L \times L}$ | $\in \mathbb{R}^{L \times L}$ | $\in \mathbb{R}^{L \times L}$ | 但 **仅 $s \in S_t$** | $\in \mathbb{R}^{d_h \times d_h}$ |
| $S_i = \text{Softmax}(\frac{A_i}{\sqrt{d_h}})$ | $S_i = \text{Softmax}(\frac{A_i}{\sqrt{d_h}})$ | $S_i = \text{Softmax}(\frac{A_i}{\sqrt{d_h+d_r}})$ | $S_i = \text{Softmax}(\frac{A_i}{\sqrt{d_h+d_r}})$ | **无 Softmax** |

### 1.8 加权输出

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| $O_i = S_i V_i$ | $O_i = S_i V_{g(i)}$ | $O_i = S_i V_i^C$ | $O_i = \sum_{s \in S_t} S_{t,s} V_s^C$ | $O = \frac{\tilde{Q} \cdot S_{\text{global}}}{\tilde{Q} \cdot Z_{\text{global}}}$ |
| $\in \mathbb{R}^{L \times d_h}$ | $\in \mathbb{R}^{L \times d_h}$ | $\in \mathbb{R}^{L \times d_h}$ | $\in \mathbb{R}^{L \times d_h}$ | $Z_{\text{global}} = \tilde{K}^\top \mathbf{1}_L$ |

### 1.9 Output 投影

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| $O = \text{Concat}(O_1, \ldots, O_{n_h}) W^O$ | 完全相同 | 完全相同 | 完全相同 | 完全相同 |
| $\in \mathbb{R}^{L \times d}$ | $\in \mathbb{R}^{L \times d}$ | $\in \mathbb{R}^{L \times d}$ | $\in \mathbb{R}^{L \times d}$ | $\in \mathbb{R}^{L \times d}$ |

### 1.10 Prefill 复杂度

| | MHA | GQA | MLA | DSA | Linear |
|--|-----|-----|-----|-----|--------|
| **时间** | $O(L^2 d_h n_h)$ | $O(L^2 d_h n_h)$ | $O(L^2 (d_h+d_r) n_h)$ | $O(L^2 d_{\text{idx}} n_{\text{idx}}) + O(Lk d_h n_h)$ | $O(L d_h^2 n_h)$ |
| **简写** | $O(L^2 d)$ | $O(L^2 d)$ | $O(L^2 d)$ | $O(L \cdot k \cdot d)$ | $O(L \cdot d_h \cdot d)$ |

---

## 二、KV Cache 存储阶段

### 2.1 物理存储内容

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| $\{\hat{K}_i, V_i\}_{i=1}^{n_h}$ | $\{\hat{K}_g, V_g\}_{g=1}^{n_g}$ | $\{C^{KV}, K^{\text{RoPE}}\}$ | $\{C^{KV}, K^{\text{RoPE}}, k^{\text{idx}}\}$ | $\{S_t, Z_t\}$ |
| 全量 K, V | 分组 K, V | 潜在向量 + RoPE Key | 潜在向量 + RoPE Key + 索引 Key | 隐状态矩阵 + 归一化向量 |

### 2.2 每层每 Token 显存维度

| | MHA | GQA | MLA | DSA | Linear |
|--|-----|-----|-----|-----|--------|
| **维度** | $2 n_h d_h$ | $2 n_g d_h$ | $d_c + d_r$ | $d_c + d_r + n_{\text{idx}} d_{\text{idx}}$ | — |
| **典型值** | $2 \times 32 \times 128$ | $2 \times 8 \times 128$ | $512 + 64$ | $512 + 64 + 256$ | — |
| | **= 8192** | **= 2048** | **= 576** | **= 832** | — |
| **压缩比 vs MHA** | 1× | 4× | 14.2× | 9.8× | — |

### 2.3 Cache 随序列长度增长方式

| | MHA | GQA | MLA | DSA | Linear |
|--|-----|-----|-----|-----|--------|
| **增长** | $O(L)$ 线性 | $O(L)$ 线性 | $O(L)$ 线性 | $O(L)$ 线性 | **$O(1)$ 恒定** |
| **128K 每层** | ~8 GB | ~2 GB | ~140 MB | ~210 MB | **~32 KB** |

### 2.4 Cache 数学表达

| | MHA | GQA | MLA | DSA | Linear |
|--|-----|-----|-----|-----|--------|
| **K Cache** | $\in \mathbb{R}^{L \times n_h \times d_h}$ | $\in \mathbb{R}^{L \times n_g \times d_h}$ | — | — | — |
| **V Cache** | $\in \mathbb{R}^{L \times n_h \times d_h}$ | $\in \mathbb{R}^{L \times n_g \times d_h}$ | — | — | — |
| **$C^{KV}$** | — | — | $\in \mathbb{R}^{L \times d_c}$ | $\in \mathbb{R}^{L \times d_c}$ | — |
| **$K^{\text{RoPE}}$** | — | — | $\in \mathbb{R}^{L \times d_r}$ | $\in \mathbb{R}^{L \times d_r}$ | — |
| **$k^{\text{idx}}$** | — | — | — | $\in \mathbb{R}^{L \times n_{\text{idx}} d_{\text{idx}}}$ | — |
| **$S_t$** | — | — | — | — | $\in \mathbb{R}^{d_h \times d_h}$ |
| **$Z_t$** | — | — | — | — | $\in \mathbb{R}^{d_h \times 1}$ |

---

## 三、推理 / Decode 阶段

输入：新 Token $x_t \in \mathbb{R}^{1 \times d}$

### 3.1 Query 投影

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| $q_{t,i} = \text{RoPE}(x_t W_i^Q)$ | $q_{t,i} = \text{RoPE}(x_t W_i^Q)$ | $c_t^Q = x_t W^{DQ}$ | 同 MLA | $\tilde{q}_t = \phi(x_t W^Q)$ |
| $\in \mathbb{R}^{1 \times d_h}$ | $\in \mathbb{R}^{1 \times d_h}$ | $q_{t,i}^C = c_t^Q W_i^{UQ}$ | + 索引 Query: | $\in \mathbb{R}^{1 \times d_h}$ |
| | | $q_{t,i}^R = \text{RoPE}(x_t W^{QR})$ | $q_t^{\text{idx}} = \text{FP8}(x_t W^{Q_{\text{idx}}})$ | |

### 3.2 索引器粗筛（DSA 独有）

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| — | — | — | $I_{t,s} = \sum_j \text{ReLU}(q_{t,j}^{\text{idx}} \cdot (k_{s,j}^{\text{idx}})^\top) \cdot w_{t,j}$ | — |
| | | | $S_t = \text{Top-}k(\{I_{t,:}\})$ | |
| | | | FLOPs: $n_{\text{idx}} \cdot d_{\text{idx}} \cdot L$ (FP8) | |

### 3.3 Cache 读取 / 状态更新

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| 全量读取 | 按组读取 | 全量读取 | **Gather Top-$k$** | **外积累加** |
| $K_i^{\text{cache}} \in \mathbb{R}^{L \times d_h}$ | $K_{g(i)}^{\text{cache}} \in \mathbb{R}^{L \times d_h}$ | $C^{KV} \in \mathbb{R}^{L \times d_c}$ | $C^{KV}[S_t] \in \mathbb{R}^{k \times d_c}$ | $S_t = S_{t-1} + \tilde{k}_t^\top v_t$ |
| $V_i^{\text{cache}} \in \mathbb{R}^{L \times d_h}$ | $V_{g(i)}^{\text{cache}} \in \mathbb{R}^{L \times d_h}$ | | $K^{\text{RoPE}}[S_t] \in \mathbb{R}^{k \times d_r}$ | $Z_t = Z_{t-1} + \tilde{k}_t^\top$ |
| 读取量: $2 n_h d_h L$ | 读取量: $2 n_g d_h L$ | 读取量: $(d_c + d_r) L$ | 读取量: $(d_c + d_r) k$ | 读取量: $d_h^2 + d_h$ |

### 3.4 矩阵吸收（MLA / DSA 独有）

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| — | — | **Q 吸收** | 同 MLA | — |
| | | $\tilde{q}_{t,i}^C = q_{t,i}^C (W_i^{UK})^\top$ | $\tilde{q}_{t,i}^C = q_{t,i}^C (W_i^{UK})^\top$ | |
| | | $\in \mathbb{R}^{1 \times d_c}$ | $\in \mathbb{R}^{1 \times d_c}$ | |
| | | **永不解压 Cache** | **永不解压 Cache** | |

### 3.5 Attention Score 计算

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| $\text{Score} = q_{t,i} \cdot (K_i^{\text{cache}})^\top$ | $\text{Score} = q_{t,i} \cdot (K_{g(i)}^{\text{cache}})^\top$ | $\text{Score} = \tilde{q}_{t,i}^C \cdot (C^{KV})^\top$ | $\text{Score} = \tilde{q}_{t,i}^C \cdot (C^{KV}[S_t])^\top$ | （无此步，直接查状态） |
| $\in \mathbb{R}^{1 \times L}$ | $\in \mathbb{R}^{1 \times L}$ | $+ \; q_{t,i}^R \cdot (K^{\text{RoPE}})^\top$ | $+ \; q_{t,i}^R \cdot (K^{\text{RoPE}}[S_t])^\top$ | |
| | | $\in \mathbb{R}^{1 \times L}$ | $\in \mathbb{R}^{1 \times k}$ | |

### 3.6 加权求和 (Value)

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| $o_{t,i} = \text{softmax} \cdot V_i^{\text{cache}}$ | $o_{t,i} = \text{softmax} \cdot V_{g(i)}^{\text{cache}}$ | **V 吸收** | **V 吸收** | $o_t = \frac{\tilde{q}_t S_t}{\tilde{q}_t Z_t}$ |
| $\in \mathbb{R}^{1 \times d_h}$ | $\in \mathbb{R}^{1 \times d_h}$ | $u_{t,i} = \text{softmax} \cdot C^{KV}$ | $u_{t,i} = \text{softmax} \cdot C^{KV}[S_t]$ | $\in \mathbb{R}^{1 \times d_h}$ |
| | | $\in \mathbb{R}^{1 \times d_c}$ | $\in \mathbb{R}^{1 \times d_c}$ | |

### 3.7 Output 投影

| MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|--------|
| $o_t = \sum_i o_{t,i} W_i^O$ | $o_t = \sum_i o_{t,i} W_i^O$ | **O 吸收** | 同 MLA | $o_t = o_t W^O$ |
| | | $o_t = \sum_i u_{t,i} \underbrace{W_i^{UV} W_i^O}_{W_i^{UV\_O}}$ | $o_t = \sum_i u_{t,i} W_i^{UV\_O}$ | |
| | | 预计算融合权重 | 预计算融合权重 | |

---

## 四、Decode 单步复杂度与性能对比

### 4.1 计算复杂度

| | MHA | GQA | MLA | DSA | Linear |
|--|-----|-----|-----|-----|--------|
| **索引器** | — | — | — | $O(L \cdot d_{\text{idx}} \cdot n_{\text{idx}})$ (FP8) | — |
| **Attention** | $O(L \cdot d_h \cdot n_h)$ | $O(L \cdot d_h \cdot n_h)$ | $O(L \cdot d_c \cdot n_h)$ | $O(k \cdot d_c \cdot n_h)$ | $O(d_h^2 \cdot n_h)$ |
| **总计** | $O(L \cdot d)$ | $O(L \cdot d)$ | $O(L \cdot d)$ | $O(L \cdot d_{\text{idx}}) + O(k \cdot d)$ | $O(d_h \cdot d)$ |
| **随 $L$ 增长** | 线性 ↑ | 线性 ↑ | 线性 ↑ | 仅索引器线性 ↑ | **恒定** |

### 4.2 Cache IO 读取量（128K 序列，每层）

| | MHA | GQA | MLA | DSA | Linear |
|--|-----|-----|-----|-----|--------|
| **读取维度** | $2 n_h d_h \times L$ | $2 n_g d_h \times L$ | $(d_c + d_r) \times L$ | $(d_c + d_r) \times k$ + 索引器 | $d_h^2 + d_h$ |
| **FP16 字节** | ~8 GB | ~2 GB | ~140 MB | ~2.3 MB + 33 MB | ~32 KB |
| **vs MHA** | 1× | 4× | **57×** | **~3640×** | **~250,000×** |

### 4.3 Decode 延迟随序列长度变化

| $L$ | MHA | GQA | MLA | DSA | Linear |
|-----|-----|-----|-----|-----|--------|
| 4K | 基准 | 基准 | 基准 | ~基准 | 基准 |
| 32K | 8× | 8× | 8× | 索引器 8×, 主注意力不变 | **不变** |
| 128K | 32× | 32× | 32× | 索引器 32×, 主注意力不变 | **不变** |
| 1M | 256× | 256× | 256× | 索引器 256×, 主注意力不变 | **不变** |

### 4.4 瓶颈类型

| | MHA | GQA | MLA | DSA | Linear |
|--|-----|-----|-----|-----|--------|
| **瓶颈** | Memory BW | Memory BW（缓解） | Memory BW（大幅缓解） | 索引器 BW（极小）+ Compute | Compute |

---

## 五、训练权重参数量对比（每层注意力模块）

| 权重矩阵 | MHA | GQA | MLA | DSA | Linear |
|----------|-----|-----|-----|-----|--------|
| $W^Q$ | $n_h \times d \times d_h$ | $n_h \times d \times d_h$ | $W^{DQ}: d \times d_c'$ | 同 MLA | $d \times d_h$ |
| | | | + $n_h \times W_i^{UQ}: d_c' \times d_h$ | + 同 MLA | |
| $W^K$ | $n_h \times d \times d_h$ | $n_g \times d \times d_h$ | $W^{DKV}: d \times d_c$ | 同 MLA | $d \times d_h$ |
| | | | + $n_h \times W_i^{UK}: d_c \times d_h$ | + 同 MLA | |
| $W^V$ | $n_h \times d \times d_h$ | $n_g \times d \times d_h$ | （含在 $W^{DKV}$ 和 $W_i^{UV}$ 中） | 同 MLA | $d \times d_h$ |
| $W^O$ | $d \times d$ | $d \times d$ | $d \times d$ | $d \times d$ | $d \times d$ |
| RoPE | （无额外权重） | （无额外权重） | $W^{QR}: d \times d_r$ | 同 MLA | — |
| | | | $W^{KR}: d \times d_r$ | | |
| 索引器 | — | — | — | $W^{Q_{\text{idx}}}: d \times n_{\text{idx}} d_{\text{idx}}$ | — |
| | | | | $W^{K_{\text{idx}}}: d \times n_{\text{idx}} d_{\text{idx}}$ | |
| 门控权重 | — | — | — | $w_{t,j}$（可学习） | — |

---

## 六、设计哲学与适用场景

| 维度 | MHA | GQA | MLA | DSA | Linear |
|------|-----|-----|-----|-----|--------|
| **压缩策略** | 无 | KV 头裁撤（物理裁撤） | KV 特征压缩（化学提纯） | 序列稀疏 + 特征压缩 | 全量信息坍缩为固定状态 |
| **精确检索** | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| **超长文本** | ✗（显存爆炸） | ✗（缓解但仍 $O(L)$） | △（Cache 仍 $O(L)$） | ✓（Decode 恒定） | ✓✓（Cache 也恒定） |
| **实现复杂度** | 低 | 低（几乎无改动） | 高（矩阵吸收 + 解耦 RoPE） | 极高（两阶段 + 索引器 + Gather） | 中（结合律 + 状态递推） |
| **生态兼容** | 所有框架 | 所有主流框架 | 需专用算子 | 需专用算子 | 需专用算子 |
| **代表模型** | GPT-3/4 | LLaMA-3, Qwen-2 | DeepSeek-V2/V3 | DeepSeek 下一代 | Mamba, RWKV, RetNet |
| **最佳场景** | 通用基准 | 生产部署（性价比最优） | 长上下文 + 低成本推理 | 百万级上下文 | 流式生成 / 无限对话 |

---

## 七、演进全景一行公式

$$\underbrace{\text{MHA}}_{\substack{\text{全量 KV} \\ \text{全量序列} \\ O(L^2), O(L)_{\text{dec}}}} \xrightarrow{\text{头裁撤}} \underbrace{\text{GQA}}_{\substack{\text{分组 KV} \\ \text{全量序列} \\ O(L^2), O(L)_{\text{dec}}}} \xrightarrow{\text{特征压缩}} \underbrace{\text{MLA}}_{\substack{\text{潜在 KV} \\ \text{全量序列} \\ O(L^2), O(L)_{\text{dec}}}} \xrightarrow{\text{序列压缩}} \underbrace{\text{DSA}}_{\substack{\text{潜在 KV} \\ \text{稀疏序列} \\ O(Lk), O(k)_{\text{dec}}}} \xrightarrow{\text{结合律}} \underbrace{\text{Linear}}_{\substack{\text{固定状态} \\ \text{无序列} \\ O(Ld_h^2), O(1)_{\text{dec}}}}$$

---

## 参考文献

1. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
2. Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023.
3. DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model.* 2024.
4. DeepSeek-AI. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* 2025.
5. Katharopoulos, A. et al. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020.
6. Gu, A. & Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023.
