# 位置编码全景对比

> Sinusoidal / Learned / RoPE / ALiBi / NoPE —— 面试必知

---

## 一、为什么需要位置编码？

- Transformer 的 Attention 是**置换不变的**（permutation invariant）
- `Attention(Q,K,V)` 不知道 token 的顺序
- 必须通过位置编码注入序列位置信息

---

## 二、各种位置编码方法

### 2.1 Sinusoidal（绝对位置编码）

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

- **方式**：`x_out = x + PE(pos)` —— 直接加到 embedding 上
- **优点**：不需要学习参数，理论上支持任意长度
- **缺点**：绝对位置 → 长度外推差，不具备相对位置感知
- **使用者**：原始 Transformer（Vaswani et al.），BERT

### 2.2 Learned Position Embedding（可学习绝对位置）

```python
pos_embed = nn.Embedding(max_len, d_model)
x_out = x + pos_embed(position_ids)
```

- **方式**：每个位置一个可学习向量
- **优点**：灵活，模型自适应
- **缺点**：不能外推到训练时没见过的长度；参数量 `max_len × d_model`
- **使用者**：GPT-2, BERT

### 2.3 RoPE（旋转位置编码）⭐ **现代 LLM 标配**

```
RoPE(x, pos) = x ⊙ cos(pos·θ) + rotate(x) ⊙ sin(pos·θ)
```

- **核心思想**：用旋转矩阵编码位置，使内积自然包含相对距离信息
- **作用位置**：只在 Q, K 上施加（不改 V，不改 embedding）
- **关键性质**：
  - `⟨RoPE(q, m), RoPE(k, n)⟩ = f(q, k, m-n)` → **内积只依赖相对距离**
  - 不需要额外参数
  - 频率参数 θ 控制不同维度的旋转速度

#### RoPE 频率设计
```
θ_i = base^(-2i/d)     base 通常 = 10000
```
- 低维度（i 小）：θ 大 → 高频旋转 → 编码**短距离**位置差异
- 高维度（i 大）：θ 小 → 低频旋转 → 编码**长距离**位置差异

#### RoPE 长度扩展方法
| 方法 | 核心思想 | 效果 |
|------|---------|------|
| **位置插值 (PI)** | 将位置 pos/scale 压缩到训练范围 | 简单但高频信息丢失 |
| **NTK-Aware** | 调整 base 而不是 pos，保留高频 | 比 PI 好 |
| **YaRN** | 不同频率分量用不同缩放策略 | **目前最优**，Qwen/Llama 都用 |
| **Dynamic NTK** | 推理时根据当前长度动态调整 base | 无需重训练 |

- **使用者**：Llama 1/2/3, Qwen, DeepSeek, Mistral, GPT-NeoX

### 2.4 ALiBi（Attention with Linear Biases）⭐ **另一大流派**

```
Attention(Q, K, V) = softmax(QK^T/√d + ALiBi_bias) V
```

- **核心思想**：不修改 Q/K，而是在 attention score 上加一个**线性距离偏置**
- **公式**：`bias(i,j) = -m × |i - j|`
  - m 是 head-specific 的斜率，按几何级数分配
  - m = {2^(-8/H), 2^(-16/H), ...}，H 个 head 用不同斜率
- **效果**：距离越远的 token 对，attention score 被减去越多 → 自然衰减

#### ALiBi vs RoPE 对比
| 特性 | RoPE | ALiBi |
|------|------|-------|
| 修改位置 | Q 和 K 的值 | Attention score（加 bias） |
| 外推能力 | 需要额外扩展（YaRN等） | **天然外推好**（训练短推理长） |
| 额外参数 | 0 | 0（m 是固定的） |
| 精度 | **略高** | 略低（长距离信号被压制） |
| 实现 | 需要旋转 Q/K | 只需加 bias matrix |
| 训练效率 | 需要 sin/cos 计算 | **略快**（只是矩阵加法） |
| KV Cache | 不影响 | 不影响 |
| 使用者 | Llama, Qwen, DeepSeek | **BLOOM, MPT, Falcon** |

#### ALiBi 的优劣势
- **优势**：
  - 天然支持长度外推（训练 2K 可推理 8K+）
  - 实现极简，不需要复杂的 sin/cos 计算
  - 对 KV Cache 无额外影响
- **劣势**：
  - 长距离信息被强制衰减 → 不适合需要精确长距离依赖的任务
  - 在超长上下文（>100K）场景不如 RoPE + YaRN
  - 固定的线性衰减模式缺乏灵活性

### 2.5 NoPE（无显式位置编码）

- 部分研究发现：causal attention mask 本身就隐含了位置信息
- 因为 causal mask 让 token i 只能看到 token 1..i → 暗含了位置
- 实验表明在某些设置下可以不加任何位置编码
- **但不是主流**，实际 LLM 都会用 RoPE 或 ALiBi

### 2.6 CoPE（Contextual Position Encoding, Meta 2024）

- 位置编码由上下文决定，而非固定的位置 index
- `gate = sigmoid(Q @ k_pos)`，动态决定位置权重
- 适合变长输入和跨文档场景

---

## 三、总对比表

| 方法 | 类型 | 作用位置 | 外推能力 | 额外参数 | 适合场景 |
|------|------|---------|---------|---------|---------|
| Sinusoidal | 绝对 | Embedding | 差 | 0 | 短序列 |
| Learned | 绝对 | Embedding | 差 | max_len×d | 短序列 |
| **RoPE** | **相对** | **Q, K** | **中（需扩展）** | **0** | **主流 LLM** |
| **ALiBi** | **相对** | **Attn Score** | **好** | **0** | **中等长度** |
| NoPE | 隐式 | — | 中 | 0 | 实验性 |
| CoPE | 上下文 | Q, K | 好 | 少量 | 新方向 |

---

## 四、面试高频问答

**Q1：RoPE 的核心原理是什么？为什么不需要额外参数？**
> RoPE 对 Q 和 K 施加位置相关的旋转变换，使得 QK 内积自然变为相对距离的函数。旋转矩阵由 sin/cos 确定，不需要学习参数。

**Q2：RoPE 如何扩展到更长的上下文？**
> 三种方法：(1) 位置插值（PI）—— 压缩位置到训练范围，简单但损失高频；(2) NTK-Aware —— 调整 base 频率；(3) YaRN —— 不同频率分量用不同缩放，目前最优。

**Q3：ALiBi 和 RoPE 的根本区别是什么？**
> RoPE 修改 Q/K 的值（旋转），ALiBi 在 attention score 上加线性距离偏置。ALiBi 天然支持长度外推但长距离信号被压制，RoPE 精度略高但需要额外扩展方法。

**Q4：为什么 Llama/Qwen 选 RoPE 而不是 ALiBi？**
> (1) RoPE 在长上下文场景（128K+）配合 YaRN 效果更好；(2) RoPE 不会强制衰减远距离信息，更适合需要精确长距离推理的任务；(3) 实验证明 RoPE 在大规模模型上的 PPL 更低。

**Q5：如果训练时只用 4K 上下文，推理时如何支持 128K？**
> 用 RoPE + YaRN 扩展：分频率段调整 base，低频外推、高频保持，再用少量长文本数据微调（<1000 步）。ALiBi 可以直接外推但精度不如 RoPE+YaRN。
