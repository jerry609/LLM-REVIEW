# Transformer 核心组件深度拆解

> FFN / 残差连接 / 归一化 / 稀疏注意力 —— 面试高频考点

---

## 一、FFN (Feed-Forward Network)

### 1.1 标准 FFN
```
FFN(x) = σ(xW₁ + b₁)W₂ + b₂
```
- 两层线性变换，中间加激活函数
- 维度变化：`d_model → d_ff → d_model`
- 通常 `d_ff = 4 × d_model`（如 d_model=4096, d_ff=16384）
- **作用**：每个 token 独立的非线性变换（"token mixer" 是 attention，"channel mixer" 是 FFN）

### 1.2 SwiGLU FFN（Llama / Qwen / DeepSeek 标配）
```
SwiGLU(x) = (SiLU(xW_gate) ⊙ xW_up) W_down
```
- **SiLU**（又叫 Swish）：`SiLU(x) = x × σ(x)`
- **门控机制**：`W_gate` 和 `W_up` 两个独立投影，`⊙` 是逐元素乘
- **维度**：`d_model → d_ff`（gate和up各一个）→ `d_ff → d_model`（down）
- **参数量**：3 个矩阵 vs 标准 FFN 的 2 个 → 参数多 50%
- **为什么更好**：门控让模型学习"哪些特征通过"，比 ReLU/GELU 更平滑

### 1.3 GeGLU / ReGLU
| 变体 | 激活函数 | 使用者 |
|------|---------|--------|
| ReLU FFN | `max(0, x)` | 原始 Transformer |
| GELU FFN | `x × Φ(x)` | GPT-2, BERT |
| SwiGLU | `x × σ(x)` (gated) | Llama3, Qwen2.5, DeepSeek-V3 |
| GeGLU | `GELU(x)` (gated) | PaLM |
| ReGLU | `ReLU(x)` (gated) | 一些变体 |

### 1.4 面试关键点
- **Q：为什么 SwiGLU 比标准 FFN 好？**
  - 门控机制提供更灵活的特征选择
  - SiLU 比 ReLU 平滑，梯度信号更好
  - 实验证明在同等 FLOPs 下质量更高
- **Q：SwiGLU 的 d_ff 怎么算？**
  - Llama: `d_ff = int(8/3 × d_model)`，再取最近的 256 倍数
  - 例如 d_model=4096 → d_ff ≈ 10922 → 取 11008

---

## 二、残差连接 (Residual Connection)

### 2.1 基本形式
```
output = x + SubLayer(x)
```
- 每个子层（Attention / FFN）的输出加上输入
- **作用**：缓解梯度消失，让梯度直接回传

### 2.2 Pre-Norm vs Post-Norm

```
Post-Norm (原始 Transformer):           Pre-Norm (现代 LLM 标配):
x → SubLayer → Add → Norm → out        x → Norm → SubLayer → Add → out
```

| 特性 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| 训练稳定性 | 较差，需要 warmup | **好**，训练更稳定 |
| 最终性能 | 略好（如果能训起来） | 略差（但差距很小） |
| 梯度流动 | 残差路径经过 Norm | **残差路径直通**（梯度无衰减） |
| 使用者 | 原始 Transformer, BERT | **Llama, Qwen, DeepSeek, GPT** |

### 2.3 为什么 Pre-Norm 更主流？
- 深层模型（>40层）Post-Norm 训练不稳定
- Pre-Norm 让残差连接形成"信息高速公路"
- 实际 PPL 差距 <0.5%，但训练稳定性差距很大

### 2.4 DeepNorm（微软）
- `output = x × α + SubLayer(Norm(x))`
- α 随层数增大而增大，β 用于缩放初始化
- 允许训练 1000 层 Transformer（Post-Norm + DeepNorm）

---

## 三、归一化 (Normalization)

### 3.1 Layer Normalization
```python
LayerNorm(x) = γ × (x - μ) / sqrt(σ² + ε) + β
# μ = mean(x), σ² = var(x)  在 hidden_dim 维度计算
# γ, β 是可学习参数
```

### 3.2 RMS Normalization（LLM 标配）
```python
RMSNorm(x) = γ × x / sqrt(mean(x²) + ε)
# 只用 RMS (Root Mean Square)，省去 mean 计算
# 没有 β (bias) 参数
```

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算 | 需要 mean + var | **只需 mean(x²)** |
| 参数 | γ, β | **只有 γ** |
| 速度 | baseline | **快 ~10-15%** |
| 效果 | baseline | 几乎无差异 |
| 使用者 | BERT, GPT-2 | **Llama, Qwen, DeepSeek** |

### 3.3 面试一句话
- "RMSNorm 省去了均值中心化，只做缩放归一化，速度快 10-15% 且效果不降。现代 LLM 几乎全部用 RMSNorm + Pre-Norm。"

---

## 四、稀疏注意力 (Sparse Attention)

### 4.1 为什么需要稀疏注意力？
- 标准 Attention：O(T²) 计算和内存
- T=128K 时：128K × 128K × 2bytes ≈ 32 GB **只存 attention matrix**
- 稀疏注意力：只计算部分 (i,j) 对，降低为 O(T × k)

### 4.2 主要稀疏模式

#### 滑动窗口注意力 (Sliding Window / Local Attention)
```
每个 token 只看前后 w 个 token
复杂度：O(T × w)，w << T
```
- **Mistral 7B**：`sliding_window = 4096`
- **优点**：实现简单，适合局部依赖
- **缺点**：无法捕获长距离依赖

#### Longformer（局部 + 全局混合）
```
┌───────────────────────────────┐
│ ■ ■ ■ ■ . . . . . . . . . . │  ← 全局 token（[CLS]等）看所有 token
│ ■ ■ ■ ■ . . . . . . . . . . │
│ ■ ■ ■ ■ ■ . . . . . . . . . │  ← 每个 token 看局部窗口
│ . ■ ■ ■ ■ ■ . . . . . . . . │
│ . . ■ ■ ■ ■ ■ . . . . . . . │
│ . . . ■ ■ ■ ■ ■ . . . . . . │
│ . . . . . . . . ■ ■ ■ ■ ■ . │
│ . . . . . . . . . ■ ■ ■ ■ ■ │
└───────────────────────────────┘
```
- 局部窗口 + 少数全局 token
- 复杂度：O(T × (w + g))，g = 全局 token 数

#### BigBird（随机 + 局部 + 全局）
```
BigBird = Local Window + Random + Global
```
- 在 Longformer 基础上增加随机连接
- 理论证明：随机稀疏图可以近似完全图

#### Dilated / Strided Attention
- 间隔 d 个 token 采样一次（类似 dilated convolution）
- 多头用不同 dilation rate → 覆盖不同范围

### 4.3 分层稀疏（MiniMax-01 / Jamba）
```
底层（局部模式）：Linear Attention / Sliding Window
  ↓  只关注局部模式
顶层（全局模式）：Full Softmax Attention
  ↓  关注全局依赖
```
- MiniMax-01：2/3 层 Lightning Attention + 1/3 层 Softmax
- Jamba：Mamba 层 + Transformer 层交替

### 4.4 Sparse Attention 对比表

| 方法 | 复杂度 | 局部 | 全局 | 随机 | 使用者 |
|------|--------|------|------|------|--------|
| Full Attention | O(T²) | ✅ | ✅ | ✅ | GPT, Llama |
| Sliding Window | O(T×w) | ✅ | ❌ | ❌ | Mistral |
| Longformer | O(T×(w+g)) | ✅ | ✅ | ❌ | Longformer |
| BigBird | O(T×(w+g+r)) | ✅ | ✅ | ✅ | BigBird |
| Ring Attention | O(T²/N) | ✅ | ✅ | - | 分布式 |
| Inf-LLM | O(T×w+k) | ✅ | ✅* | ❌ | 长文本推理 |

> *Inf-LLM 通过 offload + retrieval 实现"伪全局"

### 4.5 面试关键点
- **Q：Mistral 的 Sliding Window Attention 和标准 Attention 什么区别？**
  - 每个 token 只看窗口内的 w 个 token，KV Cache 固定大小 w
  - 跨层叠加后，信息可以传播 w×L 距离
  - 适合推理效率，但长距离依赖靠层间传递
- **Q：稀疏注意力有什么 trade-off？**
  - 效率 vs 质量：稀疏模式可能丢失关键的长距离依赖
  - 实现复杂度：需要特殊 CUDA kernel
  - 现代方案：FlashAttention + Sliding Window 是最佳实践

---

## 五、完整 Transformer Decoder Block 流程图

```
Input Embeddings (x)
         │
         ▼
┌─────────────────────┐
│    RMSNorm (Pre)     │
│         │            │
│    ┌────▼────┐       │
│    │ Q Proj  │       │
│    │ K Proj  │→ RoPE │
│    │ V Proj  │       │
│    └────┬────┘       │
│         │            │
│    GQA Attention     │
│    (Sparse/Full)     │
│         │            │
│    O Proj            │
│         │            │
│    Dropout (训练)     │
└────────┬────────────┘
         │
    + Residual ←── x
         │
         ▼
┌─────────────────────┐
│    RMSNorm (Pre)     │
│         │            │
│    ┌────▼────┐       │
│    │ W_gate  │       │
│    │ W_up    │       │
│    └────┬────┘       │
│         │            │
│    SwiGLU FFN        │
│    SiLU(gate) ⊙ up   │
│         │            │
│    W_down            │
│         │            │
│    Dropout (训练)     │
└────────┬────────────┘
         │
    + Residual ←── (上面 Attention 输出)
         │
         ▼
    Next Layer / Final RMSNorm → LM Head → logits
```

### 参数量估算（以 Llama3-70B 为例）
| 组件 | 参数量公式 | 70B 实际 |
|------|-----------|---------|
| Q Proj | `d × d` = 8192² | ~67M |
| K Proj | `d × d_kv` = 8192 × 1024 | ~8.4M |
| V Proj | `d × d_kv` = 8192 × 1024 | ~8.4M |
| O Proj | `d × d` = 8192² | ~67M |
| W_gate | `d × d_ff` = 8192 × 28672 | ~235M |
| W_up | `d × d_ff` = 8192 × 28672 | ~235M |
| W_down | `d_ff × d` = 28672 × 8192 | ~235M |
| RMSNorm ×2 | `d` × 2 = 16384 | ~16K |
| **单层总计** | | **~856M** |
| **80 层 + embedding** | | **~70B** |

---

## 面试高频问答

**Q1：Post-Norm 和 Pre-Norm 的区别？为什么现在都用 Pre-Norm？**
> Pre-Norm 把归一化放在子层之前，残差连接直通无衰减。训练更稳定，尤其是深层模型（>40 层）。Post-Norm 理论上最终性能略高但训练易崩溃。

**Q2：RMSNorm 和 LayerNorm 的区别？**
> RMSNorm 省去了均值中心化步骤，只做缩放归一化（x/RMS(x)），速度快 10-15%，效果几乎无差异。

**Q3：SwiGLU 比 ReLU FFN 好在哪里？参数量多多少？**
> SwiGLU 通过门控机制让模型学习特征选择，同等 FLOPs 下质量更高。参数量多 50%（3 个矩阵 vs 2 个），但通常配合较小的 d_ff 来平衡。

**Q4：Sliding Window Attention 的 KV Cache 大小是多少？**
> 固定为 window_size × n_kv_heads × head_dim × 2(K+V) × dtype_bytes，不随序列长度增长。Mistral 7B 的 window=4096，KV Cache 固定 ~256MB/request。
