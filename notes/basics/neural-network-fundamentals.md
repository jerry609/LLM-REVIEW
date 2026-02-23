# 神经网络核心概念

> 前向传播、反向传播、损失函数、梯度下降、激活函数 —— 手撕训练的数学基础

---

## 一、前向传播 (Forward Propagation)

### 1.1 单层线性变换
```
输入 x ∈ R^d  →  z = Wx + b  →  a = σ(z)  →  输出 a ∈ R^m
```
- `W ∈ R^{m×d}`：权重矩阵
- `b ∈ R^m`：偏置
- `σ`：激活函数

### 1.2 多层前向传播

```
x → [Linear₁ → ReLU] → [Linear₂ → ReLU] → [Linear₃ → Softmax] → ŷ
     隐藏层 1          隐藏层 2            输出层
```

数学表达：
```
h₁ = σ(W₁x + b₁)
h₂ = σ(W₂h₁ + b₂)
ŷ  = Softmax(W₃h₂ + b₃)
```

### 1.3 Transformer 中的前向传播
```
x → [Embedding + PE] → N × [Self-Attention → Add&Norm → FFN → Add&Norm] → [LM Head] → logits
```
- 每一步都是矩阵乘法 + 非线性 → 与经典全连接网络本质相同

---

## 二、损失函数 (Loss Functions)

### 2.1 交叉熵损失 (Cross-Entropy) ⭐ LLM 核心

```
L_CE = -∑ᵢ yᵢ log(ŷᵢ)
```

- `y`：真实标签的 one-hot 向量（或目标 token 的 index）
- `ŷ`：模型预测的概率分布（Softmax 输出）
- 对于语言模型（next-token prediction）：

```
L = -(1/T) ∑ₜ log P(xₜ | x₁, ..., xₜ₋₁)
```

| 任务 | 损失函数 | 说明 |
|------|---------|------|
| 语言模型 (CLM) | Cross-Entropy | 预测下一个 token |
| 分类 | Cross-Entropy | 预测类别概率 |
| 回归 | MSE | 预测连续值 |
| 对比学习 (CLIP) | InfoNCE | 图文匹配 |
| 偏好对齐 (DPO) | DPO Loss | log-ratio 优化 |

### 2.2 困惑度 (Perplexity)

```
PPL = exp(L_CE) = exp(-(1/T) ∑ₜ log P(xₜ))
```

- PPL 越低 → 模型越好
- PPL = 词表大小 → 随机猜测
- GPT-4 级模型在通用文本上 PPL ≈ 3-5

---

## 三、反向传播 (Backpropagation)

### 3.1 链式法则

```
∂L/∂W₁ = (∂L/∂ŷ) · (∂ŷ/∂h₂) · (∂h₂/∂h₁) · (∂h₁/∂W₁)
```

- 从输出层 → 隐藏层 → 输入层，逐层反向计算梯度
- 每一层的梯度 = 上游梯度 × 本层的局部梯度

### 3.2 计算图视角

```
           Forward (→)
x ─── W₁ ──→ z₁ ──→ σ ──→ h₁ ─── W₂ ──→ z₂ ──→ σ ──→ ŷ ──→ L
                                                              │
           Backward (←)                                       │
∂L/∂W₁ ←── ∂z₁/∂W₁ ←── ∂σ/∂z₁ ←── ∂h₁ ←── ... ←── ∂L/∂ŷ ←┘
```

### 3.3 反向传播的数值例子

```python
# 极简两层网络的反向传播
import numpy as np

# 前向
x = np.array([1.0, 2.0])
W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
W2 = np.array([[0.5], [0.6]])

h = np.maximum(0, W1 @ x)    # ReLU: [0.5, 1.1]
y_hat = W2.T @ h              # 0.91
y_true = 1.0
loss = (y_hat - y_true) ** 2  # MSE: 0.0081

# 反向
dL_dy = 2 * (y_hat - y_true)          # -0.18
dL_dW2 = h.reshape(-1, 1) * dL_dy     # [[-0.09], [-0.198]]
dL_dh = W2 * dL_dy                     # [[-0.09], [-0.108]]
dL_dz = dL_dh * (h > 0).reshape(-1,1) # ReLU 梯度：正值为 1，负值为 0
dL_dW1 = dL_dz @ x.reshape(1, -1)     # 2x2 梯度矩阵
```

---

## 四、梯度下降与优化器

### 4.1 基础梯度下降

```
θ_{t+1} = θ_t - η · ∇L(θ_t)
```

| 变种 | 更新规则 | 特点 |
|------|---------|------|
| **SGD** | `θ -= η·g` | 简单但震荡 |
| **SGD + Momentum** | `v = βv + g; θ -= η·v` | 加速收敛 |
| **Adam** ⭐ | 自适应学习率 + 动量 | LLM 训练标配 |
| **AdamW** ⭐ | Adam + 权重衰减解耦 | 改进正则化 |

### 4.2 Adam 优化器核心公式

```
m_t = β₁·m_{t-1} + (1 - β₁)·g_t        # 一阶矩（动量）
v_t = β₂·v_{t-1} + (1 - β₂)·g_t²       # 二阶矩（自适应学习率）
m̂_t = m_t / (1 - β₁^t)                  # 偏差修正
v̂_t = v_t / (1 - β₂^t)
θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)
```

- 默认超参：`β₁=0.9, β₂=0.999, ε=1e-8`
- AdamW 额外添加：`θ_{t+1} -= η·λ·θ_t`（权重衰减）

### 4.3 学习率调度 (Learning Rate Schedule)

```
           ↑ lr
  warmup   │  /‾‾‾‾‾‾\
           │ /          \  cosine decay
           │/              \
           └──────────────────→ step
           0    warmup     total
```

```python
# Cosine Schedule with Warmup —— LLM 预训练标准配置
def get_lr(step, warmup_steps=2000, max_steps=100000, max_lr=3e-4, min_lr=3e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps  # 线性预热
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

---

## 五、激活函数

| 激活函数 | 公式 | 特点 | 使用场景 |
|---------|------|------|---------|
| **ReLU** | `max(0, x)` | 简单高效，但有"死神经元" | 经典网络 |
| **GELU** | `x · Φ(x)` | 平滑版 ReLU | BERT, GPT |
| **SiLU (Swish)** | `x · σ(x)` | 平滑，无死区 | Llama, Qwen |
| **SwiGLU** ⭐ | `Swish(xW₁) ⊙ (xW₂)` | 门控 FFN | Llama-2/3, Qwen, DeepSeek |
| **Sigmoid** | `1/(1+e^{-x})` | 输出 (0,1) | 门控、二分类 |
| **Softmax** | `e^{xᵢ} / ∑e^{xⱼ}` | 输出概率分布 | Attention、分类输出 |

### SwiGLU FFN 实现

```python
class SwiGLUFFN(nn.Module):
    """Llama/Qwen/DeepSeek 使用的 FFN"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # down
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # up

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

---

## 六、梯度问题与解决方案

### 6.1 梯度消失 / 梯度爆炸

| 问题 | 原因 | 症状 | 解决方案 |
|------|------|------|---------|
| 梯度消失 | 多层 sigmoid/tanh 连乘 → 梯度趋近 0 | 底层不更新 | ReLU/GELU、残差连接、LayerNorm |
| 梯度爆炸 | 梯度过大导致参数更新过猛 | loss = NaN | 梯度裁剪、学习率减小 |

### 6.2 残差连接 (Residual Connection) ⭐

```
output = LayerNorm(x + SubLayer(x))
```

- 梯度直通路径：`∂L/∂x = ∂L/∂output · (1 + ∂SubLayer/∂x)`
- 即使 SubLayer 梯度为 0，仍有梯度 1 通过 → 解决梯度消失
- Transformer 每个子层（Attention、FFN）都有残差连接

### 6.3 梯度裁剪 (Gradient Clipping)

```python
# LLM 训练标配
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 七、完整训练循环

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. 模型
model = SimpleTransformer(vocab_size=32000, d_model=512, n_layers=6)
model = model.cuda()

# 2. 优化器 + 调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

# 3. 损失函数
criterion = nn.CrossEntropyLoss(ignore_index=-100)  # -100 为 padding

# 4. 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()       # (B, L)
        labels = batch['labels'].cuda()             # (B, L)

        # 前向
        logits = model(input_ids)                   # (B, L, V)
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

        # 反向
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新参数
        optimizer.step()
        scheduler.step()

        print(f"loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.2e}")
```

---

## 八、关键数值直觉

| 指标 | 正常范围 | 异常信号 |
|------|---------|---------|
| 训练 Loss | 逐步下降 | 突然飙升 → 学习率过大或数据问题 |
| 梯度范数 | 0.1 - 10 | >100 → 梯度爆炸，<0.001 → 梯度消失 |
| 学习率 | 1e-5 ~ 3e-4 | SFT 通常 1e-5~2e-5；预训练 1e-4~3e-4 |
| PPL | 下降趋势 | 回升 → 过拟合 |
| 权重范数 | 缓慢增长 | 爆炸式增长 → 需要加正则 |

---

## 面试高频问答

**Q1：为什么 Transformer 不用 ReLU 而用 GELU/SwiGLU？**
> ReLU 有"死神经元"问题（输入为负时梯度永远为 0）。GELU/SiLU 是平滑的，允许负值有微小梯度。SwiGLU 额外引入了门控机制（`gate ⊙ value`），让网络能学习性地控制信息流，实验证明效果优于单纯的 GELU FFN。

**Q2：Adam 和 SGD 的核心区别？**
> Adam 通过一阶矩（动量）和二阶矩（自适应学习率）两个维度来调节每个参数的更新步长。SGD 对所有参数使用同一个学习率，而 Adam 为梯度小的参数给大学习率、梯度大的参数给小学习率。LLM 训练几乎都用 AdamW。

**Q3：为什么需要残差连接？**
> 随着层数增加，梯度在反向传播中逐层相乘可能趋近于 0（梯度消失）。残差连接提供了一条"梯度高速公路"，使梯度可以直接传递到底层。没有残差连接，100+ 层的 Transformer 根本无法训练。

## 面试一句话
- "前向传播是矩阵乘法 + 非线性的堆叠，反向传播是链式法则逐层求梯度，AdamW 是当前 LLM 训练的标准优化器，残差连接和 LayerNorm 解决了深层网络的梯度传播问题。"
