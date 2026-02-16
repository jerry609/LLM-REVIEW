# Transformer 注意力与核心组件公式

## 1) 注意力核心公式
- `Attention(Q,K,V) = softmax(QK^T / sqrt(d_head) + M) V`
- `M` 是 mask（因果 mask + padding mask）
- 缩放因子 `1/sqrt(d_head)` 的作用：防止点积随维度增大而方差膨胀，使 softmax 梯度更稳定。

## 2) 因果约束
- 自回归生成时，token `t` 只能看 `<= t` 的位置。
- 没有 KV cache 时，每步都要重算历史 `K,V` 投影；有 KV cache 时复用历史。

## 3) 多头注意力的完整流程
```
Q = X W_Q,  K = X W_K,  V = X W_V
Q_i = Q[:, :, i*d_head:(i+1)*d_head]   # 第 i 个 head
...（拆分 H 个 head）
head_i = Attention(Q_i, K_i, V_i)
MultiHead = Concat(head_1, ..., head_H) W_O
```

## 4) 数值稳定与温度
- 常见技巧：减去行最大值再做 softmax，避免溢出。
  - `softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`
- Online softmax（FlashAttention 使用）：流式计算，逐块更新 max 和 sum。
- 采样温度：`p_i = softmax(z_i / tau)`，`tau` 越小越"确定"。

## 5) LayerNorm
- `LayerNorm(x) = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta`
- `mu = mean(x)`，`sigma^2 = var(x)`，沿最后一维计算
- 可学习参数：`gamma`（缩放）、`beta`（偏移），各 `d_model` 维

## 6) RMSNorm（LLaMA 等常用）
- `RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)`
- 去掉了均值中心化（无 beta），只做缩放归一化
- 比 LayerNorm 少一次 reduce，推理更快

## 7) RoPE（旋转位置编码）
- 对 Q 和 K 的每个 head，将 `d_head` 维两两配对
- 对第 `m` 个位置、第 `i` 对：
  ```
  theta_i = 10000^{-2i/d_head}
  [q_{2i}', q_{2i+1}'] = [q_{2i}*cos(m*theta_i) - q_{2i+1}*sin(m*theta_i),
                           q_{2i}*sin(m*theta_i) + q_{2i+1}*cos(m*theta_i)]
  ```
- 关键性质：`<q_m, k_n>` 只依赖相对位置 `m-n`
- RoPE 不引入额外参数，直接编码到 Q/K 中
- 长度外推：NTK-aware RoPE、YaRN 等通过调整 `theta` 基底来扩展上下文

## 8) ALiBi（Attention with Linear Biases）
- 不修改 Q/K，直接在注意力分数上加位置偏置：
  `S_{ij} = Q_i K_j^T - m * |i - j|`
- `m` 是每个 head 不同的斜率（几何序列：`2^{-8/H}, 2^{-16/H}, ...`）
- 优势：零参数，天然支持长度外推

## 9) 残差连接
- `output = x + SubLayer(Norm(x))`（Pre-Norm 架构，LLaMA/GPT 常用）
- 或 `output = Norm(x + SubLayer(x))`（Post-Norm 架构，原始 Transformer）
- Pre-Norm 训练更稳定，是当前主流

## 面试高频口述
- "注意力的主要代价来自 QK 相关性计算和访存；KV cache 用显存换算力，减少重复投影。"
- "RoPE 的核心是把位置信息编码为旋转角度，使内积只依赖相对位置。"
- "RMSNorm 比 LayerNorm 少一次 reduce 操作，推理效率更高，效果接近。"
