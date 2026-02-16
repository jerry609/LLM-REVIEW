# RoPE / 位置编码数学速查

## 1) 为什么需要位置编码
- 注意力机制本身是置换不变的（permutation invariant）
- 无位置信息则 `{A,B,C}` 和 `{C,A,B}` 得到相同结果
- 位置编码让模型感知 token 的顺序

## 2) 绝对正弦位置编码（原始 Transformer）
- `PE(pos, 2i) = sin(pos / 10000^{2i/d_model})`
- `PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})`
- 直接加到 embedding 上：`x = token_embedding + PE`
- 局限：不能直接外推到训练时没见过的位置

## 3) RoPE 核心形式
- 对 Q 和 K 的每个 head 的每对维度做二维旋转：
  ```
  theta_i = base^{-2i/d_head}    (base 常见 10000)

  对位置 m 的第 i 对维度：
  [q_{2i}', q_{2i+1}'] = [q_{2i}*cos(m*theta_i) - q_{2i+1}*sin(m*theta_i),
                           q_{2i}*sin(m*theta_i) + q_{2i+1}*cos(m*theta_i)]
  ```
- 等价于：`q_m' = R(m) q_m`，`R(m)` 是块对角旋转矩阵

## 4) RoPE 的关键性质
- **相对位置感知**：`<q_m', k_n'> = <R(m)q, R(n)k> = q^T R(m-n) k`
  - 内积只依赖相对位置 `m-n`
- **不引入额外参数**：直接作用于 Q 和 K
- **不作用于 V**：V 保持原始值
- **长度外推**：理论上可外推，但实际精度随超出训练长度衰减

## 5) 频率直觉
- 低维（i 小）→ `theta_i` 大 → 旋转快 → 捕捉短距离关系
- 高维（i 大）→ `theta_i` 小 → 旋转慢 → 捕捉长距离关系
- 类比傅里叶变换：不同频率分量编码不同尺度的位置信息

## 6) 长上下文扩展方法

### Position Interpolation（PI）
- 将位置缩放到训练范围内：`pos' = pos * L_train / L_target`
- 简单但可能损失分辨率

### NTK-aware RoPE
- 调整 base：`base_new = base * (L_target/L_train)^{d/(d-2)}`
- 高频分量保持分辨率，低频分量延展
- 不需要微调，"free lunch" 性质

### YaRN（Yet another RoPE extensioN）
- 结合 NTK 缩放 + 注意力温度调节 + 少量微调
- 对不同频率分量分组处理
- 效果优于纯 PI 或 NTK

### Dynamic NTK
- 根据当前序列长度动态调整 base
- 短序列不变，超出训练长度时才缩放

## 7) ALiBi（Attention with Linear Biases）
- 不修改 Q/K，在注意力分数上加线性偏置：
  `S_{ij} = Q_i K_j^T - m * |i - j|`
- `m` 是每个 head 的斜率（几何序列）：
  `m_h = 2^{-8h/H}`（h = 1,...,H）
- 优势：零参数，天然支持长度外推
- 劣势：线性偏置可能不如 RoPE 灵活

## 8) 各方法对比
| 方法 | 额外参数 | 作用位置 | 长度外推 | 主流使用 |
|------|---------|---------|---------|---------|
| 正弦 PE | 0 | embedding | 差 | GPT-2 |
| 可学习 PE | T*d | embedding | 差（受限于训练长度）| GPT-3 |
| RoPE | 0 | Q, K | 可扩展 | LLaMA, Qwen |
| ALiBi | 0 | attention score | 好 | BLOOM, MPT |

## 面试一句话
- "RoPE 把绝对位置编码成相位旋转，让注意力天然感知相对位移；长度外推通过调 base 频率实现，本质是在分辨率和覆盖范围间权衡。"
