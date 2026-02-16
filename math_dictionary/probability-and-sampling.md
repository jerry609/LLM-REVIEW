# 概率与采样速查

## 1) softmax 与温度
- `p_i = exp(z_i/tau) / sum_j exp(z_j/tau)`
- `tau` 变小：分布更尖锐，输出更确定。
- `tau → 0`：退化为 argmax（贪婪解码）
- `tau → ∞`：退化为均匀分布

## 2) Top-k / Top-p 采样
- Top-k：只在概率最高的 k 个 token 中重新归一化后采样。
- Top-p（nucleus）：选择累计概率达到 `p` 的最小集合 `S`：
  `S = argmin_{|S'|} { sum_{i in S'} p_i >= p }`
  然后在 `S` 中重新归一化后采样。
- 两者可联合使用：先 Top-k 过滤，再 Top-p 截断。
- Min-p（新方法）：过滤掉概率 < `min_p * p_max` 的 token，动态调整候选集大小。

## 3) 贪婪解码 vs 采样
- 贪婪：每步取 `argmax p_i`，确定性，可能陷入重复
- 采样：引入随机性，结果更多样但可能不一致
- Beam search：保留 top-k 个候选序列
  - 分数：`score(y) = sum_t log p(y_t|y_{<t})`
  - 长度惩罚：`score_normalized = score(y) / len(y)^alpha`

## 4) 熵（不确定性）
- `H(p) = -sum_i p_i log p_i`
- 熵高：模型更不确定（输出多样性大）
- 熵低：更确定但可能更保守
- 最大熵 = `log(V)`（均匀分布时）
- 可用于动态调节温度：熵低时加大 `tau`，避免过于保守

## 5) KL 散度（两分布间的距离）
- `KL(p||q) = sum_i p_i log(p_i / q_i)`
- 非对称：`KL(p||q) ≠ KL(q||p)`
- 用途：
  - 评估量化/蒸馏后输出分布偏移
  - RLHF 中约束策略不偏离参考模型太远：
    `objective = E[reward] - beta * KL(pi||pi_ref)`
- `KL >= 0`，当且仅当 `p = q` 时取等

## 6) 困惑度与交叉熵的关系
- 交叉熵：`CE = -1/N * sum_t log p(x_t|x_{<t})`
- 困惑度：`PPL = exp(CE) = exp(-1/N * sum_t log p(x_t|x_{<t}))`
- PPL 可理解为"平均每步有效候选数"
- PPL = 10 意味着模型平均对每个位置的不确定性等价于在 10 个等概率选项中选择

## 7) 复读惩罚
- 频率惩罚：对已出现 token 的 logit 减去 `alpha * count(token)`
- 存在惩罚：对已出现 token 的 logit 减去 `beta`（不管出现几次）
- 两者控制不同行为：频率惩罚减少高频重复，存在惩罚鼓励多样性

## 8) Speculative Decoding 中的接受概率
- draft model 提出候选 token 序列，target model 验证
- 接受概率：`P_accept = min(1, p_target(x) / p_draft(x))`
- 被拒绝时从修正分布中重新采样：`p_resample ∝ max(0, p_target(x) - p_draft(x))`
- 保证最终分布与 target model 完全一致（无损加速）
- 加速比 ∝ `1 / (1 - avg_acceptance_rate)`（理想情况）

## 面试一句话
- "采样参数影响的是分布形状，不是模型能力本身；选择策略时要在多样性和一致性间权衡。"
