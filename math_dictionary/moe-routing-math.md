# MoE（Mixture of Experts）路由与负载数学速查

## 1) MoE 架构概述
- 标准 Transformer FFN 替换为多个专家（Expert）FFN
- 每个 token 只激活 `E_active` 个专家（从 `E` 个中选）
- 例：Mixtral 8x7B → `E=8, E_active=2`，每 token 只用 2 个 expert

## 2) 门控函数（Router）
- 输入：token 的隐藏态 `x in R^{d_model}`
- 门控 logit：`g = x W_gate`，`W_gate in R^{d_model x E}`
- 路由概率：`p = softmax(g)`
- 选择 top-k：`S = TopK(p, E_active)`
- 加权输出：`output = sum_{i in S} p_i * Expert_i(x)`

## 3) 参数量分析
- Dense 等价参数量：`E * params_per_expert + shared_params`
- 活跃参数量：`E_active * params_per_expert + shared_params`
- Mixtral 8x7B：
  - 总参数约 47B（8 个 expert × ~5.6B + 共享的 attention ~1.2B per layer）
  - 每 token 活跃参数约 13B（2 expert）
  - 推理速度接近 13B dense 模型，质量接近更大的 dense 模型

## 4) FLOPs 分析
- 每 token FFN FLOPs = `E_active * FLOPs_per_expert`
- Attention FLOPs 不变（attention 是共享的，非 MoE）
- 总 FLOPs ≈ dense 模型中 FFN 部分 × `E_active/1` + Attention 部分不变
- 相比等参数量 dense 模型，FLOPs 显著降低

## 5) 负载均衡损失（Auxiliary Loss）
- 目标：让每个 expert 被选中的频率大致均匀
- 辅助损失：
  ```
  L_balance = E * sum_{i=1}^{E} f_i * P_i
  ```
  - `f_i = (1/N) * sum_tokens 1[token routed to expert i]`（实际选择频率）
  - `P_i = (1/N) * sum_tokens p_i(token)`（平均路由概率）
- 当所有 expert 均匀被选时，`L_balance` 最小
- 最终损失：`L_total = L_language + alpha * L_balance`，`alpha` 通常为 0.01

## 6) Expert Capacity（容量限制）
- 每个 expert 每 batch 最多处理 `capacity` 个 token：
  `capacity = capacity_factor * (B * T / E)`
- `capacity_factor` 通常 1.0-1.5
- 超出容量的 token 被丢弃（dropped）或路由到备选 expert
- 丢弃比例过高 → 质量下降；容量过大 → 内存浪费

## 7) 显存分析（推理）
- 所有 expert 的权重都需要加载（即使每 token 只用部分）
- 权重显存 ≈ `E * expert_params * s + shared_params * s`
- KV cache 不受 MoE 影响（attention 是共享的）
- 若用 Expert Parallelism：每张卡只存 `E/EP` 个 expert

## 8) Expert Parallelism (EP) 通信
- All-to-All 通信：
  1. 每个 GPU 将 token 路由到持有目标 expert 的 GPU
  2. 目标 GPU 上的 expert 处理
  3. 结果 All-to-All 返回
- 通信量：每 token `d_model * s` 字节，双向
- 总通信量 ∝ `B * T * d_model * s * 2`

## 9) MoE + TP/PP 组合
- TP + EP：在 expert 内部做 tensor 并行（单 expert 太大时）
- PP + EP：不同 pipeline stage 的 MoE 层各自做 EP
- 实际部署常见：8-GPU 节点内 TP=8，节点间 EP

## 面试一句话
- "MoE 用稀疏激活换算力效率：总参数大但每 token 只用部分 expert，关键挑战是负载均衡和 All-to-All 通信。"
