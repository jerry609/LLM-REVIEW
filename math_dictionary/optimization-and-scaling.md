# 优化器、Scaling Law 与训练数学速查

## 1) 交叉熵损失
- `L = -(1/N) * sum_t log p_theta(x_t|x_{<t})`
- 训练目标是最大化正确 token 概率（最小化负对数似然）。

## 2) Adam 更新（简化）
- 一阶矩：`m_t = beta1*m_{t-1} + (1-beta1)*g_t`
- 二阶矩：`v_t = beta2*v_{t-1} + (1-beta2)*g_t^2`
- 偏差校正：`m_hat_t = m_t/(1-beta1^t)`，`v_hat_t = v_t/(1-beta2^t)`
- 参数更新：`theta_t = theta_{t-1} - lr * m_hat_t/(sqrt(v_hat_t)+eps)`
- 常见超参：`beta1=0.9, beta2=0.95, eps=1e-8`（LLM 训练常用 `beta2=0.95`）

## 3) AdamW（权重衰减）
- 与 Adam 区别：权重衰减直接作用于参数，而非梯度
- `theta_t = theta_{t-1} - lr * (m_hat_t/(sqrt(v_hat_t)+eps) + lambda * theta_{t-1})`
- `lambda`：权重衰减系数，常为 0.1

## 4) 学习率调度
- Warmup：前 `T_warmup` 步线性增长 `lr: 0 → lr_max`
- Cosine decay：
  `lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * (t-T_warmup)/(T_total-T_warmup)))`
- 常见设置：warmup 占总步数的 1-5%，lr_min = lr_max * 0.1

## 5) 梯度累积
- 当显存不足以放下目标 batch size 时：
  `effective_batch_size = micro_batch_size * accumulation_steps * DP_size`
- 每 `accumulation_steps` 步做一次参数更新
- 数学上等价于大 batch（忽略 BN 等），但训练时间不变

## 6) Scaling Law（Chinchilla 版）
- 损失随模型规模、数据规模呈幂律下降：
  `L(N, D) ≈ A/N^alpha + B/D^beta + L_irr`
  - `N`：参数量，`D`：训练 token 数，`L_irr`：不可约损失
  - 典型值：`alpha ≈ 0.34, beta ≈ 0.28`
- **Chinchilla 最优**：给定 compute budget `C`
  - `C ≈ 6 * N * D`（近似 FLOPs）
  - 最优配比：`D ≈ 20 * N`（即训练 token 数 ≈ 20× 参数量）
  - 例：7B 模型 → 理论最优约 140B token
- 面试重点："预算固定时如何在模型/数据/训练步数间平衡"

## 7) 推理 Scaling（Inference-time Compute Scaling）
- 思路：在推理时通过更多计算提升质量
- 方法：
  - Best-of-N：生成 N 个回答，用 verifier 选最好的
  - Chain-of-thought：更长的推理链 → 更多 token → 更多 FLOPs
  - Tree search / MCTS：搜索更大的候选空间
- 推理 FLOPs 与质量的关系也呈现类 scaling law 的 log-linear 关系

## 8) 训练 FLOPs 估算
- 前向传播：`≈ 2 * N` FLOPs per token（`N` 为参数量）
- 反向传播：`≈ 4 * N` FLOPs per token（约前向的 2 倍）
- 总训练 FLOPs：`C ≈ 6 * N * D`
  - 例：7B 模型 × 1T token ≈ `6 * 7e9 * 1e12 = 4.2e22 FLOPs`
  - A100 (312 TFLOPS BF16)：`4.2e22 / 312e12 ≈ 1.35e8 秒 ≈ 1560 GPU-days`
  - 考虑 MFU（Model FLOPs Utilization）~50%：实际约 3120 GPU-days

## 9) MFU（Model FLOPs Utilization）
- `MFU = actual_model_flops / (peak_flops * time)`
- 衡量训练效率（排除通信、数据加载等开销后的有效利用率）
- 优秀：>50%，良好：30-50%，需优化：<30%

## 10) 与推理的关系
- 训练决定上限，推理系统决定是否把上限稳定交付给用户。
- Scaling law 指导选模型大小；推理优化决定成本可行性。
- 量化/蒸馏的目标：用更少的推理成本逼近更大模型的质量。

## 面试一句话
- "Chinchilla 告诉我们参数和数据要同步 scale；推理 scaling 则说明测试时算力也能换质量。"
