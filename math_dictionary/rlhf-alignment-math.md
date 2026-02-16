# RLHF 与对齐数学速查

## 1) RLHF 三阶段流程
1. **SFT（Supervised Fine-Tuning）**：在人类标注数据上有监督微调
2. **RM（Reward Modeling）**：训练奖励模型，学习人类偏好排序
3. **PPO（Policy Optimization）**：用 RL 优化策略，最大化奖励同时不偏离参考模型

## 2) 奖励模型（Bradley-Terry 模型）
- 人类标注：对同一 prompt 的两个回答 `y_w`（胜）和 `y_l`（负）
- 偏好概率：`P(y_w > y_l) = sigma(r(x, y_w) - r(x, y_l))`
  - `sigma` 为 sigmoid 函数
- 训练损失：
  `L_RM = -E[log sigma(r(x, y_w) - r(x, y_l))]`

## 3) PPO 目标函数
- 优化目标：
  ```
  J(theta) = E_{x~D, y~pi_theta}[r(x,y)] - beta * KL(pi_theta || pi_ref)
  ```
  - `r(x,y)`：奖励模型分数
  - `beta`：KL 惩罚系数（防止策略偏离参考模型太远）
  - `pi_ref`：参考策略（通常是 SFT 模型）
- `beta` 太小 → reward hacking（过度优化奖励模型漏洞）
- `beta` 太大 → 学不到新东西（等于原策略）

## 4) DPO（Direct Preference Optimization）
- 跳过 RM 训练，直接从偏好数据优化策略
- 将最优策略表达为 reward 的闭式解：
  `r(x,y) = beta * log(pi_theta(y|x) / pi_ref(y|x)) + const`
- DPO 损失：
  ```
  L_DPO = -E[log sigma(beta * (log(pi_theta(y_w|x)/pi_ref(y_w|x))
                               - log(pi_theta(y_l|x)/pi_ref(y_l|x))))]
  ```
- 优势：不需要训练单独的 RM，不需要 RL（无 PPO），训练更稳定
- 本质：隐式定义了一个奖励函数

## 5) KTO（Kahneman-Tversky Optimization）
- 只需要 good/bad 标签，不需要成对偏好
- 基于前景理论（loss aversion）：
  `L_KTO = E_good[-log sigma(r)] + lambda * E_bad[-log sigma(-r)]`
- `lambda > 1`：对坏样本的惩罚更大（损失厌恶）

## 6) KL 散度在对齐中的角色
- `KL(pi_theta || pi_ref) = E_{y~pi_theta}[log(pi_theta(y|x) / pi_ref(y|x))]`
- 作为正则项，防止策略崩溃
- 实践中的近似：
  - 单样本估计：`KL ≈ log(pi_theta(y) / pi_ref(y))`（高方差）
  - Clipped 估计：PPO 中的 clip 操作隐式约束了 KL

## 7) RLHF vs DPO vs 其他方法对比
| 方法 | 需要 RM | 需要 RL | 数据需求 | 训练稳定性 |
|------|--------|--------|---------|-----------|
| RLHF (PPO) | 是 | 是 | 偏好对 | 较不稳定 |
| DPO | 否 | 否 | 偏好对 | 稳定 |
| KTO | 否 | 否 | good/bad | 稳定 |
| RLAIF | 是(AI标注) | 是 | AI 生成偏好 | 中等 |

## 面试一句话
- "RLHF 用人类偏好塑造模型行为，核心权衡是 reward 最大化 vs KL 约束：太贪心会 reward hack，太保守学不到新能力。DPO 用闭式解绕开了 RL 的不稳定性。"
