# LoRA 与 RLHF

## LoRA (Low-Rank Adaptation)
### 核心思想
- 冻结预训练权重 W，只训练低秩分解 ΔW = BA
- A ∈ R^{d×r}, B ∈ R^{r×d}，r << d
- 推理时合并：W' = W + (α/r)BA → 无额外延迟

### 关键参数
| 参数 | 含义 | 推荐值 |
|------|------|--------|
| r | 秩 | 8-64 |
| α | 缩放因子 | 通常 = r 或 2r |
| target_modules | 哪些层加 LoRA | q_proj, k_proj, v_proj, o_proj |

### LoRA 变体
- **QLoRA**：4-bit 量化底座 + LoRA → 单卡微调 70B
- **DoRA**：分解方向和幅度，效果更好
- **LoRA+**：不同学习率给 A 和 B

## RLHF (从人类反馈中强化学习)
### 三阶段流水线
1. **SFT**：在人类标注对话上监督微调
2. **RM**：训练奖励模型（对比优劣回复）
3. **PPO**：用奖励模型的分数做 RL 优化

### DPO (Direct Preference Optimization)
- 跳过奖励模型，直接从偏好数据优化策略
- 公式：`L_DPO = -log σ(β(log π/π_ref(y_w) - log π/π_ref(y_l)))`
- 更稳定、更简单，已成为主流

## 面试一句话
- "LoRA 以极低参数量实现接近全量微调的效果；DPO 简化了 RLHF 流水线，从三阶段减到两阶段。"
