# 对齐流水线 (Alignment Pipeline)

## 完整流程
```
预训练 → SFT → 偏好对齐(RLHF/DPO) → 安全对齐(Red Teaming) → 部署
```

## 各阶段目标
| 阶段 | 输入 | 输出 | 核心指标 |
|------|------|------|---------|
| 预训练 | 海量文本 | base model | PPL, benchmark |
| SFT | 对话数据 | chat model | 指令跟随能力 |
| 偏好对齐 | 人类偏好数据 | aligned model | win rate, reward score |
| 安全对齐 | 对抗样本 | safe model | ASR (Attack Success Rate) |

## SFT 实践
- 数据量：1K-100K 高质量对话即可（少而精 > 多而杂）
- 格式：多轮对话，标注 assistant 回复
- LoRA SFT：单卡即可微调 70B

## RLHF vs DPO vs ORPO
| 方法 | 需要 RM | 训练稳定性 | 效果 |
|------|---------|----------|------|
| RLHF (PPO) | ✅ | 难调 | 强 |
| DPO | ❌ | 稳定 | 接近 RLHF |
| ORPO | ❌ | 最稳定 | 略弱 |

## Constitutional AI (Anthropic)
- 用 AI 自己做 red teaming + 修改回复
- 减少人工标注成本
- 迭代：生成→评价→修改→训练

## 面试一句话
- "对齐的关键是 SFT 给格式、RLHF/DPO 给偏好、Red Teaming 给安全。数据质量远比数量重要。"
