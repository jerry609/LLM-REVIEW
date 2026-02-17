# 面试题：国产大模型技术对比

## 题目
"请对比 DeepSeek-V3、Qwen2.5、GLM-4、Kimi、MiniMax-01 的核心技术差异。如果要设计推理服务，选哪个模型？"

## 参考答案

### 架构对比表

| 维度 | DeepSeek-V3 | Qwen2.5-72B | GLM-4 | Kimi | MiniMax-01 |
|------|-------------|-------------|-------|------|------------|
| 参数量 | 671B (MoE) | 72B (Dense) | ~100B | 未公开 | 456B (MoE) |
| 激活参数 | 37B | 72B | ~100B | - | 45.9B |
| Attention | MLA | GQA | GQA | 未公开 | Softmax+Linear混合 |
| 上下文 | 128K | 128K (1M Turbo) | 128K (1M Long) | 2M | 4M |
| MoE | 256 experts, 8 active | 无 | 无 | 未公开 | 32 experts, 2 active |
| 特色 | MLA+MTP | YARN外推 | Visual Expert | 长上下文 | Lightning Attention |

### 推理服务选型建议

**通用对话/短文本** -> Qwen2.5-72B（Dense，serving 简单）
**数学/编程推理** -> DeepSeek-R1 蒸馏版（推理能力强，成本低）
**长文档理解** -> Kimi or MiniMax-01（2M+ context）
**成本敏感** -> DeepSeek-V3（MoE + MLA，综合性价比最高）

## 加分点
- 能说清 MLA 的压缩原理（低秩 latent projection）
- 能对比 Linear Attention vs Softmax Attention 的 trade-off
- 提到模型选型需要考虑 serving 基础设施的支持程度
