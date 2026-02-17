# 面试题：推理模型 (Reasoning Models) 深度解析

## 题目
"请介绍一下 DeepSeek-R1 的训练方法。它和 OpenAI o1 有什么核心区别？在推理服务中如何优化 reasoning model 的成本？"

## 参考答案

### 第一部分：DeepSeek-R1 训练方法

R1 的训练分为 4 个阶段：

**Stage 1: Cold Start SFT**
- 用少量（数千条）long chain-of-thought 数据做 SFT
- 目的是让模型学会"展开推理"的格式

**Stage 2: Reasoning-oriented RL（核心创新）**
- 使用 GRPO (Group Relative Policy Optimization) 算法
- **不使用 PRM/ORM**，只用规则 reward：
  - 数学题：答案正确 r=1，错误 r=0
  - 代码题：test case 通过数
  - 格式 reward：遵循 <think> 格式 bonus
- GRPO 核心：对同一 prompt 采样 G 个回答，组内归一化计算 advantage
  - A_i = (r_i - mean(r)) / std(r)
  - 不需要 critic network（比 PPO 省一半显存）
- **关键发现**：模型自发涌现 self-verification、reflection 行为

**Stage 3: Rejection Sampling + SFT**
- 从 Stage 2 模型采样，筛选高质量推理轨迹
- 混合通用 SFT 数据，恢复非推理能力

**Stage 4: 全任务 RL**
- 推理任务 + helpfulness + safety 多目标 RL

### 第二部分：与 o1 的核心区别

| 维度 | DeepSeek-R1 | OpenAI o1 |
|------|-------------|-----------|
| Reward | 规则 reward（零标注） | PRM（大量人工标注） |
| RL 算法 | GRPO（无 critic） | PPO（有 critic） |
| 推理涌现 | 纯 RL 涌现 | 监督引导 |
| 开源 | 完全开源 | 闭源 |
| 训练成本 | 更低 | 更高 |

核心区别在于 **reward 的来源**。o1 依赖大量人工标注的过程 reward，R1 证明了纯规则 reward + RL 就能涌现推理能力。

### 第三部分：推理服务成本优化

1. **蒸馏到小模型**：R1-Distill-Qwen-32B 接近 o1-mini 水平
2. **Adaptive thinking**：简单问题不启用 reasoning（路由器判断难度）
3. **Speculative decoding**：用小模型加速 thinking tokens 生成
4. **Early stopping**：检测到答案后截断剩余 thinking
5. **MoE 架构**：R1 基于 DeepSeek-V3，每次推理只用 5.5% 参数

## 加分点
- 提到 GRPO 的数学公式
- 提到 "Aha moment"（模型突然自我修正的涌现现象）
- 对比蒸馏 vs 直接 RL 小模型的效果差异
