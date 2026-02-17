# Reasoning Models（推理模型）全景解析

## 一、概览：Test-Time Compute Scaling

### 1.1 核心思想
- **传统 scaling**：增加模型参数 / 训练数据 / 训练计算
- **Test-time scaling**：增加**推理时**的计算量来提升质量
- **方式**：让模型生成更长的思考过程（thinking tokens）

### 1.2 Scaling Law
传统：Performance 正比于 log(Train Compute)
推理时：Performance 正比于 log(Inference Compute)
        = log(thinking_tokens * model_flops_per_token)

---

## 二、主要推理模型对比

| 维度 | OpenAI o1/o3 | DeepSeek-R1 | Kimi k1.5 | QwQ |
|------|-------------|-------------|-----------|-----|
| 发布 | 2024.09/2025.01 | 2025.01 | 2025.01 | 2024.11 |
| 基座 | GPT-4 系列 | DeepSeek-V3 (MoE) | 未公开 | Qwen2.5-32B |
| 训练 | PRM-based RL | GRPO (rule reward) | RL + Long2Short | SFT + RL |
| 多模态 | 否 (o1), 是 (o3) | 否 | 是 | 否 |
| 开源 | 否 | 完全 | 否 | 权重 |
| 特色 | 首创 | 纯 RL 涌现 | 长 context RL | 轻量级 |
| 推理成本 | 极高 | 中 (MoE) | 高 | 低 (32B) |

---

## 三、技术路线对比

### 3.1 Process Reward Model (PRM) 路线 [OpenAI]
训练 PRM:
  - 收集人工标注的推理步骤评价
  - 每步标注 good/bad/neutral
  - 训练 reward model 给每步打分

RL 训练:
  - Policy 生成推理步骤
  - PRM 给每步 reward
  - PPO 更新 policy

- **优势**：过程监督 -> 推理质量高
- **劣势**：PRM 标注成本极高

### 3.2 Rule-based Reward 路线 [DeepSeek R1]
RL 训练:
  - 对数学/代码题：答案正确 -> reward=1，错误 -> reward=0
  - 格式 reward：遵循 <think>...</think> 格式 -> bonus
  - GRPO：组内相对排序，不需要 critic

- **优势**：零标注成本，可规模化
- **劣势**：仅适用于有标准答案的任务

### 3.3 Distillation 路线 [QwQ, R1-Distill]
- 从强推理模型采样大量推理轨迹
- 用这些轨迹 SFT 小模型
- 可选：再做一轮 RL 微调

- **优势**：成本最低，效果出奇好
- **发现**：蒸馏 > 直接对小模型做 RL

---

## 四、Test-Time Compute 工程化

### 4.1 Inference Optimization
- **挑战**：thinking tokens 可能很长（数万 tokens）
- **Speculative decoding**：用小模型加速 thinking 部分
- **Early stopping**：如果已经得出答案，截断 thinking
- **Adaptive compute**：简单问题少 think，难题多 think
  - 通过 prompt 控制："Think briefly" vs "Think step by step"

### 4.2 成本分析
假设 reasoning model 平均生成 10K thinking tokens + 1K output tokens:
- 传统模型：1K output tokens
- 推理模型：11K total tokens（11x 计算量）

成本优化方向：
1. 蒸馏到小模型（R1-Distill-7B）
2. MoE 减少激活参数
3. 简单任务走 fast path（不开 reasoning）
4. Speculative decoding 加速

### 4.3 Serving 设计
- **路由策略**：根据 query 复杂度决定是否启用 reasoning
- **预算控制**：设置 max_thinking_tokens
- **流式输出**：thinking 过程可以流式展示（如 DeepSeek 网页版）

---

## 五、面试高频问答

**Q: 为什么 reasoning model 不直接增大模型参数，而是增加推理时间？**
- 增大参数受 hardware 和成本限制（训练+推理）
- 增加推理时间的边际成本更低
- 且可以按需 scaling：简单问题省计算，难题加计算

**Q: PRM vs Rule-based Reward 应该选哪个？**
- 有标准答案的任务（数学/代码）-> Rule-based（零标注成本）
- 开放任务（写作/分析）-> PRM（需要过程质量评估）
- 实践中常混合使用

**Q: 推理模型会取代传统 LLM 吗？**
- 不会。推理模型成本 10x+，只适合需要深度思考的任务
- 简单对话、信息检索、创意写作不需要 reasoning
- 未来方向：智能路由，自动判断是否需要 reasoning

**Q: Self-verification 是怎么涌现的？**
- R1 的实验表明：纯 RL + rule reward -> 模型自发学会检验
- 原因：检验自己的答案可以提高 reward -> RL 强化了这个行为
- 不需要显式在训练数据中教
