# Test-Time Compute Scaling 深度解析

## 一、核心概念

### 1.1 什么是 Test-Time Compute
- **Train-time compute**：预训练和微调消耗的计算
- **Test-time compute**：推理时消耗的计算
- **Scaling 发现**：增加推理计算量 -> 性能持续提升

### 1.2 两种 Test-Time Scaling 方式

#### 方式一：串行 Scaling（Sequential）
- 让模型"想更久"：生成更长的 Chain-of-Thought
- 典型：o1, R1, k1.5
- Compute 正比于 thinking_tokens
- 受限于序列长度和 KV Cache

#### 方式二：并行 Scaling（Parallel）
- 生成多个候选答案，选最好的
- 典型：Best-of-N, Majority Voting
- Compute 正比于 num_samples
- 需要好的 verifier / reward model

### 1.3 Optimal Compute Allocation
- 给定固定 test-time compute budget，如何分配？
- 研究发现（Snell et al., 2024）：
  - 简单问题：revision（修正自己）更高效
  - 困难问题：parallel sampling（多次尝试）更高效
  - 最优策略：根据难度动态分配

---

## 二、实现技术

### 2.1 Sequential：Long Chain-of-Thought
- **训练**：RL 让模型学会何时深入思考、何时回溯
- **控制**：通过 system prompt 或 temperature 控制思考深度

### 2.2 Parallel：Best-of-N Sampling
- **Verifier 选择**：
  - ORM (Outcome Reward Model)：只看最终结果
  - PRM (Process Reward Model)：看每步质量
  - Self-consistency：多数投票（无需额外模型）

### 2.3 Tree Search
- **Monte Carlo Tree Search (MCTS)**：
  - 把推理过程建模为树搜索
  - 每个节点 = 一个推理步骤
  - 用 reward model 评估节点价值
- **Beam Search over CoT**：
  - 保留 top-K 条推理路径
  - 每步扩展最有希望的路径

---

## 三、面试高频问答

**Q: Test-time compute scaling 的上限在哪？**
- 串行：受限于 context length 和 KV Cache
- 并行：受限于 verifier 质量（bad verifier -> diminishing returns）
- 两者结合可以推得更远，但成本增长很快

**Q: 怎么判断一个问题需要多少 test-time compute？**
- **难度估计**：用小模型快速尝试，看 confidence
- **Adaptive**：先短 CoT，如果 confidence 低再加长
- **Router**：训练一个轻量分类器判断难度级别

**Q: Self-consistency vs Best-of-N 的区别？**
- Self-consistency：多次采样 -> majority vote（不需要 reward model）
- Best-of-N：多次采样 -> reward model 选最好的
- Best-of-N 通常更好，但需要额外的 reward model
