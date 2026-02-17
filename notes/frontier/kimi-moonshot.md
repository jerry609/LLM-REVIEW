# Kimi (月之暗面 Moonshot AI) 技术深度解析

## 一、Kimi k1.5（2025.01 发布）

### 1.1 核心定位
- 多模态推理模型（类 o1/R1）
- 支持 **文本 + 视觉** 联合推理
- 在多个 benchmark 达到 o1 水平

### 1.2 关键技术创新

#### Long Context Scaling for RL
- **核心洞见**：推理模型的 CoT 越长 -> 推理质量越高（test-time compute scaling）
- **挑战**：RL 训练时 context window 有限（通常 8K-16K）
- **Kimi 方案**：
  - 训练时 context 从 8K 逐步扩展到 **128K**
  - Partial rollout：不从头生成，从中间状态继续
  - 这让模型学会更长的推理链

#### Reinforcement Learning 方法
- **Long2Short**：
  - Phase 1：训练 long-thinking 模型（128K context RL）
  - Phase 2：用 long model 蒸馏 short model
  - 短模型学习"压缩版"推理，性能保持、速度提升
- **Reward 设计**：
  - Rule-based reward（数学/代码）
  - Model-based reward（开放任务）
  - Length penalty：防止无意义重复

#### 多模态推理
- 图表/图片中的数学问题
- 空间推理（几何、三维）
- 视觉 + 文本联合推理链

### 1.3 与其他推理模型对比
| 维度 | Kimi k1.5 | DeepSeek-R1 | OpenAI o1 |
|------|-----------|-------------|-----------|
| 模态 | 文本+视觉 | 纯文本 | 纯文本 |
| 训练方法 | RL + Long2Short | 4-stage GRPO | PRM-based |
| 长 context RL | 支持 128K | 不支持 | 未知 |
| 开源 | 部分 | 完全 | 闭源 |

---

## 二、长上下文技术（Kimi 核心竞争力）

### 2.1 超长上下文处理
- 支持 **200 万 tokens**（2M context window）
- 工程实现：
  1. **Sliding Window + Global Attention**：
     - 底层用 sliding window（局部 attention）
     - 顶层用 global attention（全局信息汇聚）
  2. **KV Cache 分层存储**：
     - Hot KV：GPU HBM
     - Warm KV：CPU DRAM
     - Cold KV：NVMe SSD
  3. **Prefill 优化**：
     - Chunk prefill：将 2M tokens 分 chunk 处理
     - 流水线化：chunk N 的 attention 和 chunk N+1 的 load 重叠

### 2.2 长文本理解能力
- **Needle-in-a-Haystack**：2M context 全通过
- **Long-Document QA**：多文档综合分析
- **长代码理解**：整个 repo-level 的代码分析

### 2.3 面试考点
**Q: 如何支持 2M tokens 的 KV Cache？**
- 假设 GQA-8, d_model=4096, float16：
  - KV Cache/token 约等于 2 * 8 * 64 * 2 bytes * n_layers
  - 32 层: 2M tokens 约 131GB
- 解决：分层存储 + offloading + KV compression

**Q: 长上下文推理的 latency 问题？**
- Prefill: O(n^2) -> 2M tokens 极其慢
- 方案：Ring Attention (序列并行) + Sparse Attention pattern
- Decode 还行，因为每步只算 O(n)

---

## 三、Moonshot 公司技术栈

### 3.1 推理基础设施
- 自研推理框架（类 vLLM 但针对长上下文优化）
- 动态 batch + preemption
- 多级缓存 (prefix caching + KV offloading)

### 3.2 面试常见角度
- 长上下文场景下的 attention 优化
- KV Cache 管理和调度
- 如何在保证质量的前提下减少推理成本
