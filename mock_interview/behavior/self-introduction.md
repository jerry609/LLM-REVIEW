# 自我介绍模板 & 项目表达话术

## 一、30 秒版（电话面/开场）

"面试官您好，我是 XXX。我过去 X 年主要专注在 LLM 推理优化和 Serving 系统方向。我深入研究过 KV Cache 管理（包括 PagedAttention、驱逐策略、量化压缩）、分布式推理（TP/PP/EP）、以及 Speculative Decoding 等核心技术。

我对底层实现有比较深的理解，从 FlashAttention 的 tiling 原理到 vLLM 的调度流程都有系统性的学习和代码实践。最近也关注了 DeepSeek-R1 的 GRPO 训练方法和 MLA 注意力机制。

我希望能在贵公司将这些技术应用到实际的推理服务中，提升 throughput 和降低成本。"

---

## 二、1 分钟版（正式面试）

"面试官您好，我是 XXX，专注于 LLM 推理优化方向。

**技术深度方面**：
- 我系统性地研究了 Transformer 推理优化的全栈技术，从 Attention 层（MHA/GQA/MLA、FlashAttention、RoPE）到系统层（Continuous Batching、KV Cache 管理、Prefix Caching）
- 特别在 KV Cache 方向有深入实践：实现了 Paged 分配、LRU/LFU 驱逐策略、多租户公平调度、以及自适应策略切换
- 对 vLLM、SGLang 等推理框架的核心流程有源码级理解

**前沿追踪方面**：
- 紧跟 2025 年最新进展：DeepSeek-V3 的 MLA 注意力压缩、R1 的 GRPO 推理训练、MiniMax 的 Lightning Attention
- 对比过 Qwen、DeepSeek、GLM、Kimi 等国产模型的架构差异和 serving 特点

**我的优势**是能将数学原理和工程实现打通。比如 FlashAttention 我不只知道 tiling 的原理，还能分析 SRAM 和 HBM 的 I/O 复杂度；KV Cache 驱逐策略我不只知道 LRU，还实现了基于 Jain's fairness 的多租户公平调度。

期待能为贵公司的推理服务带来性能和成本上的提升。"

---

## 三、项目描述 STAR 模板

### 项目一：KV Cache 多租户公平调度

**Situation（背景）**：
"在多租户 LLM serving 场景下，不同租户的请求频率和 prefix 热度差异很大，传统 LRU 驱逐策略会导致高频租户挤占低频租户的缓存空间。"

**Task（任务）**：
"设计并实现一个公平的 KV Cache 驱逐策略，在保证整体命中率的前提下，提升租户间的公平性。"

**Action（行动）**：
1. 首先量化了问题：用 Zipf 分布模拟真实流量，用 Jain's Fairness Index 衡量公平性
2. 实现了 Weighted Fair Eviction：按租户配额分配缓存块，优先从超配租户驱逐
3. 进一步实现了 Adaptive Strategy：基于滑动窗口命中率和不公平度，动态切换 LRU/LFU/Fair 策略
4. 增加了冷却期机制防止策略抖动

**Result（结果）**：
"Fair 策略将 Jain's Fairness Index 从 LRU 的 0.72 提升到 0.91。Adaptive 策略在综合场景下同时保证了命中率（+3%）和公平性（+15%）。"

### 项目二：Speculative Decoding 实验

**Situation**：
"LLM 的 autoregressive decode 是 memory-bound 的（GPU 利用率低），每步只生成一个 token。"

**Task**：
"实验 Speculative Decoding 的加速效果，分析不同 draft model 大小和 accept rate 对性能的影响。"

**Action**：
1. 实现了 draft-verify 框架：小模型生成 K 个候选 token，大模型一次 forward 验证
2. 分析了 acceptance rate 和 speedup 的关系
3. 实验了不同 K 值（2-8）和不同 draft model size 的组合

**Result**：
"在 K=5、acceptance rate 0.7 的场景下，实现了约 2.3x 的 decode 加速。"

---

## 四、反问面试官的问题

### 技术类
1. "团队目前主要用的推理框架是什么？有做过定制开发吗？"
2. "推理服务的 SLO 是怎么设定的？主要关注 TTFT 还是 TPOT？"
3. "目前模型规模大概在什么量级？有用 MoE 还是 Dense？"
4. "KV Cache 管理有遇到什么挑战吗？比如长上下文或多租户场景？"

### 业务类
5. "这个岗位主要负责的方向是模型训练还是推理优化？"
6. "团队最近半年的技术 roadmap 是什么？"
7. "有考虑过 Speculative Decoding 或 Prefix Caching 的落地吗？"

### 团队类
8. "团队大概多少人？推理和训练是同一个团队吗？"
9. "新人入职后一般多久能上手第一个任务？"
10. "团队的技术分享文化怎么样？有内部 tech talk 吗？"

---

## 五、压力面 / 挑战性问题话术

**Q: "你没有生产环境经验，怎么证明你能做好？"**
A: "虽然我还没有大规模生产环境经验，但我的准备方式是系统性的：
1. 我从数学原理出发理解每个技术（不只是会用框架）
2. 我实现了完整的模拟系统（KV Cache、调度器、量化器）来验证理论
3. 我关注了生产环境的故障模式（OOM、latency spike、fairness 问题）
这种从原理到实践的能力可以让我快速上手生产系统。"

**Q: "这些你都是网上学的，有什么自己的 insight？"**
A: "举个例子：在 KV Cache 多租户实验中，我最初用标准的 Fair 策略（按租户均分配额），发现 fairness 反而没有提升。分析后发现原因是高频租户的 prefix 天然更热，均分配额反而浪费了低频租户的缓存空间。于是我改成了按请求比例加权的公平策略，结果才真正提升了 fairness。这种发现只能通过动手实验获得。"
