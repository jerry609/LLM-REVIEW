# STAR 故事精选 —— KV Cache & Serving 深度打磨版

> 面试时不要背诵，用自然语言讲述，数据要记牢。

---

## Story 1: KV Cache 内存碎片优化（最强故事）

### Situation
我们的 LLM 推理服务（Llama-70B，8×A100）上线后，随着并发量增长到 50+ QPS，
频繁出现 OOM crash。日志显示 GPU 显存利用率只有 65% 就开始 OOM，
说明大量显存被碎片化占用。P99 延迟从 80ms 飙升到 500ms+，影响核心客户。

### Task
作为推理系统负责人，我需要在 **2 周内**将 OOM 率降为 0，P99 TPOT 降到 50ms 以下，
同时不牺牲吞吐量。

### Action
**第 1 步：诊断**
- 用 Nsight Systems 做 GPU memory profiling，发现连续分配模式下 KV Cache 碎片率 60%
- 分析原因：不同请求的序列长度差异大（128~4096），连续分配导致大量不可用空洞

**第 2 步：实现 PagedAttention**
- 参考 vLLM 的设计，将 KV Cache 切成固定大小的 block（block_size=16 tokens）
- 实现 Block Table 映射：逻辑连续 → 物理分散，类似操作系统虚拟内存
- 支持 Copy-on-Write：beam search 时多个候选共享前缀 blocks

**第 3 步：Chunked Prefill**
- 将长 prompt 的 prefill 分成 chunk（chunk_size=512），每步只处理一个 chunk
- 防止单个长请求独占 GPU 导致其他请求的 decode 被阻塞
- 实现 decode-first 调度：每步先处理所有 decode token，再用剩余 budget 做 prefill

**第 4 步：FP8 KV 量化**
- 对 KV Cache 做 per-channel FP8 量化（E4M3 格式）
- 误差分析：在困难度量 benchmark 上 perplexity 仅增加 0.02（可接受）

### Result
| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| OOM 率 | 5 次/天 | **0** |
| P99 TPOT | 500ms | **35ms** |
| 吞吐量 | 30 QPS | **70 QPS** (+133%) |
| 显存利用率 | 65% | **92%** |
| KV Cache 碎片率 | 60% | **<3%** |

**亮点**：这个方案后来被推广到公司所有 LLM 服务，成为内部标准架构。

### 追问准备
- **Q: 为什么选 block_size=16？**
  A: 实验过 8/16/32。block 太小管理开销大（Block Table 膨胀），太大浪费空间。16 在碎片率和管理成本间最优。
  
- **Q: Chunked Prefill 会增加首 token 延迟(TTFT)吗？**
  A: 会，从 50ms 增加到 ~70ms。但这是有意的 trade-off：牺牲 10% TTFT 换取整体 P99 下降 85%。

- **Q: FP8 量化的误差怎么保证？**
  A: 两层保障：(1) per-channel scale 比 per-tensor 精度高；(2) 关键层保留 FP16（attention output layer）。上线前跑了 3 个 benchmark 验证。

---

## Story 2: 多租户公平调度系统

### Situation
我们的 LLM 平台服务 20+ 内部团队（租户），其中 3 个大客户占 85% 流量。
小客户反馈：高峰期首 token 延迟(TTFT) 退化 5 倍以上（从 200ms → 1000ms+），
严重影响他们的产品体验。

### Task
在**不影响大客户 SLO** 的前提下，确保所有租户 TTFT < 500ms（P99），
并且建立长期可持续的公平性保障机制。

### Action
**第 1 步：根因分析**
- 监控发现：大客户的长 prompt 请求（avg 3000 tokens）占满了 KV Cache 容量
- 小客户的请求被频繁 preempt（swap 到 CPU），导致需要重新 prefill
- Jain's fairness index 仅 0.65（理想值 1.0）

**第 2 步：配额驱逐 (Quota-Aware Eviction)**
- 每个租户按 SLA 等级分配 KV Cache 配额（权重比例）
- 驱逐时优先从超配租户中选择 LRU 条目
- 实现窗口自适应策略：根据近 5 分钟命中率和不公平度，在 LRU/LFU/Fair 间自动切换

**第 3 步：Weighted Fair Queuing**
- 请求队列改为 WFQ：每个租户有独立的虚拟时钟
- 权重 = SLA 等级 × 请求 token 数的倒数（短请求优先 + SLA 优先）
- 添加 rate limiting：硬上限防止单个租户独占所有 GPU

**第 4 步：实时监控面板**
- 开发 Grafana dashboard：实时显示每个租户的 hit rate / TTFT / Jain index
- 设置告警：Jain index < 0.8 自动触发调度策略调整

### Result
| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| Jain's fairness index | 0.65 | **0.95** |
| 小客户 P99 TTFT | 1000ms+ | **350ms** |
| 大客户 P99 TTFT | 300ms | **320ms** (+7%, 可接受) |
| 全平台 SLO 达标率 | 72% | **99.5%** |

### 追问准备
- **Q: 为什么选 Jain's fairness index？**
  A: Jain index 对分布偏斜敏感，一个租户极差就会大幅拉低指标，比简单的 max/min ratio 更全面。公式 = (Σhi)² / (n × Σhi²)，取值 [0,1]。

- **Q: 大客户会不会反对？**
  A: 提前与 PM 沟通，设计 SLA 分级。大客户付费更高 → 权重更大 → 保证更高 cache 配额。实测大客户 TTFT 只增加 7%，在 SLO 内。

---

## Story 3: 投机解码 vs 蒸馏的技术决策

### Situation
公司希望将 LLM 推理成本降低 50%，有两个技术方案：
- 方案 A：Speculative Decoding（用 1.3B draft model 加速 70B target model）
- 方案 B：直接蒸馏一个 7B 模型替代 70B

两个方案各有支持者，团队内部分歧严重。

### Task
作为技术 lead，我需要设计 A/B 测试框架，在 1 周内给出数据驱动的推荐。

### Action
**第 1 步：定义评估维度**
- 延迟 (TPOT P50/P99)
- 吞吐 (tokens/sec/GPU)
- 质量 (人工评测 + 自动指标)
- 成本 ($/M tokens)

**第 2 步：实验**
- 投机解码：draft model acceptance rate ~70%，decode 加速 1.8×
  - 问题：额外显存 2GB 放 draft model，但对 70B 来说不大
- 蒸馏 7B：推理成本降 70%
  - 问题：代码生成任务 pass@1 从 68% 降到 52%（-16pp）

**第 3 步：场景化推荐**
- 对话/闲聊：蒸馏 7B（质量差异感知度低）
- 代码生成：投机解码 70B（质量不可妥协）
- 长文档摘要：蒸馏 + 质量检查器（兼顾成本和质量）

### Result
- 混合方案总成本降低 55%（超目标）
- 质量退化 < 0.5%（加权平均）
- 方案被采纳为公司标准推理策略

### 追问准备
- **Q: 投机解码的 acceptance rate 怎么提升？**
  A: 三个方向：(1) 用 MoE 做 draft model，质量更高；(2) 多 token 预测（Medusa）；(3) 知识蒸馏让 draft 更接近 target。

- **Q: 蒸馏质量差异用什么指标衡量？**
  A: 代码用 pass@k + HumanEval，对话用 GPT-4 打分 + 人工盲测 win-rate，摘要用 ROUGE + 事实一致性。

---

## Story 4: 紧急线上事故处理

### Situation
周五晚上 11 点收到告警：推理服务吞吐量突然降到正常的 30%。
值班期间发现模型团队当天下午更新了模型权重（v3.2 → v3.3）。

### Task
60 分钟内恢复服务，并找到根因防止再次发生。

### Action
**第 1 步：紧急回滚** (10分钟)
- 立即回滚到 v3.2 权重，吞吐恢复正常
- 同时通知模型团队和 PM

**第 2 步：根因分析** (30分钟)
- diff 模型配置发现：v3.3 的 `n_kv_heads` 从 8 改为 32
- 这导致 GQA ratio 从 4 降到 1（等于 MHA），KV Cache 大小翻 4 倍
- 显存不足 → 频繁 swap → 吞吐暴跌
- 确认模型团队误用了训练配置（训练时用 MHA 做消融实验）

**第 3 步：防御措施**
- 建立 CI 流水线：模型上线前自动跑推理 benchmark
  - 检查项：KV Cache 大小、TPOT、吞吐量、显存占用
  - 任一指标退化 >10% 则阻断上线
- 制定模型-推理接口规范：n_kv_heads、hidden_dim、n_layers 等必须在 config 中声明
- 模型配置变更需要推理团队 code review

### Result
- 从告警到恢复：35 分钟（SLA 要求 60 分钟）
- CI 流水线上线后，0 次因模型配置导致的推理事故（6 个月）
- 规范文档被公司全部 ML 团队采用

### 追问准备
- **Q: 如果回滚也不行怎么办？**
  A: 降级方案：切到备用的轻量模型（7B），保证服务可用但质量降低，同时并行排查。

- **Q: CI 流水线跑多久？**
  A: 完整 benchmark 15 分钟，卡在推理性能测试。用固定 prompt 集（1000 条），覆盖短/中/长序列。

---

## 通用追问清单

| 问题 | 对应故事 | 关键点 |
|------|---------|--------|
| "最大的技术挑战？" | Story 1 | 碎片化问题一开始被误判为硬件问题 |
| "如何处理团队分歧？" | Story 3 | 用数据说话，设计 A/B 测试 |
| "紧急情况处理？" | Story 4 | 先回滚再排查，15分钟内决策 |
| "跨团队协作？" | Story 2/4 | 提前沟通 SLA，建立规范 |
| "最自豪的成就？" | Story 1 | 方案被全公司采用 |
| "失败经历？" | Story 1 | 最初 3 天误判为硬件问题 |
