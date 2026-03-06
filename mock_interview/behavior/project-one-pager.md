# 项目一页纸 —— 面试快速展示模板

> 用 1 分钟让面试官理解你做了什么、为什么重要、结果如何。

---

## 🔖 模板结构

```
项目名称: xxxxxxxxxx
时间: xxxx.xx - xxxx.xx
角色: xxxx
团队: x 人

【一句话】解决了什么问题，带来了什么价值

【背景】为什么要做这个项目
【方案】核心技术选型和架构
【挑战】最难的技术问题是什么
【结果】量化数据
```

---

## 📄 项目一页纸 #1：高性能 LLM 推理服务

```
项目: LLM 推理服务性能优化
时间: 2024.03 - 2024.09
角色: 推理系统 Tech Lead
团队: 4 人（2 后端 + 1 GPU 工程师 + 1 SRE）
```

### 一句话
通过 PagedAttention + Continuous Batching + FP8 量化，将 70B 模型推理吞吐提升 2.3×，P99 延迟降低 85%。

### 背景
公司核心 AI 产品使用 Llama-70B 提供对话服务。随着用户增长到 100K DAU，推理服务频繁 OOM、延迟飙升，影响用户留存。

### 核心技术选型

| 组件 | 技术 | 选型理由 |
|------|------|---------|
| **KV Cache** | PagedAttention (block_size=16) | 碎片率从 60% 降到 <3% |
| **调度** | Continuous Batching + Chunked Prefill | 消除 head-of-line blocking |
| **量化** | FP8 (E4M3) per-channel | 容量翻倍，质量损失 <0.1% |
| **并行** | TP=8 (8×A100 per node) | 70B 单节点放下 |
| **框架** | 基于 vLLM 二次开发 | 社区活跃，PagedAttention 成熟 |

### 架构图

```
Client → Gateway (Rate Limit)
       → Router (Consistent Hash + WFQ)
       → Engine Pool (50 nodes × 8×A100)
         ├── Scheduler (decode-first + chunked prefill)
         ├── Worker (TP=8, NCCL)
         └── Block Manager (Paged KV Cache + FP8)
       → Monitoring (Prometheus + Grafana)
```

### 最大挑战

**挑战 1: KV Cache 碎片化**
- 问题：不同序列长度导致连续分配碎片率 60%
- 方案：PagedAttention，block 粒度分配
- 关键细节：block_size=16 是实验出来的最优值（8 太碎，32 浪费）

**挑战 2: 多租户不公平**
- 问题：大客户挤占缓存，小客户 TTFT 退化 5×
- 方案：Quota-Aware Eviction + Weighted Fair Queuing
- 关键指标：Jain fairness index 从 0.65 → 0.95

**挑战 3: 模型更新引发事故**
- 问题：模型配置变更导致 KV Cache 翻倍，吞吐暴跌
- 方案：CI 流水线自动 benchmark + 接口规范

### 量化结果

| 指标 | Before | After | 提升 |
|------|--------|-------|------|
| 吞吐量 | 30 QPS | 70 QPS | **+133%** |
| P99 TPOT | 500ms | 35ms | **-93%** |
| OOM 频率 | 5次/天 | 0次 | **-100%** |
| GPU 利用率 | 55% | 85% | **+30pp** |
| 成本/token | USD 0.04 / 1K | USD 0.017 / 1K | **-57%** |
| SLO 达标率 | 72% | 99.5% | **+27.5pp** |

### 我的贡献
- 主导整体架构设计和技术选型
- 亲手实现 PagedAttention 和 Quota-Aware Eviction
- 设计并推广 CI benchmark 流水线（全公司采用）
- 产出 2 篇内部技术文档（30+ 人阅读）

---

## 📄 项目一页纸 #2：RAG + Prefix Caching 系统

```
项目: RAG 服务 Prefix 缓存优化
时间: 2024.06 - 2024.08
角色: 核心开发
团队: 3 人
```

### 一句话
通过 Radix Tree + Consistent Hashing 实现跨请求 prefix KV Cache 复用，RAG 服务 TTFT 降低 60%，GPU 成本降低 35%。

### 背景
RAG 应用中，每个请求都会拼接 5-10 个 retrieved chunk 作为 context。不同请求可能命中相同的 chunk，但每次都重新 prefill，浪费大量计算。

### 核心技术

| 组件 | 技术 | 效果 |
|------|------|------|
| **Prefix 管理** | Radix Tree | 自动合并公共前缀 |
| **请求路由** | Consistent Hashing | prefix-aware，最大化缓存命中 |
| **缓存驱逐** | LRU + 引用计数 | 热点 prefix 常驻 |
| **多轮复用** | Token-level prefix match | 多轮对话历史复用 KV |

### 量化结果

| 指标 | Before | After |
|------|--------|-------|
| TTFT (avg) | 800ms | **320ms** (-60%) |
| Prefix cache hit rate | 0% | **72%** |
| GPU 计算节省 | - | **35%** |
| 每请求平均 prefill tokens | 3000 | **850** |

---

## 📄 项目一页纸 #3：投机解码上线

```
项目: Speculative Decoding 生产化
时间: 2024.09 - 2024.11
角色: 核心开发
团队: 2 人
```

### 一句话
在生产环境上线 Speculative Decoding，无损加速 1.8×，每月节省约 $12K GPU 成本。

### 核心技术

| 维度 | 设计 |
|------|------|
| **Draft Model** | Llama-3.2-1B（与 target 同族，acceptance 高） |
| **验证方式** | 并行 verify + rejection sampling |
| **Spec Length** | 动态调整（根据 acceptance rate 自适应） |
| **显存管理** | Draft 和 Target 共享 KV Cache blocks |

### 最大挑战
- Draft model 质量不够 → acceptance rate 只有 55%
- 解决：用 target 模型对 draft 做知识蒸馏，acceptance rate 提升到 72%

### 量化结果

| 指标 | Before | After |
|------|--------|-------|
| TPOT (avg) | 45ms | **25ms** (-44%) |
| 等效吞吐 | 40 QPS | **72 QPS** (+80%) |
| 质量 | baseline | **完全无损** (数学保证) |
| 月度 GPU 成本 | USD 36K | **USD 24K** (-33%) |

---

## 🎤 讲述技巧

### 1 分钟版（电梯演讲）
> "我负责公司 70B 模型的推理优化。主要做了三件事：
> 第一，用 PagedAttention 解决了 KV Cache 碎片化导致的 OOM 问题；
> 第二，设计了多租户公平调度系统，Jain index 从 0.65 提到 0.95；
> 第三，推动了全公司的模型上线 CI 流水线。
> 最终吞吐提升 2.3 倍，P99 延迟降低 85%。"

### 3 分钟版（技术面试）
1 分钟版 + 展开最有深度的一个挑战（KV Cache 碎片化），用 STAR 讲

### 5 分钟版（项目介绍）
3 分钟版 + 第二个挑战（多租户）+ 成本优化数据

### 追问应对
- "最难的部分是什么？" → KV Cache 碎片化最初被误判
- "你和别人不一样的地方？" → 用 Jain fairness index 量化公平性（大多数人只看 P99）
- "如果重来会怎么做？" → 一开始就建 CI 流水线，不是出了事故才建
- "有什么遗憾？" → 没有做 disaggregated prefill-decode，这是下一步计划
