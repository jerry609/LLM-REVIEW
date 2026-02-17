# 系统设计面试方法论（LLM Serving 专用）

## 四步框架

### Step 1: 需求明确（2-3 分钟）
- 功能需求：单模型 / 多模型？对话 / 补全 / embedding？流式 / 非流式？
- 非功能需求：QPS、延迟 SLO (P99 TTFT, P99 TPOT)、可用性、成本预算
- 规模：模型大小、上下文长度、用户量、峰值倍数

### Step 2: 架构设计（10-15 分钟）
```
Client → Load Balancer → API Gateway → Router
                                          ↓
                                    Scheduler (continuous batching)
                                          ↓
                                    GPU Worker Pool (TP/PP)
                                          ↓
                                    KV Cache Manager (PagedAttention)
```
- 重点讲：Scheduler 如何 continuous batching、KV Cache 如何分页管理
- 并行策略：机内 TP + 跨机 PP

### Step 3: 深入组件（10 分钟）
面试官会追问 1-2 个组件：
- **KV Cache 管理**：分页、驱逐（LRU/LFU/Fair）、前缀缓存、量化
- **调度器**：decode 优先、chunked prefill、admission control
- **容量规划**：显存模型、吞吐估算、GPU 数量

### Step 4: 权衡与扩展（5 分钟）
- 权衡讨论：延迟 vs 吞吐、精度 vs 成本
- 扩展方向：P/D 分离、投机解码、弹性伸缩

## 常见追问
1. "QPS 翻 10 倍怎么办？" → 弹性伸缩 + 蒸馏 + 前缀缓存
2. "上下文从 4K 变 128K 怎么办？" → KV 量化 + 稀疏化 + Ring Attention
3. "怎么保证公平性？" → 多租户配额 + Fair 驱逐 + Rate Limiting

## 面试一句话
- "LLM serving 系统设计的核心矛盾是显存有限 vs 长上下文需求，解法是分页管理 + 动态调度 + 分级存储。"
