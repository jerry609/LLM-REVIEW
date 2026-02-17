# 面试题：设计一个 LLM Serving 系统

## 题目
"请设计一个服务 Llama3-70B 的推理系统，支持 1000 QPS，P99 TTFT < 2s，P99 TPOT < 50ms。"

## 答题框架

### 1. 需求确认
- 模型：Llama3-70B, GQA, 128K context
- 流量：1000 QPS, 峰值 2000 QPS
- SLO：P99 TTFT < 2s, P99 TPOT < 50ms
- 输入：平均 1K token, 输出：平均 200 token

### 2. 架构
```
Client → LB → API Gateway → Request Queue
         → Scheduler (continuous batching, decode-first)
         → GPU Worker Pool (TP=8 per node)
         → KV Cache Manager (PagedAttention + Prefix Cache)
         → Response Streamer → Client
```

### 3. 容量规划
- 权重显存：70B INT8 → 70 GB → TP=8 单 node (8×80GB H100)
- 每请求 KV: ~320KB/token × 1K ≈ 320 MB
- 可用 KV 空间: (640-70) GB ≈ 570 GB → ~1780 并发请求
- 实测吞吐 ~40 QPS/node → 需要 ~25 nodes = 200 H100
- 加 30% buffer → 260 H100

### 4. 优化手段
- FP8 KV 量化 → KV 占用减半 → batch 加大
- 前缀缓存 → 重复 system prompt 命中 → TTFT 降低
- Chunked Prefill → 控制 P99 TPOT
- Admission Control → 超载时 reject 而非 timeout

### 5. 监控与告警
- 核心指标：TTFT P99, TPOT P99, QPS, KV 使用率, 队列深度
- 告警条件：KV > 90%, 队列 > 100, TPOT P99 > 40ms
- 自动扩容触发：连续 5min 负载 > 70%

## 追问准备
1. "上下文变 128K 怎么办？" → Ring Attention + KV 量化 + 稀疏化
2. "怎么保证多租户公平？" → 配额驱逐 + Rate Limiting
3. "成本太高怎么降？" → 投机解码 + 模型蒸馏 + Spot Instance
