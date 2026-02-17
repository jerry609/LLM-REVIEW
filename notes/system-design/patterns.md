# LLM Serving 常见设计模式

## Pattern 1: 分级存储 (Tiered Storage)
```
Hot: GPU HBM (bf16/fp8) — 活跃请求的 KV
Warm: GPU HBM (int4) — 量化后的缓存前缀
Cold: CPU RAM — Offload 的冷 KV
Evicted: 重算 — 缓存未命中时重新 prefill
```

## Pattern 2: Admission Control
- 当系统负载过高时拒绝新请求（返回 429），而不是让所有请求超时
- 信号：KV 块占用率 > 90%、排队深度 > 阈值、P99 TPOT 超 SLO
- 实现：token bucket rate limiter + backpressure

## Pattern 3: Graceful Degradation
- 层级降级：全精度 → 量化 → 短上下文 → 小模型 → 缓存回复
- 例：负载 >80% 时自动切 fp8 KV；>95% 时截断上下文到 4K

## Pattern 4: Request Scheduling
```
优先级：
1. Decode 请求（用户在等 next token）
2. Chunked Prefill 片段
3. 新 Prefill 请求

排序：
- 同优先级内按 arrival time (FCFS)
- 或按 shortest-job-first（短上下文优先，减少 P99）
```

## Pattern 5: Model Routing
- 多模型部署：简单问题路由到小模型，复杂问题路由到大模型
- 路由方式：① 关键词规则；② 小分类器；③ 先试小模型，质量不够升级
- 成本节省：70-80% 请求用小模型 → 总成本降 50%+

## Pattern 6: Prefix Cache Warming
- 预热常见前缀（system prompt）到缓存
- 在服务启动或冷启动后主动发送预热请求
- 避免前 N 个请求全部 cache miss

## 面试一句话
- "好的 LLM serving 系统不只是推理快，更要有 admission control、graceful degradation 和多模型路由。"
