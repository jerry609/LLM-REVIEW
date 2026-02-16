# 推理服务指标速查

## 核心指标
- `TTFT`：请求到首 token 返回的时间（Time To First Token）
- `TPOT`：平均每输出 token 时间（Time Per Output Token）
- `ITL`：token 间延迟（Inter-Token Latency，与 TPOT 口径接近）
- `Throughput`：tokens/s 或 req/s（必须说明口径）
- `P95/P99`：尾延迟分位数
- `E2E Latency`：端到端延迟 = `TTFT + N_out * TPOT`

## 有效吞吐（Goodput）
- `Goodput = SLO_satisfied_requests / total_time`
- 只统计满足 SLO 约束（如 TTFT < 2s, TPOT < 50ms）的请求
- 区别于裸吞吐：裸吞吐高但大量违反 SLO 无意义

## 衍生指标
- `KV Hit Rate = hits / lookups`（prefix caching 场景）
- `Refill Rate = evicted_then_recomputed / total_evictions`
- `OOM Rate = oom_requests / total_requests`
- `Preemption Rate = preempted_requests / total_requests`
- `Queue Depth = avg_waiting_requests`
- `Batch Utilization = avg_active_batch / max_batch_capacity`
- `GPU Utilization = active_compute_time / total_time`
- `KV Cache Utilization = used_kv_blocks / total_kv_blocks`

## 面试常考指标关系
- `Throughput_tok = active_batch_size / TPOT`
- `max_batch ≈ KV_budget / (bytes_per_token * avg_seq_len)`
- `TTFT ≈ queue_wait + prefill_time`（无 chunked prefill 时）

## 诊断规则
- 命中率升、TPOT差：常见是回迁/反量化抖动。
- 吞吐升、TTFT差：常见是批处理偏 decode，prefill 被挤压。
- P99抖动：优先查驱逐峰值、队列长度、跨卡通信峰值。
- GPU 利用率低但 TPOT 高：memory-bound，考虑增大 batch 或优化 KV 访存。
- 队列深度持续增长：服务容量不足，需扩容或限流。

## 告警阈值（参考）
- KV 利用率 > 90%：准备触发驱逐或拒绝新请求
- P99 TTFT > SLO * 0.8：预警接近 SLO 上限
- OOM Rate > 0.1%：需要调整 batch 策略或 KV 预算
