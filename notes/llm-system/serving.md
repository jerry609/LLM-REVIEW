# LLM 推理服务核心机制

## Continuous Batching
- 传统 static batching：一批请求全部完成后才开始下一批 → 短请求等长请求
- Continuous batching：请求完成一个就腾出一个 slot，新请求立即填入
- 效果：吞吐提升 2-10×（取决于请求长度方差）

## Chunked Prefill
- 问题：长 prefill 阻塞所有 decode slot → TPOT 飙升
- 方案：将 prefill 切成 chunk（如 512 token/chunk），与 decode 交替执行
- 权衡：prefill 总时间略增（多次调度开销），但 decode 不被阻塞

## Decode 优先调度
- 策略：decode 请求优先占用 GPU（延迟敏感），剩余算力给 prefill
- 原因：decode 每步只生成 1 token，但用户在等；prefill 可以容忍更高延迟

## Prefill-Decode 分离（P/D Disaggregation）
- Splitwise / DistServe 思路：prefill 和 decode 分到不同 GPU
- Prefill GPU：compute-bound，追求大 batch + 高 MFU
- Decode GPU：memory-bound，追求低延迟 + 高带宽
- 挑战：KV 迁移带宽（P→D 传输 KV）

## 关键服务指标
| 指标 | 含义 | 目标方向 |
|------|------|---------|
| TTFT | Time To First Token | ↓ 越低越好 |
| TPOT | Time Per Output Token | ↓ 越低越好 |
| Throughput | tokens/s (系统级) | ↑ 越高越好 |
| Goodput | 满足 SLO 的有效吞吐 | ↑ 越高越好 |
| P99 TPOT | 尾延迟 | ↓ 尾巴要短 |

## 面试一句话
- "推理服务的核心矛盾是延迟 vs 吞吐。Continuous Batching + Decode 优先调度是基线，Chunked Prefill 控制尾延迟，P/D 分离是终极方案。"
