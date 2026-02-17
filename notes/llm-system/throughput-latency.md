## Throughput vs Latency（推理系统速记）

### 1) 定义
- `Latency`：单请求完成时间（常看 TTFT、TPOT、E2E）。
- `Throughput`：单位时间完成的 token 或请求数量（tokens/s, req/s）。
- `Goodput`：满足 SLO 的有效吞吐（超时/失败不算）。

### 2) 三个核心指标
- **TTFT**（Time To First Token）：从请求进入到第一个 token 输出。
- **TPOT**（Time Per Output Token）：平均每个生成 token 的耗时。
- **E2E Latency**：完整响应时间，约等于 `TTFT + output_len * TPOT`。

### 3) Prefill 与 Decode 的瓶颈差异
- **Prefill**：常 compute-bound，受算力和 kernel 效率影响更大。
- **Decode**：常 memory-bound，受 KV 访存和带宽影响更大。
- 这也是为什么优化策略通常是：Prefill 看算子，Decode 看缓存和调度。

### 4) 调参直觉
- 提高 batch size：通常吞吐上升，但尾延迟和 OOM 风险上升。
- 开启 continuous batching：吞吐改善明显，但需防止长请求饿死短请求。
- 做 prefill/decode 分离：可降低相互干扰，但引入跨节点传输成本。

### 5) 面试回答模板（30 秒）
1. 先定义指标（TTFT/TPOT/吞吐/Goodput）。
2. 再分瓶颈（Prefill compute-bound，Decode memory-bound）。
3. 最后讲取舍（批量、调度、缓存、SLO 与成本）。

### 6) 复盘清单
- 这次优化主要改善了哪个指标？（TTFT / TPOT / 吞吐 / P99）
- 代价是什么？（显存、质量、复杂度、稳定性）
- 是否满足业务 SLO？是否有回滚阈值？
