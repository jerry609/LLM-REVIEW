# Prefill/Decode 延迟与吞吐模型

## 1) 延迟分解
- `TTFT = T_queue + T_prefill + T_first_decode`
- `E2E = TTFT + N_out * TPOT`
- `T_prefill ≈ FLOPs_prefill / GPU_throughput`（compute-bound 近似）
- `TPOT ≈ bytes_read / BW`（memory-bound 近似，bytes_read 包括模型权重 + KV cache）

## 2) 吞吐近似
- token 吞吐：`Throughput_tok ≈ active_batch / TPOT`
- 请求吞吐与输出长度分布强相关，不宜只给单值。
- Goodput（有效吞吐）：只计算满足 SLO 的请求的吞吐

## 3) 资源视角
- Prefill：目标是缩短 `T_prefill`（提升首 token 体验）。
- Decode：目标是稳定 `TPOT` 与 P99（减少抖动）。

## 4) 批处理权衡
- 更大 decode batch → 更高吞吐，但 TTFT 可能恶化。
- 需要设"新请求准入配额"，防止 prefill 饥饿。

## 5) Continuous Batching（连续批处理）
- 传统 static batching：一批请求中最长的完成后才接受新请求
- Continuous batching：任意请求完成后立即替换为新请求
  - 吞吐提升可达 2-5× 以上
  - iteration-level scheduling：每个 decode step 独立调度
- 核心实现：维护 running pool 和 waiting queue
  - 每步检查：是否有请求完成（移出）+ 是否有空间（加入）

## 6) Chunked Prefill（分块预填充）
- 将长 prompt 切分为固定大小的 chunk（如 512 token）
- 每个 chunk 作为一个"迭代"与 decode 请求交替执行
- 优势：
  - 防止长 prompt prefill 阻塞 decode 请求
  - 保持 TPOT 稳定
  - `TTFT` 增加但更可控
- `chunk_size` 选择：越小 → decode 抖动越小，但 prefill 效率越低

## 7) Prefill-Decode 分离（Disaggregation）
- 将 prefill 和 decode 放在不同 GPU/实例上
- Prefill 实例：compute-bound，高利用率
- Decode 实例：memory-bound，需要高带宽
- 中间需传输 KV cache：`transfer_size = bytes_per_token * T_input`
- 适用场景：长 prompt + 短输出的场景（如 RAG）

## 8) 压测建议
- 固定输入长度分桶：短/中/长上下文。
- 同时观察 `TTFT, TPOT, P95, P99, tokens/s, OOM率`。
- 关注 Goodput（满足 SLO 约束的有效吞吐），而非只看裸吞吐。
- 建议测试不同并发压力下的 latency-throughput 曲线。
